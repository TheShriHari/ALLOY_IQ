"""
ALLOY IQ — Master Data Ingestion Pipeline
==========================================
Orchestrates the full ingestion workflow:

  Step 1 — Pull structured data (Matminer + Materials Project)
  Step 2 — Pull AFLOW thermodynamic data (async REST)
  Step 3 — Extract tabular data from PDFs in data/pdfs/
  Step 4 — Merge all sources + clean (standardize, outlier-flag, deduplicate)
  Step 5 — Export partitioned .parquet files + update agent_tracker.json

Design principles:
  • Each step is independent; failure in one logs and continues.
  • No hardcoded credentials — only .env via python-dotenv.
  • Progress + errors logged to logs/ingestion_errors.log and logs/ingestion_full.log.
  • Script is idempotent: re-running overwrites existing .parquet files.

CLI usage:
    python -m backend.ingestion.pipeline [OPTIONS]

    Options:
      --skip-matminer     Skip Matminer/MP fetch (fast dev mode)
      --skip-aflow        Skip AFLOW fetch
      --skip-pdf          Skip PDF extraction
      --aflow-nmax N      Max AFLOW entries (default 2000)
      --dry-run           Run all steps but don't write parquet files

Programmatic usage:
    from backend.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    result = pipeline.run()
    print(result.summary())
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

from backend.ingestion.aflow_client import AflowClient
from backend.ingestion.oqmd_client import OqmdClient
from backend.ingestion.cleaner import DataCleaner
from backend.ingestion.exporter import ParquetExporter
from backend.ingestion.logger import get_logger
from backend.ingestion.matminer_retriever import MatminerRetriever
from backend.ingestion.pdf_extractor import PDFExtractor
from backend.ingestion.kaggle_loader import KaggleLoader
from backend.ingestion.hea_repos import HEARepositoryLoader
from backend.ingestion.literature_scraper import LiteratureScraper
from backend.ingestion.schema import make_empty_frame

load_dotenv()
log = get_logger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Holds the outcome of a pipeline run for inspection or CI assertions."""
    raw_rows_matminer: int = 0
    raw_rows_aflow: int = 0
    raw_rows_pdf: int = 0
    clean_rows_total: int = 0
    outlier_rows: int = 0
    parquet_files_written: List[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    success: bool = False
    error_message: str = ""

    def summary(self) -> str:
        lines = [
            "═══════════════════════════════════════════════",
            " ALLOY IQ — Ingestion Pipeline Summary",
            "═══════════════════════════════════════════════",
            f"  Matminer rows collected : {self.raw_rows_matminer:,}",
            f"  AFLOW rows collected    : {self.raw_rows_aflow:,}",
            f"  PDF rows collected      : {self.raw_rows_pdf:,}",
            f"  ─────────────────────────────────────────────",
            f"  Clean rows (total)      : {self.clean_rows_total:,}",
            f"  Outlier-flagged rows    : {self.outlier_rows:,}",
            f"  Parquet files written   : {len(self.parquet_files_written)}",
        ]
        for p in self.parquet_files_written:
            lines.append(f"    • {p}")
        lines += [
            f"  Elapsed time            : {self.elapsed_seconds:.1f}s",
            f"  Status                  : {'✓ SUCCESS' if self.success else '✗ FAILED'}",
        ]
        if self.error_message:
            lines.append(f"  Error                   : {self.error_message}")
        lines.append("═══════════════════════════════════════════════")
        return "\n".join(lines)


# ── Pipeline class ────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates all data ingestion steps for ALLOY IQ.
    Each step is wrapped in a try/except; failures are logged
    and the pipeline continues with what it has.
    """

    def __init__(
        self,
        skip_matminer: bool = False,
        skip_aflow: bool = False,
        skip_pdf: bool = False,
        aflow_nmax: int = 2000,
        dry_run: bool = False,
        pdf_dir: Optional[Path] = None,
    ) -> None:
        self.skip_matminer = skip_matminer
        self.skip_aflow = skip_aflow
        self.skip_pdf = skip_pdf
        self.aflow_nmax = aflow_nmax
        self.dry_run = dry_run
        self.pdf_dir = pdf_dir

        self.matminer = MatminerRetriever()
        self.aflow = AflowClient()
        self.oqmd = OqmdClient()
        self.pdf = PDFExtractor()
        self.kaggle = KaggleLoader()
        self.hea_repos = HEARepositoryLoader()
        self.literature = LiteratureScraper()
        self.cleaner = DataCleaner()
        self.exporter = ParquetExporter()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> PipelineResult:
        """
        Execute the full pipeline. Returns a PipelineResult with
        counts and file paths for inspection.
        """
        result = PipelineResult()
        t_start = time.perf_counter()

        log.info("══════════════════════════════════════════")
        log.info(" ALLOY IQ Data Ingestion Pipeline — START ")
        log.info("══════════════════════════════════════════")

        frames: List[pd.DataFrame] = []

        # ── Step 1: Matminer ──────────────────────────────────────────────────
        if not self.skip_matminer:
            matminer_frames = self._step_matminer()
            for df in matminer_frames:
                df["source_tier"] = "tier1"
                result.raw_rows_matminer += len(df)
            frames.extend(matminer_frames)
        else:
            log.info("Step 1 [Matminer] — SKIPPED (--skip-matminer)")

        # ── Step 2: AFLOW & OQMD ─────────────────────────────────────────────
        if not self.skip_aflow:
            df_aflow = self._step_aflow()
            result.raw_rows_aflow = len(df_aflow)
            if not df_aflow.empty:
                df_aflow["source_tier"] = "tier1"
                frames.append(df_aflow)
                
            import asyncio
            df_oqmd = asyncio.run(self.oqmd.fetch_async(limit=1000))
            if not df_oqmd.empty:
                frames.append(df_oqmd)
        else:
            log.info("Step 2 [AFLOW/OQMD] — SKIPPED (--skip-aflow)")

        # ── Step 3: PDF extraction & Literature Scraper ───────────────────────
        if not self.skip_pdf:
            import asyncio
            asyncio.run(self.literature.run_scraping())
            
            df_pdf = self._step_pdf()
            result.raw_rows_pdf = len(df_pdf)
            if not df_pdf.empty:
                df_pdf["source_tier"] = "tier4"
                frames.append(df_pdf)
        else:
            log.info("Step 3 [PDF/Literature] — SKIPPED (--skip-pdf)")
            
        # ── Step 3.5: Kaggle & HEA Repos ──────────────────────────────────────
        df_kaggle = self.kaggle.load_datasets()
        if not df_kaggle.empty:
            frames.append(df_kaggle)
            
        import asyncio
        df_hea = asyncio.run(self.hea_repos.fetch_all())
        if not df_hea.empty:
            frames.append(df_hea)

        # ── Step 4: Merge + Clean ─────────────────────────────────────────────
        if not frames:
            msg = "All data sources returned empty — aborting pipeline."
            log.error(msg)
            result.error_message = msg
            result.elapsed_seconds = time.perf_counter() - t_start
            return result

        log.info("Step 4 — Merging & cleaning all sources …")
        try:
            df_clean = self.cleaner.merge_sources(*frames)
            result.clean_rows_total = len(df_clean)

            if "notes" in df_clean.columns:
                result.outlier_rows = (
                    df_clean["notes"].str.contains("OUTLIER", na=False).sum()
                )

        except Exception as exc:
            log.error(f"Step 4 [Merge/Clean] failed: {exc}", exc_info=True)
            result.error_message = str(exc)
            result.elapsed_seconds = time.perf_counter() - t_start
            return result

        # ── Step 5: Export ────────────────────────────────────────────────────
        if self.dry_run:
            log.info("Step 5 [Export] — SKIPPED (--dry-run mode)")
            log.info(f"  Would have exported {len(df_clean)} rows.")
        else:
            log.info("Step 5 — Exporting .parquet files …")
            try:
                written = self.exporter.export(df_clean)
                result.parquet_files_written = written
            except Exception as exc:
                log.error(f"Step 5 [Export] failed: {exc}", exc_info=True)
                result.error_message = str(exc)
                result.elapsed_seconds = time.perf_counter() - t_start
                return result

        result.success = True
        result.elapsed_seconds = time.perf_counter() - t_start

        log.info(result.summary())
        return result

    # ── Step implementations ──────────────────────────────────────────────────

    def _step_matminer(self) -> List[pd.DataFrame]:
        """Fetch all configured Matminer datasets. Returns list of DataFrames."""
        log.info("Step 1 — Fetching Matminer / Materials Project data …")
        frames: List[pd.DataFrame] = []

        # Steel strength (primary dataset)
        try:
            df = self.matminer.fetch_steel_strength()
            if not df.empty:
                frames.append(df)
                log.info(f"  ✓ steel_strength: {len(df)} rows")
            else:
                log.warning("  ✗ steel_strength returned empty")
        except Exception as exc:
            log.error(f"  Matminer steel_strength failed: {exc}", exc_info=True)

        # JARVIS-DFT (thermodynamic properties)
        try:
            df = self.matminer.fetch_jarvis_dft()
            if not df.empty:
                frames.append(df)
                log.info(f"  ✓ jarvis_dft_3d: {len(df)} rows")
        except Exception as exc:
            log.error(f"  Matminer JARVIS-DFT failed: {exc}", exc_info=True)

        # Materials Project — Fe-Cr-Ni ternary (stainless steel space)
        for chemsys, nmax in [
            ("Fe-Cr-Ni", 3000),
            ("Fe-C", 1000),
            ("Al-Mg-Si", 2000),
            ("Al-Cu", 1500),
            ("Co-Cr-Fe-Mn-Ni", 500),
        ]:
            try:
                df = self.matminer.fetch_mp_composition_properties(
                    chemsys=chemsys, max_rows=nmax
                )
                if not df.empty:
                    frames.append(df)
                    log.info(f"  ✓ MP [{chemsys}]: {len(df)} rows")
            except Exception as exc:
                log.error(f"  MP [{chemsys}] failed: {exc}", exc_info=True)

        return frames

    def _step_aflow(self) -> pd.DataFrame:
        """Fetch AFLOW thermodynamic data. Returns DataFrame or empty."""
        log.info(f"Step 2 — Fetching AFLOW data (nmax={self.aflow_nmax}) …")
        try:
            df = self.aflow.fetch_sync(nmax=self.aflow_nmax)
            if not df.empty:
                log.info(f"  ✓ AFLOW: {len(df)} rows")
            else:
                log.warning("  ✗ AFLOW returned empty")
            return df
        except Exception as exc:
            log.error(f"  AFLOW fetch failed: {exc}", exc_info=True)
            return make_empty_frame()

    def _step_pdf(self) -> pd.DataFrame:
        """Extract tables from all PDFs. Returns merged DataFrame or empty."""
        log.info("Step 3 — Extracting tables from PDFs …")
        try:
            df = self.pdf.process_directory(self.pdf_dir)
            if not df.empty:
                log.info(f"  ✓ PDF: {len(df)} rows from {self.pdf_dir or 'default dir'}")
            else:
                log.info("  ✗ PDF: no tables extracted (no PDFs in directory or all empty)")
            return df
        except Exception as exc:
            log.error(f"  PDF extraction failed: {exc}", exc_info=True)
            return make_empty_frame()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backend.ingestion.pipeline",
        description="ALLOY IQ — Master Data Ingestion Pipeline",
    )
    p.add_argument("--skip-matminer", action="store_true",
                   help="Skip Matminer / Materials Project fetch")
    p.add_argument("--skip-aflow", action="store_true",
                   help="Skip AFLOW REST fetch")
    p.add_argument("--skip-pdf", action="store_true",
                   help="Skip PDF table extraction")
    p.add_argument("--aflow-nmax", type=int, default=2000, metavar="N",
                   help="Max AFLOW entries to fetch (default: 2000)")
    p.add_argument("--dry-run", action="store_true",
                   help="Run all steps but do not write .parquet files")
    p.add_argument("--pdf-dir", type=Path, default=None,
                   help="Override PDF directory (default: backend/data/pdfs)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    pipeline = IngestionPipeline(
        skip_matminer=args.skip_matminer,
        skip_aflow=args.skip_aflow,
        skip_pdf=args.skip_pdf,
        aflow_nmax=args.aflow_nmax,
        dry_run=args.dry_run,
        pdf_dir=args.pdf_dir,
    )

    result = pipeline.run()
    print(result.summary())
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
