"""
ALLOY IQ — Parquet Export & Agent Tracker
==========================================
Handles all output operations:

  1. Partitioned .parquet export — one file per alloy family + property domain
     (e.g., processed/steel_mechanical.parquet, processed/hea_thermo.parquet)

  2. Compression — snappy (fast) by default, lz4 optional

  3. agent_tracker.json — updated atomically after each successful write.
     Records: file path, row count, column count, checksum, write timestamp.

  4. All write errors are logged and re-raised as RuntimeError so the
     pipeline's top-level handler can decide to retry or abort.

Usage:
    from backend.ingestion.exporter import ParquetExporter
    exporter = ParquetExporter()
    exporter.export(df_clean)            # auto-partitions by alloy_family
    exporter.export_partition(df, label="steel_mechanical")  # explicit label
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from backend.ingestion.logger import get_logger
from backend.ingestion.schema import MECHANICAL_COLS, THERMO_COLS, CORROSION_COLS

load_dotenv()
log = get_logger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "backend/data/processed"))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

AGENT_TRACKER_PATH = Path(os.getenv("AGENT_TRACKER_PATH", "agent_tracker.json"))
COMPRESSION = "snappy"  # or "lz4", "gzip", "brotli"

# ── Property domain detection ─────────────────────────────────────────────────
_DOMAIN_COLS: Dict[str, List[str]] = {
    "mechanical": MECHANICAL_COLS,
    "thermo":     THERMO_COLS,
    "corrosion":  CORROSION_COLS,
}


def _infer_domain(df: pd.DataFrame) -> str:
    """
    Pick the property domain label based on which columns are most populated.
    Returns 'mechanical', 'thermo', or 'corrosion'.
    """
    best = "mechanical"
    best_count = 0
    for domain, cols in _DOMAIN_COLS.items():
        present = [c for c in cols if c in df.columns]
        count = df[present].notna().sum().sum()
        if count > best_count:
            best_count = count
            best = domain
    return best


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ParquetExporter:
    """
    Writes cleaned DataFrames to partitioned .parquet files and
    maintains the agent_tracker.json manifest.
    """

    def export(self, df: pd.DataFrame) -> List[Path]:
        """
        Auto-partition `df` by 'alloy_family', 'property domain', and 'source_tier',
        then export each partition to its own .parquet file.

        Returns
        -------
        List of Path objects for successfully written files.
        """
        if df.empty:
            log.warning("ParquetExporter.export: received empty DataFrame — nothing written.")
            return []

        written: List[Path] = []

        families = (
            df["alloy_family"].dropna().unique()
            if "alloy_family" in df.columns
            else ["unknown"]
        )
        
        tiers = (
            df["source_tier"].dropna().unique()
            if "source_tier" in df.columns
            else ["tierX"]
        )

        for family in families:
            for tier in tiers:
                mask = pd.Series(True, index=df.index)
                if "alloy_family" in df.columns:
                    mask &= (df["alloy_family"] == family)
                if "source_tier" in df.columns:
                    mask &= (df["source_tier"] == tier)
                    
                partition = df[mask].copy()

                if partition.empty:
                    continue

                domain = _infer_domain(partition)
                label = f"{str(family).lower().replace(' ', '_')}_{domain}_{str(tier).lower()}"

                path = self.export_partition(partition, label=label)
                if path:
                    written.append(path)

        return written

    def export_partition(
        self,
        df: pd.DataFrame,
        label: str,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Write a single DataFrame partition to a .parquet file.

        Parameters
        ----------
        df : pd.DataFrame
            Data to export (should be pre-cleaned).
        label : str
            File stem (e.g. "steel_mechanical").
        output_dir : Path, optional
            Override the default PROCESSED_DIR.

        Returns
        -------
        Path of the written file, or None on failure.
        """
        out_dir = output_dir or PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{label}.parquet"

        try:
            # Cast object columns with mixed types to string to avoid Arrow errors
            df_safe = self._sanitize_for_parquet(df)

            df_safe.to_parquet(
                out_path,
                engine="pyarrow",
                compression=COMPRESSION,
                index=False,
            )

            checksum = _sha256_file(out_path)
            log.info(
                f"Exported → {out_path.name} "
                f"({len(df_safe)} rows, {out_path.stat().st_size / 1024:.1f} KB, "
                f"sha256={checksum[:12]}…)"
            )

            self._update_tracker(
                file_path=out_path,
                label=label,
                row_count=len(df_safe),
                col_count=df_safe.shape[1],
                checksum=checksum,
            )

            return out_path

        except Exception as exc:
            log.error(f"Failed to write {out_path}: {exc}", exc_info=True)
            raise RuntimeError(f"Parquet export failed for label={label}") from exc

    # ── agent_tracker.json ─────────────────────────────────────────────────────

    def _update_tracker(
        self,
        file_path: Path,
        label: str,
        row_count: int,
        col_count: int,
        checksum: str,
    ) -> None:
        """
        Atomically update agent_tracker.json with a new parquet record.
        Uses read-modify-write with a .tmp file to be crash-safe.
        """
        try:
            # Load existing tracker
            if AGENT_TRACKER_PATH.exists():
                with open(AGENT_TRACKER_PATH, "r", encoding="utf-8") as f:
                    tracker: Dict[str, Any] = json.load(f)
            else:
                tracker = {"parquet_files": {}, "last_updated": None}

            tracker.setdefault("parquet_files", {})
            tracker["parquet_files"][label] = {
                "path": str(file_path.resolve()),
                "label": label,
                "row_count": row_count,
                "col_count": col_count,
                "checksum_sha256": checksum,
                "compression": COMPRESSION,
                "written_at": datetime.now(timezone.utc).isoformat(),
            }
            tracker["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Atomic write via temp file
            tmp_path = AGENT_TRACKER_PATH.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(tracker, f, indent=2)
            tmp_path.replace(AGENT_TRACKER_PATH)

            log.info(f"agent_tracker.json updated → label={label}")

        except Exception as exc:
            log.error(f"Failed to update agent_tracker.json: {exc}", exc_info=True)

    # ── Parquet sanitization ──────────────────────────────────────────────────

    @staticmethod
    def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame is Arrow-compatible:
          - object columns with mixed types → string
          - Categorical already supported
          - Remove columns that are entirely empty list / dict objects
        """
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                # Check if any value is list or dict (not serializable to parquet directly)
                sample = df[col].dropna()
                if not sample.empty and isinstance(sample.iloc[0], (list, dict)):
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
                        if x is not None else None
                    )
                else:
                    df[col] = df[col].astype("string")
        return df
