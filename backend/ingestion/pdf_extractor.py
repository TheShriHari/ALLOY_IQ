"""
ALLOY IQ — PDF Table Extraction Pipeline
=========================================
Extracts tabular alloy composition / property data from academic PDFs
using a three-tier strategy (best → fallback → fallback):

  Tier 1 (preferred)  — pdfplumber  : camelot-style lattice/stream table detection
  Tier 2 (fallback)   — Nougat      : Meta's OCR-based academic PDF parser
                        (requires nougat-ocr installed + model downloaded)
  Tier 3 (LLM-vision) — Google Gemini Vision API (if GEMINI_API_KEY is set)
                        Identifies and extracts tables containing alloy data

Post-extraction:
  - Column synonym normalization via schema.SYNONYM_MAP
  - Numeric coercion + unit inference (MPa, HV, wt%)
  - Each extracted table is tagged with src_pdf + page_number
  - Failed tables are logged with reason, page, and PDF path

Usage:
    from backend.ingestion.pdf_extractor import PDFExtractor
    extractor = PDFExtractor()
    df = extractor.process_pdf(Path("data/pdfs/duplex_steel_review.pdf"))
    df_all = extractor.process_directory(Path("data/pdfs/"))
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from backend.ingestion.logger import get_logger
from backend.ingestion.schema import (
    ELEMENT_SYMBOLS,
    SYNONYM_MAP,
    make_empty_frame,
    standardize_columns,
)

load_dotenv()
log = get_logger(__name__)

PDF_DIR = Path(os.getenv("PDF_DIR", "backend/data/pdfs"))
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

# Regex to detect alloy-relevant headers (fast filter before LLM call)
_ALLOY_HEADER_RE = re.compile(
    r"\b(yield|tensile|hardness|uts|ys|rp|strength|elongation|"
    r"composition|wt%|at%|weight|alloy|steel|hea|aluminum|mpа|mpa|hv|hrc)\b",
    re.IGNORECASE,
)


class PDFExtractor:
    """
    Three-tier PDF table extractor for metallurgical literature.
    """

    def __init__(
        self,
        use_nougat: bool = True,
        use_llm_vision: bool = True,
    ) -> None:
        self.use_nougat = use_nougat
        self.use_llm_vision = use_llm_vision and bool(GEMINI_API_KEY)

        if self.use_llm_vision and not GEMINI_API_KEY:
            log.warning("GEMINI_API_KEY not set — LLM-vision tier disabled.")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: Path) -> pd.DataFrame:
        """
        Extract all alloy-relevant tables from a single PDF.

        Returns
        -------
        pd.DataFrame with canonical schema columns. Empty frame if nothing found.
        """
        log.info(f"Processing PDF: {pdf_path.name}")
        frames: List[pd.DataFrame] = []

        # Tier 1: pdfplumber
        tier1 = self._extract_pdfplumber(pdf_path)
        frames.extend(tier1)

        if not frames and self.use_nougat:
            # Tier 2: Nougat (only if pdfplumber found nothing)
            log.info(f"  pdfplumber found no tables — trying Nougat for {pdf_path.name}")
            tier2 = self._extract_nougat(pdf_path)
            frames.extend(tier2)

        if not frames and self.use_llm_vision:
            # Tier 3: LLM-vision
            log.info(f"  Nougat found nothing — trying LLM-vision for {pdf_path.name}")
            tier3 = self._extract_llm_vision(pdf_path)
            frames.extend(tier3)

        if not frames:
            log.warning(f"No alloy tables extracted from {pdf_path.name} — PDF skipped.")
            return make_empty_frame()

        combined = pd.concat(frames, ignore_index=True)
        log.info(f"  → {pdf_path.name}: extracted {len(combined)} rows from {len(frames)} tables")
        return combined

    def process_directory(self, directory: Optional[Path] = None) -> pd.DataFrame:
        """
        Process all PDF files in a directory.

        Returns merged DataFrame or empty frame if no PDFs found.
        """
        target = directory or PDF_DIR
        pdfs = sorted(target.glob("*.pdf"))

        if not pdfs:
            log.warning(f"No PDF files found in {target}")
            return make_empty_frame()

        log.info(f"Processing {len(pdfs)} PDFs from {target}")
        all_frames: List[pd.DataFrame] = []
        for pdf in pdfs:
            try:
                df = self.process_pdf(pdf)
                if not df.empty:
                    all_frames.append(df)
            except Exception as exc:
                log.error(f"Unhandled error processing {pdf.name}: {exc}", exc_info=True)

        if not all_frames:
            return make_empty_frame()

        merged = pd.concat(all_frames, ignore_index=True)
        log.info(f"PDF pipeline complete: {len(merged)} total rows from {len(all_frames)} PDFs")
        return merged

    # ── Tier 1: pdfplumber ───────────────────────────────────────────────────

    def _extract_pdfplumber(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Use pdfplumber to extract lattice and stream tables.
        Filters to only tables that contain alloy-relevant headers.
        """
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            log.error("pdfplumber not installed. Run: pip install pdfplumber")
            return []

        frames: List[pd.DataFrame] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    if not tables:
                        continue

                    for tbl_idx, table in enumerate(tables):
                        try:
                            df = self._table_to_df(table)
                            if df is None:
                                continue

                            if not self._is_alloy_table(df):
                                continue

                            df = self._clean_table(df)
                            df["src_name"] = "pdf_pdfplumber"
                            df["src_url"] = str(pdf_path)
                            df["notes"] = f"page={page_num} table_idx={tbl_idx}"
                            frames.append(df)

                        except Exception as exc:
                            log.warning(
                                f"pdfplumber table parse failed "
                                f"[{pdf_path.name} p.{page_num} t.{tbl_idx}]: {exc}"
                            )
        except Exception as exc:
            log.error(f"pdfplumber failed to open {pdf_path.name}: {exc}", exc_info=True)

        return frames

    # ── Tier 2: Nougat (Meta's academic PDF parser) ──────────────────────────

    def _extract_nougat(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Run Nougat CLI to produce MMD (Markdown) from PDF, then parse
        Markdown tables containing alloy data.

        Requires: pip install nougat-ocr
        First run: nougat (downloads weights ~1.6 GB)
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log.debug(f"Running Nougat on {pdf_path.name} …")
                result = subprocess.run(
                    ["nougat", str(pdf_path), "--out", tmpdir, "--no-skipping"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    log.warning(
                        f"Nougat non-zero exit [{pdf_path.name}]: {result.stderr[:300]}"
                    )
                    return []

                mmd_files = list(Path(tmpdir).glob("*.mmd"))
                if not mmd_files:
                    log.warning(f"Nougat produced no .mmd output for {pdf_path.name}")
                    return []

                mmd_text = mmd_files[0].read_text(encoding="utf-8")
                return self._parse_markdown_tables(mmd_text, src_pdf=pdf_path)

        except FileNotFoundError:
            log.warning(
                "Nougat CLI not found — install with: pip install nougat-ocr. "
                "Falling through to next tier."
            )
        except subprocess.TimeoutExpired:
            log.error(f"Nougat timed out on {pdf_path.name} (>300s)")
        except Exception as exc:
            log.error(f"Nougat extraction failed [{pdf_path.name}]: {exc}", exc_info=True)

        return []

    # ── Tier 3: LLM-vision (Google Gemini) ───────────────────────────────────

    def _extract_llm_vision(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Send PDF pages as images to Google Gemini Vision.
        Asks the model to extract alloy composition + property tables as JSON.

        Requires:
          - pip install google-generativeai pymupdf
          - GEMINI_API_KEY in .env
        """
        try:
            import fitz  # PyMuPDF  # type: ignore
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")

        except ImportError as e:
            log.warning(f"LLM-vision deps missing ({e}). Install: pip install google-generativeai pymupdf")
            return []

        frames: List[pd.DataFrame] = []

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            log.error(f"PyMuPDF failed to open {pdf_path.name}: {exc}")
            return []

        PROMPT = (
            "You are a materials science data extraction assistant. "
            "Look at this image of an academic paper page. "
            "Find any table that contains alloy compositions (element wt% or at%) "
            "or mechanical properties (yield strength, tensile strength, hardness, elongation). "
            "If found, output a JSON array of objects where each key is the column header "
            "and each element is a row value. "
            "Use null for missing values. "
            "If no relevant table is present, respond with an empty array []."
        )

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")

                response = model.generate_content([
                    PROMPT,
                    {"mime_type": "image/png", "data": img_bytes},
                ])

                raw = response.text.strip()
                # Strip markdown code fences if present
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

                if not raw or raw == "[]":
                    continue

                import json
                records = json.loads(raw)
                if not isinstance(records, list) or not records:
                    continue

                df = pd.DataFrame(records)
                df = standardize_columns(df)
                df = self._clean_table(df)
                df["src_name"] = "pdf_llm_vision"
                df["src_url"] = str(pdf_path)
                df["notes"] = f"page={page_num + 1} gemini_vision"
                frames.append(df)
                log.info(
                    f"  LLM-vision: extracted {len(df)} rows from "
                    f"{pdf_path.name} p.{page_num + 1}"
                )

            except Exception as exc:
                log.warning(
                    f"LLM-vision failed [{pdf_path.name} p.{page_num + 1}]: {exc}"
                )

        doc.close()
        return frames

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _table_to_df(table: List[List]) -> Optional[pd.DataFrame]:
        """Convert raw pdfplumber table (list of lists) to DataFrame."""
        if not table or len(table) < 2:
            return None
        headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
        rows = table[1:]
        if not rows:
            return None
        return pd.DataFrame(rows, columns=headers)

    @staticmethod
    def _is_alloy_table(df: pd.DataFrame) -> bool:
        """Return True if ANY column header matches alloy-relevant keywords."""
        header_str = " ".join(str(c) for c in df.columns)
        return bool(_ALLOY_HEADER_RE.search(header_str))

    @staticmethod
    def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and coerce numeric columns.
        Strip units from values (e.g. '850 MPa' → 850.0).
        """
        df = standardize_columns(df)

        # Strip inline units and convert to float where possible
        for col in df.columns:
            if df[col].dtype == object:
                cleaned = (
                    df[col]
                    .astype(str)
                    .str.extract(r"([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)", expand=False)
                )
                coerced = pd.to_numeric(cleaned, errors="coerce")
                # Only replace if >50% of values are numeric
                if coerced.notna().mean() > 0.5:
                    df[col] = coerced

        return df

    def _parse_markdown_tables(
        self, mmd_text: str, src_pdf: Path
    ) -> List[pd.DataFrame]:
        """
        Parse Nougat's MMD (Markdown) output for pipe-delimited tables.
        """
        frames: List[pd.DataFrame] = []
        table_blocks = re.findall(
            r"(\|.+?\|(?:\n\|.+?\|)+)", mmd_text, re.DOTALL
        )

        for block in table_blocks:
            try:
                lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
                # Filter separator rows (|---|---|)
                lines = [l for l in lines if not re.match(r"^\|[-:\s|]+\|$", l)]
                if len(lines) < 2:
                    continue

                rows = [
                    [cell.strip() for cell in line.strip("|").split("|")]
                    for line in lines
                ]
                headers = rows[0]
                data = rows[1:]

                df = pd.DataFrame(data, columns=headers)
                if not self._is_alloy_table(df):
                    continue

                df = self._clean_table(df)
                df["src_name"] = "pdf_nougat"
                df["src_url"] = str(src_pdf)
                frames.append(df)

            except Exception as exc:
                log.warning(f"Markdown table parse error [{src_pdf.name}]: {exc}")

        return frames
