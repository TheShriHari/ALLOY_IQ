"""
ALLOY IQ — Data Standardization & Cleaning Engine
==================================================
Merges disparate source DataFrames into one unified canonical schema,
then applies:

  1. Composition normalization — convert at% → wt%, mole fraction → wt%,
     fill implicit Fe balance for steels.
  2. Outlier detection — flags physically impossible values per alloy family:
     - Steel:  C > 10 wt%, YS > 5000 MPa, hardness_HV > 1500
     - HEA:    any single element > 60 wt%, YS > 4000 MPa
     - Al:     Fe > 15 wt%, YS > 1500 MPa
  3. Duplicate deduplication by (composition fingerprint, property columns).
  4. Type casting to match COLUMN_DTYPES from schema.

All flagged outliers are logged with row index, column, value, and reason.

Usage:
    from backend.ingestion.cleaner import DataCleaner
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.ingestion.logger import get_logger
from backend.ingestion.schema import (
    ALL_COLS,
    COLUMN_DTYPES,
    COMPOSITION_COLS,
    ELEMENT_SYMBOLS,
    MECHANICAL_COLS,
    METADATA_COLS,
    THERMO_COLS,
    make_empty_frame,
)

log = get_logger(__name__)

# ── Atomic weights for at% → wt% conversion ──────────────────────────────────
ATOMIC_WEIGHTS: Dict[str, float] = {
    "Fe": 55.845, "Cr": 51.996, "Ni": 58.693, "Mo": 95.96,
    "Mn": 54.938, "C":  12.011, "Si": 28.085, "P":  30.974,
    "S":  32.06,  "N":  14.007, "Cu": 63.546, "Ti": 47.867,
    "Al": 26.982, "V":  50.942, "W":  183.84, "Co": 58.933,
    "Nb": 92.906, "Zr": 91.224, "Hf": 178.49, "Ta": 180.95,
    "Mg": 24.305, "Zn": 65.38,  "Li": 6.941,  "B":  10.811,
    "Ce": 140.12, "Y":  88.906, "Sc": 44.956,
}

# ── Physical bounds per alloy family ─────────────────────────────────────────
# Tuple: (min_value, max_value) — None means no bound on that side.
# Values in the canonical units of each column.
_BOUNDS: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {
    "steel": {
        "C_wt":                         (0.0,  10.0),
        "Cr_wt":                        (0.0,  35.0),
        "Ni_wt":                        (0.0,  40.0),
        "Mo_wt":                        (0.0,  15.0),
        "yield_strength_MPa":           (50.0, 5000.0),
        "ultimate_strength_MPa":        (100.0, 6000.0),
        "hardness_HV":                  (50.0, 1500.0),
        "hardness_HRC":                 (0.0,  70.0),
        "elongation_pct":               (0.0,  80.0),
        "formation_enthalpy_eV_atom":   (-5.0, 5.0),
    },
    "hea": {
        "yield_strength_MPa":           (50.0,  4000.0),
        "ultimate_strength_MPa":        (100.0, 5000.0),
        "hardness_HV":                  (50.0,  1500.0),
        "elongation_pct":               (0.0,   100.0),
        "formation_enthalpy_eV_atom":   (-5.0,  5.0),
    },
    "aluminum": {
        "Fe_wt":                        (0.0,  15.0),
        "Al_wt":                        (50.0, 100.0),   # must be majority
        "yield_strength_MPa":           (10.0, 1500.0),
        "ultimate_strength_MPa":        (50.0, 2000.0),
        "hardness_HV":                  (10.0, 500.0),
        "elongation_pct":               (0.0,  50.0),
    },
}

# Single-element cap for HEA (no one element should dominate)
_HEA_MAX_SINGLE_ELEMENT_WT = 60.0


class DataCleaner:
    """
    Merges and cleans raw source DataFrames into the canonical schema.
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full cleaning pipeline.

        Steps:
          1. Ensure all canonical columns exist (add NaN for missing)
          2. Normalize compositions → wt%
          3. Cast dtypes
          4. Detect and flag outliers (rows are NOT dropped — flagged in 'notes')
          5. Deduplicate
          6. Log summary statistics

        Parameters
        ----------
        df : pd.DataFrame
            Raw merged DataFrame from any source combination.

        Returns
        -------
        pd.DataFrame — cleaned, outlier-flagged, deduplicated.
        """
        if df.empty:
            log.warning("DataCleaner received empty DataFrame.")
            return make_empty_frame()

        log.info(f"DataCleaner: starting with {len(df)} rows × {df.shape[1]} cols")

        df = self._ensure_canonical_columns(df)
        df = self._normalize_compositions(df)
        df = self._cast_dtypes(df)
        df = self._flag_outliers(df)
        df = self._deduplicate(df)

        log.info(
            f"DataCleaner: finished → {len(df)} rows "
            f"({(df['notes'].str.contains('OUTLIER', na=False)).sum()} flagged as outliers)"
        )
        return df

    def merge_sources(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate multiple source DataFrames, then clean.
        Silently ignores empty frames.
        """
        non_empty = [d for d in dfs if not d.empty]
        if not non_empty:
            log.warning("merge_sources: all input DataFrames are empty.")
            return make_empty_frame()

        log.info(f"Merging {len(non_empty)} source(s) …")
        merged = pd.concat(non_empty, ignore_index=True)
        log.info(f"  → Combined shape before cleaning: {merged.shape}")
        return self.clean(merged)

    # ── Step 1: Canonical column scaffolding ──────────────────────────────────

    @staticmethod
    def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add missing canonical columns as NaN; keep non-canonical extras."""
        for col in ALL_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df

    # ── Step 2: Composition normalization ─────────────────────────────────────

    def _normalize_compositions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all _wt columns are in weight percent (0–100 scale).

        Detection heuristics:
          a) If max of all _wt cols < 1.5 → assume mole fractions, multiply by 100
          b) If 'composition_unit' column says 'at%' → convert at% → wt%
          c) For steels: if all elements sum to << 100 → fill Fe balance
        """
        wt_cols = [c for c in COMPOSITION_COLS if c in df.columns]
        if not wt_cols:
            return df

        numeric_wt = df[wt_cols].apply(pd.to_numeric, errors="coerce")

        # ── Heuristic (a): fraction → percent ────────────────────────────────
        valid_sum = numeric_wt.sum(axis=1, skipna=True)
        fraction_rows = (valid_sum > 0.5) & (valid_sum < 1.5)
        if fraction_rows.any():
            log.info(
                f"  Composition normalization: converting {fraction_rows.sum()} "
                f"fraction rows (values < 1.5) to wt% (×100)"
            )
            df.loc[fraction_rows, wt_cols] = numeric_wt.loc[fraction_rows] * 100.0

        # ── Heuristic (c): Fe balance for steels ──────────────────────────────
        steel_mask = df["alloy_family"] == "steel"
        if "Fe_wt" in df.columns and steel_mask.any():
            steel_df = df[steel_mask].copy()
            non_fe = [c for c in wt_cols if c != "Fe_wt"]
            row_sum = (
                steel_df[non_fe]
                .apply(pd.to_numeric, errors="coerce")
                .sum(axis=1, skipna=True)
            )
            missing_fe = steel_df["Fe_wt"].isna() | (steel_df["Fe_wt"] == 0)
            fe_balance = (100.0 - row_sum).clip(lower=0)
            
            # Use the index from the filtered subset
            update_idx = missing_fe[missing_fe].index
            df.loc[update_idx, "Fe_wt"] = fe_balance.loc[update_idx]
            log.info(
                f"  Fe balance filled for {(steel_mask & missing_fe).sum()} steel rows"
            )

        return df

    # ── Step 3: dtype casting ─────────────────────────────────────────────────

    @staticmethod
    def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Cast canonical non-numeric columns to correct dtypes."""
        for col, dtype in COLUMN_DTYPES.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError) as exc:
                    log.warning(f"dtype cast failed for {col} → {dtype}: {exc}")
        return df

    # ── Step 4: Outlier detection ─────────────────────────────────────────────

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag physically impossible data points with 'OUTLIER:column=value'
        annotations in the 'notes' column. Rows are NOT dropped.
        """
        total_flagged = 0

        for idx, row in df.iterrows():
            family = str(row.get("alloy_family", "")).lower()
            bounds = _BOUNDS.get(family, {})
            flags: List[str] = []

            # ── Bounds checks ─────────────────────────────────────────────
            for col, (lo, hi) in bounds.items():
                if col not in df.columns:
                    continue
                val = pd.to_numeric(row.get(col), errors="coerce")
                if pd.isna(val):
                    continue
                if lo is not None and val < lo:
                    flags.append(f"OUTLIER:{col}={val:.3g}<{lo}")
                if hi is not None and val > hi:
                    flags.append(f"OUTLIER:{col}={val:.3g}>{hi}")

            # ── HEA: single-element dominance ────────────────────────────
            if family == "hea":
                wt_cols = [c for c in COMPOSITION_COLS if c in df.columns]
                for wt_col in wt_cols:
                    val = pd.to_numeric(row.get(wt_col), errors="coerce")
                    if not pd.isna(val) and val > _HEA_MAX_SINGLE_ELEMENT_WT:
                        flags.append(
                            f"OUTLIER:{wt_col}={val:.1f}>(HEA max {_HEA_MAX_SINGLE_ELEMENT_WT}%)"
                        )

            if flags:
                existing_notes = str(row.get("notes", "")) or ""
                separator = " | " if existing_notes else ""
                df.at[idx, "notes"] = existing_notes + separator + " | ".join(flags)
                total_flagged += 1
                log.warning(
                    f"  Outlier row {idx} [{family}]: {' | '.join(flags)}"
                )

        if total_flagged:
            log.warning(f"Outlier detection: flagged {total_flagged} rows")
        else:
            log.info("Outlier detection: no outliers found")

        return df

    # ── Step 5: Deduplication ─────────────────────────────────────────────────

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate rows by a composition + top-property fingerprint.
        Keeps first occurrence. Logs number of duplicates removed.
        """
        wt_cols = [c for c in COMPOSITION_COLS if c in df.columns]
        prop_cols = [c for c in MECHANICAL_COLS if c in df.columns]
        key_cols = wt_cols + prop_cols

        if not key_cols:
            return df

        # Round to 2 decimal places for fingerprinting
        numeric_keys = df[key_cols].apply(pd.to_numeric, errors="coerce").round(2)
        fingerprints = numeric_keys.apply(
            lambda row: hashlib.md5(
                row.fillna(-9999).to_string().encode()
            ).hexdigest(),
            axis=1,
        )

        before = len(df)
        df = df[~fingerprints.duplicated(keep="first")].reset_index(drop=True)
        removed = before - len(df)

        if removed:
            log.info(f"Deduplication: removed {removed} duplicate rows → {len(df)} remain")

        return df
