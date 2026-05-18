"""
ALLOY IQ — Ingestion Pipeline Smoke Test
=========================================
Validates the core pipeline infrastructure without making live API calls.
Runs in < 5 seconds. Safe to run in CI.

Tests:
  1. Schema module — canonical columns, synonym map, make_empty_frame()
  2. Cleaner module — fraction normalization, Fe-balance fill, outlier flagging,
     deduplication
  3. Exporter module — parquet write → read roundtrip, agent_tracker update
  4. Logger module — log file creation, no crashes on WARNING/ERROR

Run with:
    python backend/ingestion/test_pipeline_smoke.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── ensure project root on path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.ingestion.schema import (
    COMPOSITION_COLS,
    ELEMENT_SYMBOLS,
    ALL_COLS,
    SYNONYM_MAP,
    make_empty_frame,
    standardize_columns,
)
from backend.ingestion.cleaner import DataCleaner
from backend.ingestion.exporter import ParquetExporter

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" ({detail})"
    print(msg)
    results.append(condition)


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. Schema module ===")

empty = make_empty_frame()
check("make_empty_frame has all canonical columns",
      all(c in empty.columns for c in ALL_COLS))

# Synonym map test
df_syn = pd.DataFrame({"Yield Strength": [800.0], "UTS": [950.0], "HV": [250.0]})
df_std = standardize_columns(df_syn)
check("SYNONYM_MAP: 'Yield Strength' → 'yield_strength_MPa'",
      "yield_strength_MPa" in df_std.columns)
check("SYNONYM_MAP: 'UTS' → 'ultimate_strength_MPa'",
      "ultimate_strength_MPa" in df_std.columns)
check("SYNONYM_MAP: 'HV' → 'hardness_HV'",
      "hardness_HV" in df_std.columns)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Cleaner module ===")

cleaner = DataCleaner()

# 2a: fraction → wt% normalization
df_frac = pd.DataFrame({
    "Fe_wt": [0.70],
    "Cr_wt": [0.18],
    "Ni_wt": [0.10],
    "Mo_wt": [0.02],
    "yield_strength_MPa": [800.0],
    "alloy_family": ["steel"],
    "src_name": ["test"],
})
df_clean = cleaner.clean(df_frac)
check("Fraction→wt% conversion (0.70 → 70.0)",
      abs(df_clean["Fe_wt"].iloc[0] - 70.0) < 0.01,
      f"Fe_wt={df_clean['Fe_wt'].iloc[0]:.2f}")

# 2b: Fe balance fill
df_nofe = pd.DataFrame({
    "C_wt": [0.45],
    "Mn_wt": [1.20],
    "Cr_wt": [0.95],
    "alloy_family": ["steel"],
    "src_name": ["test"],
})
df_bal = cleaner.clean(df_nofe)
check("Fe balance fill for steel",
      df_bal["Fe_wt"].iloc[0] > 90.0,
      f"Fe_wt={df_bal['Fe_wt'].iloc[0]:.2f}")

# 2c: Outlier flagging — C > 10 wt%
df_outlier = pd.DataFrame({
    "C_wt": [15.0],     # impossible
    "Fe_wt": [85.0],
    "yield_strength_MPa": [800.0],
    "alloy_family": ["steel"],
    "src_name": ["test"],
})
df_flagged = cleaner.clean(df_outlier)
flagged = df_flagged["notes"].str.contains("OUTLIER", na=False).any()
check("Outlier flagged: C_wt=15.0 in steel",
      flagged,
      df_flagged["notes"].iloc[0])

# 2d: Outlier flagging — YS > 5000 MPa
df_ys_out = pd.DataFrame({
    "yield_strength_MPa": [9999.0],
    "Fe_wt": [70.0],
    "alloy_family": ["steel"],
    "src_name": ["test"],
})
df_ys_flagged = cleaner.clean(df_ys_out)
check("Outlier flagged: yield_strength_MPa=9999 in steel",
      df_ys_flagged["notes"].str.contains("OUTLIER", na=False).any())

# 2e: Deduplication
df_dup = pd.DataFrame({
    "Fe_wt": [70.0, 70.0, 80.0],
    "Cr_wt": [18.0, 18.0, 12.0],
    "yield_strength_MPa": [800.0, 800.0, 650.0],
    "alloy_family": ["steel", "steel", "steel"],
    "src_name": ["a", "b", "c"],
})
df_dedup = cleaner.clean(df_dup)
check("Deduplication removes exact duplicates",
      len(df_dedup) == 2,
      f"{len(df_dedup)} rows remain")

# 2f: merge_sources with one empty frame
df_partial = pd.DataFrame({
    "Fe_wt": [65.0], "Cr_wt": [25.0], "Ni_wt": [10.0],
    "yield_strength_MPa": [900.0],
    "alloy_family": ["steel"], "src_name": ["test"],
})
merged = cleaner.merge_sources(make_empty_frame(), df_partial)
check("merge_sources tolerates empty frames",
      not merged.empty and len(merged) >= 1)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Exporter module ===")

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    tracker_path = tmp_path / "agent_tracker.json"

    # Monkeypatch exporter paths for test isolation
    import backend.ingestion.exporter as exporter_mod
    original_processed = exporter_mod.PROCESSED_DIR
    original_tracker = exporter_mod.AGENT_TRACKER_PATH
    exporter_mod.PROCESSED_DIR = tmp_path
    exporter_mod.AGENT_TRACKER_PATH = tracker_path

    exporter = ParquetExporter()

    df_export = pd.DataFrame({
        "Fe_wt": [70.0, 65.0],
        "Cr_wt": [18.0, 20.0],
        "yield_strength_MPa": [800.0, 920.0],
        "alloy_family": ["steel", "steel"],
        "src_name": ["test", "test"],
        "src_id": [None, None],
        "src_url": [None, None],
        "notes": ["", ""],
    })

    written = exporter.export(df_export)
    check("Parquet files written", len(written) > 0, f"{len(written)} file(s)")

    if written:
        df_read = pd.read_parquet(written[0])
        check("Parquet roundtrip — row count preserved",
              len(df_read) == len(df_export),
              f"{len(df_read)} rows")
        check("Parquet roundtrip — yield_strength_MPa column present",
              "yield_strength_MPa" in df_read.columns)

    check("agent_tracker.json created", tracker_path.exists())

    if tracker_path.exists():
        with open(tracker_path) as f:
            tracker = json.load(f)
        check("agent_tracker.json has parquet_files entry",
              len(tracker.get("parquet_files", {})) > 0)
        check("agent_tracker.json has last_updated timestamp",
              tracker.get("last_updated") is not None)

    # Restore
    exporter_mod.PROCESSED_DIR = original_processed
    exporter_mod.AGENT_TRACKER_PATH = original_tracker

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. Logger module ===")

from backend.ingestion.logger import get_logger
log = get_logger("smoke_test")
log.info("Logger smoke test — INFO")
log.warning("Logger smoke test — WARNING")
log.error("Logger smoke test — ERROR")

log_dir = Path("logs")
check("logs/ directory created", log_dir.exists())
check("ingestion_full.log created", (log_dir / "ingestion_full.log").exists())
check("global_ingestion_errors.log created", (log_dir / "global_ingestion_errors.log").exists())

# ══════════════════════════════════════════════════════════════════════════════
total = len(results)
passed = sum(results)
failed = total - passed

print(f"\n{'═' * 50}")
print(f"  Smoke Test Results: {passed}/{total} passed", end="")
if failed:
    print(f"  < {failed} FAILED")
else:
    print("  > ALL PASSED")
print(f"{'═' * 50}\n")

sys.exit(0 if failed == 0 else 1)
