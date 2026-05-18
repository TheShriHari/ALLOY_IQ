"""
ALLOY IQ — Unified Schema Definition
=====================================
Defines the canonical column names and dtypes for the unified
metallurgical DataFrame. All ingestion sources must be standardized
to this schema before parquet export.

Column naming convention:
  - Elemental compositions: element symbol in wt%  (e.g. "Fe_wt", "Cr_wt")
  - Properties: lowercase snake_case with units as suffix where helpful
  - Source metadata: prefixed with "src_"
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

# ── Canonical column name groups ─────────────────────────────────────────────

#: Supported element symbols (major alloying elements)
ELEMENT_SYMBOLS: List[str] = [
    "Fe", "Cr", "Ni", "Mo", "Mn", "C", "Si", "P", "S", "N",
    "Cu", "Ti", "Al", "V", "W", "Co", "Nb", "Zr", "Hf", "Ta",
    "Mg", "Zn", "Li", "B", "Ce", "Y", "Sc",
]

#: Canonical composition column names (wt%)
COMPOSITION_COLS: List[str] = [f"{el}_wt" for el in ELEMENT_SYMBOLS]

#: Canonical mechanical property columns
MECHANICAL_COLS: List[str] = [
    "yield_strength_MPa",       # YS / Rp0.2
    "ultimate_strength_MPa",    # UTS / Rm
    "hardness_HV",              # Vickers
    "hardness_HRC",             # Rockwell C
    "elongation_pct",           # El%
    "reduction_area_pct",       # RA%
    "youngs_modulus_GPa",       # E
    "fracture_toughness_MPa_sqrtm",  # KIc
    "fatigue_limit_MPa",        # σ_f
]

#: Canonical thermodynamic / structural property columns
THERMO_COLS: List[str] = [
    "formation_enthalpy_eV_atom",
    "volume_A3_atom",
    "density_g_cm3",
    "melting_point_K",
]

#: Canonical corrosion property columns
CORROSION_COLS: List[str] = [
    "pren",                         # Pitting Resistance Equivalent Number
    "pitting_potential_mV_SCE",
    "corrosion_rate_mpy",
]

#: Processing parameter columns
PROCESSING_COLS: List[str] = [
    "heat_treat_temp_C",
    "cooling_rate_C_s",
    "aging_temp_C",
    "aging_time_h",
    "cold_work_pct",
]

#: Source metadata columns
METADATA_COLS: List[str] = [
    "src_name",          # e.g. "matminer_citrine", "aflow", "pdf_extraction"
    "src_id",            # original ID in the source system
    "src_url",           # URL or DOI link
    "alloy_family",      # "steel" | "hea" | "aluminum"
    "source_tier",       # "tier1", "tier2", "tier3", "tier4"
    "material_id",       # MP material ID if applicable
    "notes",
]

#: All canonical columns in order
ALL_COLS: List[str] = (
    METADATA_COLS
    + COMPOSITION_COLS
    + PROCESSING_COLS
    + MECHANICAL_COLS
    + THERMO_COLS
    + CORROSION_COLS
)

#: Dtypes for non-numeric columns
COLUMN_DTYPES: Dict[str, str] = {
    "src_name": "category",
    "src_id": "string",
    "src_url": "string",
    "alloy_family": "category",
    "source_tier": "category",
    "material_id": "string",
    "notes": "string",
}

# ── Column-name synonym map ──────────────────────────────────────────────────
# Maps varied naming conventions from raw sources to canonical column names.
# Keys are lowercase for case-insensitive matching.

SYNONYM_MAP: Dict[str, str] = {
    # Yield Strength
    "yield strength": "yield_strength_MPa",
    "ys": "yield_strength_MPa",
    "rp0.2": "yield_strength_MPa",
    "rp 0.2": "yield_strength_MPa",
    "0.2% proof stress": "yield_strength_MPa",
    "proof stress": "yield_strength_MPa",
    "tensile yield strength": "yield_strength_MPa",
    "yield_strength": "yield_strength_MPa",
    "yield strength (mpa)": "yield_strength_MPa",
    "σy": "yield_strength_MPa",
    # Ultimate Tensile Strength
    "ultimate tensile strength": "ultimate_strength_MPa",
    "uts": "ultimate_strength_MPa",
    "tensile strength": "ultimate_strength_MPa",
    "rm": "ultimate_strength_MPa",
    "tensile_strength": "ultimate_strength_MPa",
    "ultimate strength": "ultimate_strength_MPa",
    "σuts": "ultimate_strength_MPa",
    # Hardness
    "hardness": "hardness_HV",
    "vickers hardness": "hardness_HV",
    "hv": "hardness_HV",
    "hvn": "hardness_HV",
    "hv0.5": "hardness_HV",
    "hrc": "hardness_HRC",
    "rockwell c": "hardness_HRC",
    # Elongation
    "elongation": "elongation_pct",
    "el%": "elongation_pct",
    "elongation (%)": "elongation_pct",
    "total elongation": "elongation_pct",
    "failure_strain": "elongation_pct",
    # Reduction of Area
    "reduction of area": "reduction_area_pct",
    "ra%": "reduction_area_pct",
    "ra (%)": "reduction_area_pct",
    # Young's Modulus
    "young's modulus": "youngs_modulus_GPa",
    "youngs modulus": "youngs_modulus_GPa",
    "elastic modulus": "youngs_modulus_GPa",
    "e (gpa)": "youngs_modulus_GPa",
    # Fracture Toughness
    "fracture toughness": "fracture_toughness_MPa_sqrtm",
    "kic": "fracture_toughness_MPa_sqrtm",
    "k1c": "fracture_toughness_MPa_sqrtm",
    # Fatigue
    "fatigue limit": "fatigue_limit_MPa",
    "endurance limit": "fatigue_limit_MPa",
    # Formation Enthalpy
    "formation enthalpy": "formation_enthalpy_eV_atom",
    "enthalpy_formation_atom": "formation_enthalpy_eV_atom",
    "enthalpy of formation": "formation_enthalpy_eV_atom",
    "delta_hf": "formation_enthalpy_eV_atom",
    # Volume
    "volume_atom": "volume_A3_atom",
    "volume per atom": "volume_A3_atom",
    # PREN
    "pren": "pren",
    "pitting resistance equivalent": "pren",
}


def make_empty_frame() -> pd.DataFrame:
    """Return an empty DataFrame with all canonical columns."""
    df = pd.DataFrame(columns=ALL_COLS)
    for col, dtype in COLUMN_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw column headers to canonical names using SYNONYM_MAP.
    Case-insensitive, strips whitespace.
    Unknown columns are kept as-is (not dropped).
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        normalized = str(col).strip().lower()
        if normalized in SYNONYM_MAP:
            rename_map[col] = SYNONYM_MAP[normalized]
    return df.rename(columns=rename_map)
