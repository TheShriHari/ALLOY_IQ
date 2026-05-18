"""
ALLOY IQ — Matminer & Materials Project Retrieval
==================================================
Pulls bulk mechanical property + composition datasets from:
  1. Matminer built-in dataset loaders (steel_strength, jarvis, etc.)
  2. Materials Project via mp-api (for crystal / thermodynamic properties)

Fault-tolerance:
  - All exceptions are caught, logged, and return an empty DataFrame
    rather than crashing the pipeline.
  - Credentials are loaded exclusively from .env via python-dotenv.

Usage:
    from backend.ingestion.matminer_retriever import MatminerRetriever
    retriever = MatminerRetriever()
    df_steel = retriever.fetch_steel_strength()
    df_mp    = retriever.fetch_mp_mechanical(elements=["Fe","Cr","Ni"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

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

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "backend/data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

MP_API_KEY: Optional[str] = os.getenv("MP_API_KEY")


class MatminerRetriever:
    """
    Retrieves structured datasets from Matminer's built-in loaders
    and the Materials Project REST API.
    """

    # ── Built-in Matminer datasets ────────────────────────────────────────────

    def fetch_steel_strength(self) -> pd.DataFrame:
        """
        Loads the Matminer 'steel_strength' dataset (~312 steels).
        Contains composition (wt%) + yield strength, tensile strength,
        elongation, and hardness.

        Returns canonical-schema DataFrame or empty frame on failure.
        """
        try:
            from matminer.datasets import load_dataset  # type: ignore

            log.info("Fetching matminer: steel_strength dataset …")
            df = load_dataset("steel_strength")
            log.info(f"  → Fetched {len(df)} rows from steel_strength")

            df = self._normalize_steel_strength(df)
            df["src_name"] = "matminer_steel_strength"
            df["alloy_family"] = "steel"
            df["src_url"] = "https://hackingmaterials.lbl.gov/matminer/dataset_summary.html"
            return df

        except ImportError:
            log.error("matminer not installed. Run: pip install matminer")
        except Exception as exc:
            log.error(f"Failed to fetch steel_strength: {exc}", exc_info=True)

        return make_empty_frame()

    def fetch_glass_forming_ability(self) -> pd.DataFrame:
        """
        Loads the Matminer 'glass_formation' dataset — useful for HEA
        research as amorphous/crystalline stability relates to ΔSmix/VEC.
        """
        try:
            from matminer.datasets import load_dataset  # type: ignore

            log.info("Fetching matminer: glass_formation dataset …")
            df = load_dataset("glass_formation")
            log.info(f"  → Fetched {len(df)} rows from glass_formation")

            df = standardize_columns(df)
            df["src_name"] = "matminer_glass_formation"
            df["alloy_family"] = "hea"  # overlapping HEA territory
            return df

        except ImportError:
            log.error("matminer not installed.")
        except Exception as exc:
            log.error(f"Failed to fetch glass_formation: {exc}", exc_info=True)

        return make_empty_frame()

    def fetch_jarvis_dft(self) -> pd.DataFrame:
        """
        Loads Matminer's JARVIS-DFT dataset for bulk modulus, shear modulus,
        and formation energies. Useful for Al alloy thermodynamic properties.
        """
        try:
            from matminer.datasets import load_dataset  # type: ignore

            log.info("Fetching matminer: jarvis_dft_2d dataset …")
            df = load_dataset("jarvis_dft_3d")
            log.info(f"  → Fetched {len(df)} rows from JARVIS-DFT-3D")

            # Select only relevant columns
            keep = [c for c in df.columns if c in [
                "formula", "bulk_modulus", "shear_modulus",
                "formation_energy_peratom", "e_form", "optB88vdW_bandgap",
                "density",
            ]]
            df = df[keep].copy()
            df = standardize_columns(df)
            df["src_name"] = "matminer_jarvis_dft_3d"
            df["alloy_family"] = "hea"  # mixed, filtered downstream
            return df

        except ImportError:
            log.error("matminer not installed.")
        except Exception as exc:
            log.error(f"Failed to fetch JARVIS-DFT: {exc}", exc_info=True)

        return make_empty_frame()

    # ── Materials Project API ────────────────────────────────────────────────

    def fetch_mp_mechanical(
        self,
        elements: Optional[List[str]] = None,
        max_rows: int = 5000,
    ) -> pd.DataFrame:
        """
        Queries the Materials Project API for elastic/mechanical data.
        Requires MP_API_KEY in .env.

        Parameters
        ----------
        elements : list[str], optional
            Filter to structures containing ALL of these elements.
        max_rows : int
            Cap on results (avoids very large downloads).

        Returns canonical-schema DataFrame or empty frame on failure.
        """
        if not MP_API_KEY:
            log.warning(
                "MP_API_KEY not set in .env — skipping Materials Project query. "
                "Get a key at https://next-gen.materialsproject.org/api"
            )
            return make_empty_frame()

        try:
            from mp_api.client import MPRester  # type: ignore

            log.info(f"Querying Materials Project: elements={elements}, max={max_rows} …")

            with MPRester(MP_API_KEY) as mpr:
                fields = [
                    "material_id", "formula_pretty", "symmetry",
                    "bulk_modulus", "shear_modulus", "universal_anisotropy",
                    "homogeneous_poisson",
                ]
                docs = mpr.materials.elasticity.search(
                    elements=elements,
                    fields=fields,
                    num_chunks=1,
                    chunk_size=min(max_rows, 1000),
                )

            records = []
            for doc in docs[:max_rows]:
                records.append({
                    "material_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "bulk_modulus_GPa": (
                        doc.bulk_modulus.voigt if doc.bulk_modulus else None
                    ),
                    "shear_modulus_GPa": (
                        doc.shear_modulus.voigt if doc.shear_modulus else None
                    ),
                    "src_name": "materials_project",
                    "src_url": f"https://next-gen.materialsproject.org/materials/{doc.material_id}",
                })

            df = pd.DataFrame(records)
            log.info(f"  → Fetched {len(df)} rows from Materials Project elasticity")
            return df

        except ImportError:
            log.error("mp-api not installed. Run: pip install mp-api")
        except Exception as exc:
            log.error(f"Materials Project query failed: {exc}", exc_info=True)

        return make_empty_frame()

    def fetch_mp_composition_properties(
        self,
        chemsys: Optional[str] = None,
        max_rows: int = 10000,
    ) -> pd.DataFrame:
        """
        Fetches thermodynamic summary data from MP for a given chemical system.

        Parameters
        ----------
        chemsys : str, optional
            e.g. "Fe-Cr-Ni" to get all ternary combinations in this system.
        max_rows : int
            Cap on results.

        Returns canonical-schema DataFrame.
        """
        if not MP_API_KEY:
            log.warning("MP_API_KEY not set — skipping MP composition fetch.")
            return make_empty_frame()

        try:
            from mp_api.client import MPRester  # type: ignore

            log.info(f"Querying Materials Project summary: chemsys={chemsys} …")

            with MPRester(MP_API_KEY) as mpr:
                docs = mpr.materials.summary.search(
                    chemsys=chemsys,
                    fields=[
                        "material_id", "formula_pretty", "composition",
                        "formation_energy_per_atom", "volume",
                        "density", "energy_above_hull",
                    ],
                )

            records = []
            for doc in docs[:max_rows]:
                comp = dict(doc.composition.fractional_composition)
                row = {
                    "material_id": str(doc.material_id),
                    "formation_enthalpy_eV_atom": doc.formation_energy_per_atom,
                    "volume_A3_atom": doc.volume / doc.composition.num_atoms if doc.volume else None,
                    "density_g_cm3": doc.density,
                    "src_name": "materials_project",
                    "src_id": str(doc.material_id),
                    "src_url": f"https://next-gen.materialsproject.org/materials/{doc.material_id}",
                }
                # Add fractional composition → convert to wt%
                for el_sym, frac in comp.items():
                    col = f"{el_sym}_wt"
                    row[col] = frac  # fractional; convert to wt% in cleaner
                records.append(row)

            df = pd.DataFrame(records)
            log.info(f"  → Fetched {len(df)} rows from MP summary ({chemsys})")
            return df

        except ImportError:
            log.error("mp-api not installed.")
        except Exception as exc:
            log.error(f"MP composition query failed: {exc}", exc_info=True)

        return make_empty_frame()

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_steel_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map Matminer steel_strength columns to the canonical schema.
        The dataset uses wt% fractions (0–100 scale) for elements.
        """
        col_map = {
            # Compositional columns (already in wt%)
            "C": "C_wt",
            "Si": "Si_wt",
            "Mn": "Mn_wt",
            "P": "P_wt",
            "S": "S_wt",
            "Cr": "Cr_wt",
            "Ni": "Ni_wt",
            "Mo": "Mo_wt",
            "Cu": "Cu_wt",
            "V": "V_wt",
            "Nb": "Nb_wt",
            "Co": "Co_wt",
            "W": "W_wt",
            "Al": "Al_wt",
            "Ti": "Ti_wt",
            "N": "N_wt",
            "B": "B_wt",
            # Property columns
            "yield strength": "yield_strength_MPa",
            "tensile strength": "ultimate_strength_MPa",
            "elongation": "elongation_pct",
            "reduction of area": "reduction_area_pct",
            "hardness": "hardness_HV",
        }
        # Apply only columns that exist
        existing = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=existing)
        return df
