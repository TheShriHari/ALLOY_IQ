"""
ALLOY IQ — Physics-Informed Feature Engineering
================================================
Computes Magpie + domain-specific descriptors for three alloy families:
  • Steels   : Carbon Equivalent (CE), PREN, HAZ proxy
  • HEAs     : ΔSmix, VEC, δ (atomic size mismatch), ΔHmix (Miedema)
  • Al alloys: precipitation strengthening proxy, quench sensitivity index

Usage:
    from backend.data.features import FeatureEngineer
    fe = FeatureEngineer("steel")
    df_features = fe.transform(df_compositions)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Dict, List

# ---------------------------------------------------------------------------
# Elemental property table (subset for common alloy elements)
# Values: (atomic_radius_pm, VEC, molar_mass_g_mol)
# ---------------------------------------------------------------------------
ELEMENT_PROPS: Dict[str, tuple] = {
    "Fe": (126, 8,  55.845),
    "Cr": (128, 6,  51.996),
    "Ni": (124, 10, 58.693),
    "Mo": (139, 6,  95.96),
    "Mn": (127, 7,  54.938),
    "C":  (77,  4,  12.011),
    "Si": (111, 4,  28.085),
    "P":  (106, 5,  30.974),
    "S":  (102, 6,  32.06),
    "N":  (75,  5,  14.007),
    "Cu": (128, 11, 63.546),
    "Ti": (147, 4,  47.867),
    "Al": (143, 3,  26.982),
    "V":  (134, 5,  50.942),
    "W":  (139, 6,  183.84),
    "Co": (125, 9,  58.933),
    "Nb": (146, 5,  92.906),
    "Zr": (160, 4,  91.224),
    "Hf": (159, 4,  178.49),
    "Ta": (146, 5,  180.95),
    "Mg": (160, 2,  24.305),
    "Zn": (134, 12, 65.38),
    "Li": (167, 1,  6.941),
}

# Miedema interaction parameters (simplified, ΔHmix proxy, kJ/mol)
MIEDEMA_H: Dict[frozenset, float] = {
    frozenset(["Cr", "Fe"]): -1.5,
    frozenset(["Ni", "Fe"]): -2.1,
    frozenset(["Mo", "Fe"]): -2.8,
    frozenset(["Co", "Cr"]): -4.2,
    frozenset(["Ni", "Co"]): 0.0,
    frozenset(["Al", "Cr"]): -10.0,
    frozenset(["Al", "Fe"]): -11.0,
    frozenset(["Al", "Ni"]): -22.0,
    frozenset(["Ti", "Al"]): -30.0,
    frozenset(["Ti", "Ni"]): -35.0,
}


@dataclass
class FeatureEngineer:
    """Transforms a composition DataFrame into model-ready features."""
    alloy_family: Literal["steel", "hea", "aluminum"]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Columns are element symbols (e.g. 'Fe', 'Cr', …) with
            values as weight-fraction (0–1).  May include processing
            columns: 'heat_treat_temp_C', 'cooling_rate_C_s'.

        Returns
        -------
        pd.DataFrame with engineered features appended.
        """
        out = df.copy()
        element_cols = [c for c in df.columns if c in ELEMENT_PROPS]

        # --- Universal descriptors ---
        out = self._magpie_proxies(out, element_cols)

        # --- Family-specific descriptors ---
        if self.alloy_family == "steel":
            out = self._steel_features(out)
        elif self.alloy_family == "hea":
            out = self._hea_features(out, element_cols)
        elif self.alloy_family == "aluminum":
            out = self._al_features(out)

        return out

    # ------------------------------------------------------------------ #
    #  Universal: Magpie-proxy descriptors                                #
    # ------------------------------------------------------------------ #
    def _magpie_proxies(self, df: pd.DataFrame, elements: List[str]) -> pd.DataFrame:
        """Mean, range, and std of atomic radius and molar mass weighted by composition."""
        radii   = np.array([ELEMENT_PROPS[e][0] for e in elements])
        masses  = np.array([ELEMENT_PROPS[e][2] for e in elements])
        fracs   = df[elements].values  # shape (n, k)

        # Weighted mean
        df["mean_atomic_radius"] = fracs @ radii
        df["mean_molar_mass"]    = fracs @ masses

        # Weighted std (diversity of atomic sizes)
        r_mean = df["mean_atomic_radius"].values[:, None]
        df["std_atomic_radius"] = np.sqrt(
            (fracs * (radii - r_mean) ** 2).sum(axis=1)
        )

        # Range
        r_max = (fracs > 0).astype(float) @ radii  # approximate
        df["range_atomic_radius"] = radii.max() - radii.min()  # global range placeholder

        return df

    # ------------------------------------------------------------------ #
    #  Steel-specific                                                      #
    # ------------------------------------------------------------------ #
    def _steel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Carbon Equivalent (IIW formula), PREN (Pitting Resistance Equivalent),
        and a simple HAZ (Heat Affected Zone) hardenability proxy.
        """
        def get(col: str) -> pd.Series:
            return df.get(col, pd.Series(0.0, index=df.index))

        # --- Carbon Equivalent (IIW) ---
        # CE = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15
        df["carbon_equivalent"] = (
            get("C")
            + get("Mn") / 6.0
            + (get("Cr") + get("Mo") + get("V")) / 5.0
            + (get("Ni") + get("Cu")) / 15.0
        )

        # --- PREN (Pitting Resistance Equivalent Number) ---
        # PREN = Cr + 3.3*Mo + 16*N
        df["PREN"] = get("Cr") + 3.3 * get("Mo") + 16.0 * get("N")

        # --- Martensite start temperature proxy (Andrews 1965) ---
        # Ms(°C) = 539 − 423C − 30.4Mn − 17.7Ni − 12.1Cr − 7.5Mo (wt%)
        # Convert wt-fraction → wt% (* 100)
        f = 100.0
        df["Ms_proxy"] = (
            539
            - 423 * get("C") * f
            - 30.4 * get("Mn") * f
            - 17.7 * get("Ni") * f
            - 12.1 * get("Cr") * f
            - 7.5  * get("Mo") * f
        )

        # --- Hardenability proxy (simplified Jominy) ---
        df["hardenability_proxy"] = (
            get("C") * 0.64
            + get("Mn") * 4.10
            + get("Si") * 0.64
            + get("Cr") * 2.33
            + get("Ni") * 0.52
            + get("Mo") * 3.14
            + get("V")  * 1.22
        )

        return df

    # ------------------------------------------------------------------ #
    #  HEA-specific                                                        #
    # ------------------------------------------------------------------ #
    def _hea_features(self, df: pd.DataFrame, elements: List[str]) -> pd.DataFrame:
        """
        ΔSmix (configurational entropy), VEC (valence electron concentration),
        δ (atomic size mismatch), ΔHmix (Miedema pair interactions).
        """
        fracs = df[elements].values  # shape (n, k)
        vecs  = np.array([ELEMENT_PROPS[e][1] for e in elements])
        radii = np.array([ELEMENT_PROPS[e][0] for e in elements])

        # --- ΔSmix = -R * Σ(xi * ln(xi)) ---
        R = 8.314  # J / (mol·K)
        safe_fracs = np.where(fracs > 0, fracs, 1e-12)
        df["delta_Smix"] = -R * (fracs * np.log(safe_fracs)).sum(axis=1)

        # --- VEC = Σ(xi * VECi) ---
        df["VEC"] = fracs @ vecs

        # --- δ = sqrt(Σ xi*(1 - ri/r̄)²) * 100 ---
        r_bar = (fracs @ radii)[:, None]
        df["delta_atomic_size"] = np.sqrt(
            (fracs * (1.0 - radii / r_bar) ** 2).sum(axis=1)
        ) * 100.0

        # --- ΔHmix (Miedema pair interactions) ---
        n = len(elements)
        hmix = np.zeros(len(df))
        for i in range(n):
            for j in range(i + 1, n):
                key = frozenset([elements[i], elements[j]])
                h_ij = MIEDEMA_H.get(key, 0.0)
                hmix += 4.0 * fracs[:, i] * fracs[:, j] * h_ij
        df["delta_Hmix"] = hmix

        # --- Ω parameter (Yang & Zhang criterion) ---
        # Ω = Tm_avg * ΔSmix / |ΔHmix|
        # Use a rough Tm proxy from literature averages
        TM_APPROX = {
            "Fe": 1811, "Cr": 2180, "Ni": 1728, "Mo": 2896,
            "Co": 1768, "Al": 933,  "Ti": 1941, "V":  2183,
            "Cu": 1358, "Mn": 1519, "Nb": 2750, "Zr": 2128,
            "W":  3695, "Ta": 3290, "Hf": 2506,
        }
        tm_vals = np.array([TM_APPROX.get(e, 1800) for e in elements])
        tm_avg  = fracs @ tm_vals
        denom   = np.where(np.abs(hmix) > 1e-6, np.abs(hmix), 1e-6)
        df["omega"] = tm_avg * df["delta_Smix"] / denom

        return df

    # ------------------------------------------------------------------ #
    #  Aluminum-specific                                                   #
    # ------------------------------------------------------------------ #
    def _al_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Precipitation strengthening proxy (Orowan–Ashby style),
        quench sensitivity index, and solid-solution strengthening proxy.
        """
        def get(col: str) -> pd.Series:
            return df.get(col, pd.Series(0.0, index=df.index))

        # --- Solid solution strengthening proxy ---
        # Contributions from major solutes (wt-fraction)
        df["ss_strengthening"] = (
            get("Mg") * 66.3
            + get("Si") * 46.5
            + get("Cu") * 13.2
            + get("Zn") * 6.9
            + get("Mn") * 30.0
        )

        # --- Precipitation strengthening proxy ---
        # Primary precipitate-forming elements
        df["precip_potential"] = (
            get("Mg") * get("Si") * 1000.0   # β'' (Mg2Si) in 6xxx
            + get("Cu") * 500.0               # θ' (Al2Cu) in 2xxx
            + get("Mg") * get("Zn") * 800.0  # η (MgZn2) in 7xxx
        )

        # --- Quench sensitivity index (simplified) ---
        df["quench_sensitivity"] = (
            get("Si") * 0.6
            + get("Cu") * 0.35
            + get("Mn") * 0.4
            + get("Cr") * 0.25
        )

        return df


# ---------------------------------------------------------------------------
# CLI: quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal test — duplex stainless steel 2205
    data = {
        "Fe": [0.69], "Cr": [0.225], "Ni": [0.05],
        "Mo": [0.03], "N": [0.002], "Mn": [0.015],
    }
    fe = FeatureEngineer("steel")
    result = fe.transform(pd.DataFrame(data))
    print("\n=== Steel 2205 features ===")
    for col in result.columns:
        print(f"  {col:30s}: {result[col].iloc[0]:.4f}")

    # CoCrFeMnNi (Cantor HEA)
    data_hea = {
        "Co": [0.20], "Cr": [0.20], "Fe": [0.20],
        "Mn": [0.20], "Ni": [0.20],
    }
    fe_hea = FeatureEngineer("hea")
    result_hea = fe_hea.transform(pd.DataFrame(data_hea))
    print("\n=== Cantor HEA features ===")
    for col in result_hea.columns:
        print(f"  {col:30s}: {result_hea[col].iloc[0]:.4f}")
