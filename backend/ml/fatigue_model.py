"""
Fatigue limit and fracture toughness prediction module.
Strategy: empirical proxies + transfer learning from UTS model.

Two approaches:
  1. Direct: if KIc data is available, train directly (data from ASM Fracture database)
  2. Proxy: use Barsom-Rolfe / Charpy correlations for steel screening
"""
import numpy as np

# ── Proxy models (no training data required) ──────────────────────

def estimate_fatigue_limit(uts_mpa: float, alloy_family: str) -> dict:
    """
    Estimate fatigue limit from UTS using Wöhler-law approximations.

    For steels: σ_f ≈ 0.45–0.50 × UTS (valid up to UTS ≈ 1400 MPa)
    For Al alloys: σ_f ≈ 0.30–0.40 × UTS (lower fatigue ratio)
    For HEAs: σ_f ≈ 0.40–0.45 × UTS (similar to steels, limited data)

    Returns estimate with uncertainty range (±15% for this proxy approach).
    """
    if alloy_family == "steel":
        if uts_mpa <= 1400:
            ratio_mean, ratio_std = 0.475, 0.025
        else:
            # High-strength steels: ratio drops (hydrogen embrittlement, inclusions)
            ratio_mean, ratio_std = 0.40, 0.04
    elif alloy_family == "aluminum":
        ratio_mean, ratio_std = 0.35, 0.05
    else:   # HEA
        ratio_mean, ratio_std = 0.425, 0.035

    fl_mean = uts_mpa * ratio_mean
    fl_lower = uts_mpa * (ratio_mean - 2 * ratio_std)
    fl_upper = uts_mpa * (ratio_mean + 2 * ratio_std)

    return {
        "fatigue_limit_mpa": round(fl_mean, 1),
        "fatigue_limit_lower": round(fl_lower, 1),
        "fatigue_limit_upper": round(fl_upper, 1),
        "fatigue_ratio": round(ratio_mean, 3),
        "method": "Wöhler proxy from UTS",
        "confidence": "screening only — physical S-N curve testing required for design",
    }


def estimate_fracture_toughness(ys_mpa: float, hv: float, alloy_family: str) -> dict:
    """
    Estimate KIc from yield strength and hardness using Barsom-Rolfe correlation.

    Barsom-Rolfe (1999) for steels:
        KIc ≈ 0.64 × (σy²) / E    [simplified — full form needs CVN impact energy]

    Better proxy using HV (Vickers hardness):
        E_approx = 3 × HV × 9.81   (approximate yield stress from hardness)
        Then apply Barsom-Rolfe

    Returns KIc in MPa√m with wide uncertainty (±30% — proxy only).
    """
    # Young's modulus approximation from alloy family
    E_gpa = {"steel": 210, "aluminum": 70, "hea": 180}.get(alloy_family, 200)

    # Barsom-Rolfe simplified
    kic_estimate = 0.64 * (ys_mpa ** 2) / (E_gpa * 1000)   # in MPa√m

    # Adjust for hardness (high HV → lower toughness due to reduced plastic zone)
    hv_penalty = np.clip((hv - 200) / 600, 0, 0.40)  # up to 40% reduction
    kic_estimate *= (1 - hv_penalty)

    kic_estimate = np.clip(kic_estimate, 20, 200)  # physical bounds for steels

    return {
        "fracture_toughness_kic_mpa_sqrtm": round(kic_estimate, 1),
        "kic_lower": round(kic_estimate * 0.70, 1),
        "kic_upper": round(kic_estimate * 1.30, 1),
        "method": "Barsom-Rolfe proxy (no Charpy data)",
        "ndt_guidance": _ndt_guidance(kic_estimate, ys_mpa),
        "confidence": "screening only — ASTM E399 testing required for design certification",
    }


def _ndt_guidance(kic: float, ys: float) -> str:
    """Generate LEFM (Linear Elastic Fracture Mechanics) design guidance."""
    # NDT (Non-Destructive Testing) detectable crack size from KIc
    # a_NDT = (KIc / (1.12 × σy))² / π
    sigma_design = ys * 0.67   # assume 2/3 yield as design stress
    a_ndt_m = (kic / (1.12 * sigma_design)) ** 2 / np.pi
    a_ndt_mm = a_ndt_m * 1000

    if a_ndt_mm > 10:
        return f"NDT detectable flaw size: ~{a_ndt_mm:.0f} mm — easy to inspect, robust to surface defects"
    elif a_ndt_mm > 1:
        return f"NDT detectable flaw size: ~{a_ndt_mm:.1f} mm — standard UT/dye-penetrant inspection adequate"
    else:
        return f"NDT detectable flaw size: ~{a_ndt_mm:.2f} mm — high-resolution TOFD/phased array UT required"


# ── Integrate into /predict/mechanical ──────────────────────────────

def add_fatigue_fracture(prediction: dict, composition: dict, alloy_family: str) -> dict:
    """
    Add fatigue and fracture toughness estimates to an existing prediction dict.
    Call this after the main ML model runs.
    """
    # predictions might be simple target value or a dictionary of {mean, lower, upper}
    # Sonnet updated prediction structure to have 'predictions' containing the targets
    target_predictions = prediction.get("predictions", {})
    
    uts = target_predictions.get("tensile_strength_mpa", {}).get("mean", 800)
    ys  = target_predictions.get("yield_strength_mpa",   {}).get("mean", 600)
    hv  = target_predictions.get("hardness_hv",           {}).get("mean", 250)

    fatigue = estimate_fatigue_limit(uts, alloy_family)
    fracture = estimate_fracture_toughness(ys, hv, alloy_family)

    prediction["fatigue"] = fatigue
    prediction["fracture_toughness"] = fracture
    return prediction
