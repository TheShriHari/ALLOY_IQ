"""
Post-processing for Pareto front analysis.
Filters, ranks, and annotates Pareto-optimal alloy candidates.
"""
import numpy as np

def rank_pareto_candidates(candidates: list[dict], priorities: list[str]) -> list[dict]:
    """
    Rank Pareto-front candidates by weighted priority.
    priorities: ordered list of property names by user importance
               e.g. ["yield_strength_mpa", "corrosion_pren", "elongation_pct"]
    """
    if not candidates:
        return []

    scored = []
    for c in candidates:
        preds = c.get("predictions", {})
        score = 0.0
        for rank, prop in enumerate(priorities):
            weight = 1.0 / (rank + 1)   # higher priority → higher weight
            score += preds.get(prop, 0) * weight
        scored.append({**c, "_priority_score": score})

    return sorted(scored, key=lambda x: x["_priority_score"], reverse=True)


def classify_candidate(candidate: dict) -> dict:
    """
    Add metallurgical classification and application suggestions to a candidate.
    Returns the candidate dict with added 'classification' and 'suggested_applications' fields.
    """
    preds = candidate.get("predictions", {})
    comp = candidate.get("composition", {})

    ys = preds.get("yield_strength_mpa", 0)
    hv = preds.get("hardness_hv", 0)
    cr = comp.get("Cr", 0) * 100  # approximate wt%
    pren = preds.get("corrosion_pren", cr)

    # Steel classification
    if ys > 1500:
        alloy_class = "Ultra-high-strength steel (UHSS)"
        applications = ["Aerospace structural", "Armor plate", "High-performance fasteners"]
    elif ys > 900:
        alloy_class = "High-strength steel (HSS)"
        applications = ["Automotive structural", "Pressure vessels", "Tool steel"]
    elif ys > 500:
        alloy_class = "Medium-strength alloy"
        applications = ["General engineering", "Pipelines", "Construction"]
    else:
        alloy_class = "Low-strength / ductile alloy"
        applications = ["Sheet metal forming", "Deep drawing", "Electrical applications"]

    # Corrosion classification overlay
    if pren >= 40:
        alloy_class += " + Super corrosion resistant"
        applications.append("Offshore oil & gas")
        applications.append("Chemical processing equipment")
    elif pren >= 25:
        alloy_class += " + Corrosion resistant"
        applications.append("Marine environment")

    return {
        **candidate,
        "classification": alloy_class,
        "suggested_applications": applications[:3],
    }


def filter_feasible(candidates: list[dict], constraints: dict) -> list[dict]:
    """
    Remove candidates where element fractions violate user-specified hard constraints.
    constraints: {"frac_C": {"max": 0.012}, "frac_Cr": {"min": 0.10}, ...}
    """
    feasible = []
    for c in candidates:
        comp = c.get("composition", {})
        valid = True
        for el_key, bounds in constraints.items():
            el = el_key.replace("frac_", "")
            val = comp.get(el, 0.0)
            if "min" in bounds and val < bounds["min"]:
                valid = False; break
            if "max" in bounds and val > bounds["max"]:
                valid = False; break
        if valid:
            feasible.append(c)
    return feasible
