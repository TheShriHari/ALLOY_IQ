"""
Generates plain-English metallurgical narratives from SHAP values.
Output is what a senior materials scientist would say about the prediction.
"""

ELEMENT_NAMES = {
    "C": "carbon", "Cr": "chromium", "Ni": "nickel", "Mo": "molybdenum",
    "Mn": "manganese", "V": "vanadium", "Nb": "niobium", "Si": "silicon",
    "W": "tungsten", "Co": "cobalt", "Ti": "titanium", "Al": "aluminum",
    "Cu": "copper", "N": "nitrogen", "B": "boron", "S": "sulfur", "P": "phosphorus",
    "Fe": "iron (base matrix)", "Zn": "zinc", "Mg": "magnesium",
}

MECHANISMS = {
    # Element → property → mechanism explanation
    ("C",  "yield_strength_mpa"):    "interstitial solid-solution strengthening and carbide precipitation",
    ("C",  "hardness_hv"):           "martensite formation and carbide precipitation hardening",
    ("C",  "elongation_pct"):        "increased brittleness from carbide networks (inverse relationship)",
    ("Cr", "hardness_hv"):           "secondary hardening via M7C3 and M23C6 carbide formation",
    ("Cr", "corrosion_pren"):        "passive oxide layer formation (Cr₂O₃) — PREN increase",
    ("Mo", "yield_strength_mpa"):    "solid-solution strengthening and secondary hardening",
    ("Mo", "corrosion_pren"):        "enhanced pitting resistance in chloride environments",
    ("Ni", "elongation_pct"):        "austenite stabilization improving ductility",
    ("Mn", "yield_strength_mpa"):    "solid-solution strengthening and hardenability improvement",
    ("Nb", "yield_strength_mpa"):    "grain refinement via NbC precipitation (Hall-Petch effect)",
    ("V",  "yield_strength_mpa"):    "vanadium carbide precipitation hardening",
    ("N",  "corrosion_pren"):        "PREN contribution: 16×N coefficient in pitting resistance formula",
    ("S",  "elongation_pct"):        "MnS inclusion embrittlement (negative effect)",
    ("S",  "yield_strength_mpa"):    "MnS inclusion weakening (detrimental)",
}

MAGNITUDE_WORDS = [
    (100, "dominant"),
    (50,  "significant"),
    (20,  "moderate"),
    (5,   "minor"),
    (0,   "negligible"),
]

def _magnitude_word(shap_abs: float) -> str:
    for threshold, word in MAGNITUDE_WORDS:
        if shap_abs >= threshold:
            return word
    return "negligible"

def _element_from_feature(feature_name: str) -> str | None:
    """Extract element symbol from feature name like 'frac_C' or 'frac_Cr'."""
    if feature_name.startswith("frac_"):
        return feature_name[5:]
    return None

def generate_narrative(
    shap_dict: dict[str, float],
    prediction: dict,
    intervals: dict,
    target: str = "yield_strength_mpa",
) -> str:
    """
    Generate a 3-sentence plain-English explanation of a prediction.

    Args:
        shap_dict: {feature_name: shap_value} sorted by abs value descending
        prediction: {"mean": 820.0, "lower": 720.0, "upper": 910.0}
        intervals:  same structure — used for confidence sentence
        target:     which property is being explained

    Returns:
        A 3-sentence string ready for display in the UI.
    """
    target_label = {
        "yield_strength_mpa": "yield strength",
        "tensile_strength_mpa": "tensile strength",
        "hardness_hv": "Vickers hardness",
        "elongation_pct": "elongation",
        "corrosion_pren": "pitting resistance (PREN)",
    }.get(target, target)

    units = {
        "yield_strength_mpa": "MPa",
        "tensile_strength_mpa": "MPa",
        "hardness_hv": "HV",
        "elongation_pct": "%",
        "corrosion_pren": "PREN units",
    }.get(target, "")

    # Find top positive and negative SHAP contributors
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    positives = [(k, v) for k, v in sorted_shap if v > 0]
    negatives = [(k, v) for k, v in sorted_shap if v < 0]

    sentences = []

    # Sentence 1: dominant positive driver
    if positives:
        feat, val = positives[0]
        el = _element_from_feature(feat)
        el_name = ELEMENT_NAMES.get(el, feat) if el else feat
        magnitude = _magnitude_word(abs(val))
        mechanism = MECHANISMS.get((el, target), "compositional effect") if el else "compositional effect"
        sentences.append(
            f"{el_name.capitalize()} is the {magnitude} positive driver of {target_label}, "
            f"contributing +{abs(val):.1f} {units} above baseline through {mechanism}."
        )

    # Sentence 2: dominant negative driver (if meaningful)
    if negatives and abs(negatives[0][1]) > 5:
        feat, val = negatives[0]
        el = _element_from_feature(feat)
        el_name = ELEMENT_NAMES.get(el, feat) if el else feat
        mechanism = MECHANISMS.get((el, target), "compositional effect") if el else "compositional effect"
        sentences.append(
            f"Conversely, {el_name} is the largest risk factor, "
            f"reducing predicted {target_label} by {abs(val):.1f} {units} via {mechanism}."
        )
    elif len(positives) > 1:
        # Secondary positive driver instead
        feat, val = positives[1]
        el = _element_from_feature(feat)
        el_name = ELEMENT_NAMES.get(el, feat) if el else feat
        sentences.append(
            f"Secondary reinforcement comes from {el_name} (+{abs(val):.1f} {units}), "
            f"with remaining elements having combined minor influence."
        )

    # Sentence 3: confidence context
    mean_val = prediction["mean"]
    lower = prediction["lower"]
    upper = prediction["upper"]
    interval_width = upper - lower
    avg_width = mean_val * 0.18   # 18% of mean is roughly average interval width for these properties
    if interval_width < avg_width * 1.2:
        confidence_text = (
            f"The 90% prediction interval is [{lower:.0f}, {upper:.0f}] {units} — "
            f"narrow relative to the mean, indicating this composition is well-represented in training data."
        )
    elif interval_width < avg_width * 2.0:
        confidence_text = (
            f"The 90% prediction interval is [{lower:.0f}, {upper:.0f}] {units}. "
            f"Moderate uncertainty suggests this composition sits near the boundary of well-studied alloys — "
            f"physical validation is recommended."
        )
    else:
        confidence_text = (
            f"⚠ Wide prediction interval [{lower:.0f}, {upper:.0f}] {units}. "
            f"This composition is extrapolating beyond well-studied alloy space. "
            f"Treat as directional guidance only — physical testing is required."
        )
    sentences.append(confidence_text)

    return " ".join(sentences)


def generate_full_report(shap_dicts: dict, predictions: dict, intervals: dict) -> dict:
    """Generate narratives for all predicted properties."""
    return {
        target: generate_narrative(shap_dicts.get(target, {}), predictions[target], predictions[target], target)
        for target in predictions
    }
