import os
import json

def estimate_phase_fractions(composition: dict, predictions: dict) -> dict:
    """
    Physically informed heuristic to map nominal chemical composition and
    forward property predictions to approximate microstructure phases.
    """
    # Normalize composition keys to capitalize element symbols (e.g. C, Cr, Ni, Fe)
    comp = {}
    for k, v in composition.items():
        # Remove any leading 'frac_'
        elem = k.replace("frac_", "").capitalize()
        comp[elem] = v

    # Extract UTS or Yield Strength
    ys = 600.0
    if "yield_strength_mpa" in predictions:
        ys = predictions["yield_strength_mpa"]
    elif "predictions" in predictions:
        # Check inside nested dictionary
        preds = predictions["predictions"]
        if "yield_strength_mpa" in preds:
            ys = preds["yield_strength_mpa"].get("mean", 600.0)
        elif "yield_strength" in preds:
            ys = preds["yield_strength"].get("mean", 600.0)

    # Metallurgical heuristics for steel / HEA
    c_wt = comp.get("C", 0.0) * 100.0
    cr_wt = comp.get("Cr", 0.0) * 100.0
    ni_wt = comp.get("Ni", 0.0) * 100.0

    # Martensite fraction increases with strength and carbon content
    martensite = min(95.0, max(0.0, c_wt * 80.0 + (ys - 400.0) * 0.12))
    
    # Carbides increase with Cr and C
    carbide = min(20.0, max(1.0, cr_wt * 1.5 + c_wt * 10.0))
    
    # Austenite increases with Ni
    austenite = min(30.0, max(0.0, ni_wt * 2.5))
    
    # Ferrite is the soft remainder
    ferrite = max(0.0, 100.0 - martensite - carbide - austenite)
    
    # Re-normalize to exactly 100%
    tot = martensite + carbide + austenite + ferrite
    if tot > 0:
        martensite = round((martensite / tot) * 100.0, 2)
        carbide = round((carbide / tot) * 100.0, 2)
        austenite = round((austenite / tot) * 100.0, 2)
        ferrite = round((ferrite / tot) * 100.0, 2)

    # Grain size decreases as yield strength increases (Hall-Petch relation)
    grain_size = max(5.0, min(80.0, 50.0 - (ys - 300.0) * 0.05))

    return {
        "martensite_pct": martensite,
        "ferrite_pct": ferrite,
        "carbide_pct": carbide,
        "austenite_pct": austenite,
        "grain_size_um": round(grain_size, 2),
    }


def get_generator_path() -> str:
    """Return absolute path to microstructure generator script."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try using v2 first if SciPy is available, fallback to v1 otherwise
    v2_path = os.path.join(current_dir, "microstructure_generator_v2.py")
    v1_path = os.path.join(current_dir, "microstructure_generator.py")
    
    try:
        import scipy
        if os.path.exists(v2_path):
            return v2_path
    except ImportError:
        pass
        
    return v1_path
