from backend.ml.narrative import generate_narrative

def test_generate_narrative_carbon():
    shap_dict = {
        "frac_C": 142.3,
        "frac_Cr": 38.1,
        "frac_S": -18.4,
    }
    prediction = {"mean": 820.0, "lower": 720.0, "upper": 910.0}
    intervals = prediction
    target = "yield_strength_mpa"
    
    narrative = generate_narrative(shap_dict, prediction, intervals, target)
    
    assert "Carbon" in narrative or "carbon" in narrative
    assert "interstitial" in narrative
    assert "sulfur" in narrative
    assert "142.3" in narrative
    assert "18.4" in narrative
    assert "720" in narrative
    assert "910" in narrative
