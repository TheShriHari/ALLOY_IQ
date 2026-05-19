"""
Connector layer wrapping the stacking ensemble prediction engine for inverse design.
Exposes a fast evaluator `predict_composition` completing in <5ms.
"""
from typing import Dict
import pandas as pd
from backend.ml.model_engine import AlloyModelEngine
from backend.data.features import FeatureEngineer

# Shared singleton for prediction speed
_engine = AlloyModelEngine()

def predict_composition(composition: Dict[str, float], alloy_family: str = "steel") -> Dict[str, float]:
    """
    Predict properties for a single composition dict.
    Returns:
        {
          "yield_strength_mpa": 850.0,
          "tensile_strength_mpa": 1020.0,
          "hardness_hv": 285.0,
          "elongation_pct": 14.2,
          "corrosion_pren": 38.2,   # if applicable
        }
    """
    row = dict(composition)
    df = pd.DataFrame([row])
    
    # 1. Transform features
    fe = FeatureEngineer(alloy_family)
    df_transformed = fe.transform(df)
    
    # 2. Get predictions
    res = _engine.predict(alloy_family, df_transformed)
    predictions = res["predictions"]
    
    # 3. Flatten predictions to mean values
    out = {
        name: details["mean"]
        for name, details in predictions.items()
    }
    
    # 4. Inject corrosion PREN
    if "corrosion_pren" not in out:
        out["corrosion_pren"] = res.get("corrosion_analysis", {}).get("pren_calculated", 0.0)
        
    return out
