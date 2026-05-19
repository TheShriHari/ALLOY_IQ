import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

def compute_pdp(
    model,
    scaler,
    X_median: np.ndarray,
    X_lo: np.ndarray,
    X_hi: np.ndarray,
    feature_name: str,
    feature_names: list[str],
    n_points: int = 50,
) -> dict:
    """
    Compute partial dependence of all targets on one feature.
    Sweeps the feature from its p5 to p95 in n_points steps,
    holding all others at their training-set median.
    """
    feat_idx = feature_names.index(feature_name)
    lo = X_lo[feat_idx]
    hi = X_hi[feat_idx]
    sweep_vals = np.linspace(lo, hi, n_points)

    # Build synthetic samples: median for all features, sweep target feature
    X_sweep = np.tile(X_median, (n_points, 1))
    X_sweep[:, feat_idx] = sweep_vals

    # Predict (model expects scaled input)
    # Note: X_train already scaled — sweep vals are already in scaled space
    y_pred_log = model.predict(X_sweep)
    y_pred = np.expm1(y_pred_log)   # inverse log-transform

    target_names = ["yield_strength_mpa","tensile_strength_mpa","hardness_hv","elongation_pct"]

    # Map feature name to human-readable label
    label_map = {f"frac_{el}": f"{el_name} fraction" for el, el_name in {
        "C":"Carbon","Cr":"Chromium","Ni":"Nickel","Mo":"Molybdenum",
        "Mn":"Manganese","V":"Vanadium","Fe":"Iron","Al":"Aluminum",
    }.items()}

    return {
        "feature": feature_name,
        "x_values": sweep_vals.tolist(),
        "predictions": {
            name: y_pred[:, i].tolist()
            for i, name in enumerate(target_names)
        },
        "x_label": label_map.get(feature_name, feature_name),
        "x_unit": "mol fraction",
    }
