import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from backend.ml.pdp import compute_pdp

def test_pdp_monotonic_increase():
    # Synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    
    # y increases with X[:, 0] (frac_C)
    y1 = X[:, 0] * 100 + X[:, 1] * 10
    y_all = np.vstack([y1, y1, y1, y1]).T  # 4 targets
    
    y_log = np.log1p(y_all)
    
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y_log)
    
    feature_names = ["frac_C", "frac_Cr", "frac_Ni", "frac_Mo", "frac_Mn"]
    
    X_median = np.median(X, axis=0)
    X_lo = np.percentile(X, 5, axis=0)
    X_hi = np.percentile(X, 95, axis=0)
    
    res = compute_pdp(
        model=model,
        scaler=None,  # Not used in our mock
        X_median=X_median,
        X_lo=X_lo,
        X_hi=X_hi,
        feature_name="frac_C",
        feature_names=feature_names,
        n_points=10,
    )
    
    assert res["feature"] == "frac_C"
    
    ys_preds = res["predictions"]["yield_strength_mpa"]
    
    # Check monotonic increase
    for i in range(1, len(ys_preds)):
        assert ys_preds[i] > ys_preds[i-1]
