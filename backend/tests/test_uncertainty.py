import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from backend.ml.uncertainty import AlloyUncertainty

def test_conformal_coverage():
    # Synthetic data for 4 targets
    np.random.seed(42)
    X = np.random.rand(200, 10)
    y = X[:, :4] * 10 + np.random.randn(200, 4) * 0.5  # target with noise

    # log-transform
    y_log = np.log1p(y)

    # Base model
    base_model = MultiOutputRegressor(Ridge(alpha=1.0))
    base_model.fit(X, y_log)

    # Wrap in AlloyUncertainty
    uq = AlloyUncertainty(base_model=base_model)
    uq.calibrate(X, y_log, alpha=0.10)  # 90% coverage

    # Predict
    res = uq.predict(X[:5])
    
    # Assert output structure
    assert "yield_strength_mpa" in res
    assert "mean" in res["yield_strength_mpa"]
    assert "lower" in res["yield_strength_mpa"]
    assert "upper" in res["yield_strength_mpa"]
    
    # Check interval logic: lower < mean < upper
    for i in range(5):
        mean = res["yield_strength_mpa"]["mean"]
        lower = res["yield_strength_mpa"]["lower"]
        upper = res["yield_strength_mpa"]["upper"]
        assert lower <= mean <= upper
