"""
Conformal prediction wrapper for ALLOY IQ.
Uses MAPIE (SplitConformalRegressor) to produce calibrated prediction intervals.
Coverage guarantee: 90% of true values fall within the returned interval.
"""
from mapie.regression import SplitConformalRegressor
import numpy as np
import joblib

class SingleTargetWrapper:
    """Wrapper to expose a single output target from a multi-output model for MAPIE."""
    def __init__(self, multi_output_model, target_idx: int):
        self.multi_output_model = multi_output_model
        self.target_idx = target_idx
        self._estimator_type = "regressor"
        self.n_features_in_ = getattr(multi_output_model, "n_features_in_", 10)
        self.fitted_ = True

    def fit(self, X, y):
        pass

    def predict(self, X):
        preds = self.multi_output_model.predict(X)
        return preds[:, self.target_idx]


class AlloyUncertainty:
    def __init__(self, base_model_path: str = None, base_model = None):
        if base_model_path:
            self.base_model = joblib.load(base_model_path)
        elif base_model:
            self.base_model = base_model
        else:
            raise ValueError("Provide either base_model_path or base_model")
        self.mapies = []

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10):
        """
        Calibrate conformal predictor on a held-out calibration set.
        alpha=0.10 → 90% coverage guarantee.
        X_cal: scaled feature matrix
        y_cal: log-transformed target matrix (n_samples, 4)
        """
        self.mapies = []
        for idx in range(4):
            wrapper = SingleTargetWrapper(self.base_model, idx)
            # confidence_level = 1.0 - alpha
            mapie = SplitConformalRegressor(
                estimator=wrapper,
                confidence_level=1.0 - alpha,
                prefit=True
            )
            mapie.conformalize(X_cal, y_cal[:, idx])
            self.mapies.append(mapie)
        self.alpha = alpha

    def predict(self, X: np.ndarray) -> dict:
        """
        Returns point estimates + 90% prediction intervals for all targets.
        All values inverse-log-transformed back to original units.

        Returns:
            {
              "yield_strength_mpa":    {"mean": 820.0, "lower": 720.0, "upper": 910.0},
              "tensile_strength_mpa":  {"mean": 1020.0, "lower": 890.0, "upper": 1130.0},
              "hardness_hv":           {"mean": 285.0, "lower": 240.0, "upper": 320.0},
              "elongation_pct":        {"mean": 14.2,  "lower": 9.8,   "upper": 18.6},
            }
        """
        target_names = ["yield_strength_mpa","tensile_strength_mpa","hardness_hv","elongation_pct"]
        result = {}
        
        for idx, name in enumerate(target_names):
            mapie = self.mapies[idx]
            # predict_interval returns (y_pred, y_pis) where y_pis is of shape (n_samples, 2, 1)
            y_pred_log, y_intervals_log = mapie.predict_interval(X)
            
            result[name] = {
                "mean":  float(np.expm1(y_pred_log[0])),
                "lower": float(np.expm1(y_intervals_log[0, 0, 0])),
                "upper": float(np.expm1(y_intervals_log[0, 1, 0])),
            }
        return result

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AlloyUncertainty":
        return joblib.load(path)
