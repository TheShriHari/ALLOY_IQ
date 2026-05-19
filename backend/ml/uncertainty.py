"""
Conformal prediction wrapper for ALLOY IQ.
Uses MAPIE (pip install mapie) to produce calibrated prediction intervals.
Coverage guarantee: 90% of true values fall within the returned interval.
"""
from mapie.regression import MapieRegressor
from mapie.multi_output import MapieMultiOutputRegressor
import numpy as np
import joblib
from pathlib import Path

class AlloyUncertainty:
    def __init__(self, base_model_path: str = None, base_model = None):
        if base_model_path:
            self.base_model = joblib.load(base_model_path)
        elif base_model:
            self.base_model = base_model
        else:
            raise ValueError("Provide either base_model_path or base_model")
        self.mapie = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10):
        """
        Calibrate conformal predictor on a held-out calibration set.
        alpha=0.10 → 90% coverage guarantee.
        X_cal: scaled feature matrix (run through RobustScaler first)
        y_cal: log-transformed target matrix
        """
        self.mapie = MapieMultiOutputRegressor(
            estimator=self.base_model,
            method="naive",     # fast; switch to "jackknife+" for tighter intervals on small data
            cv="prefit",        # base model already fitted — only calibrate conformity scores
        )
        self.mapie.fit(X_cal, y_cal)
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
        # mapie.predict with MapieMultiOutputRegressor returns (y_pred, y_pis)
        # y_pred: (n_samples, n_targets)
        # y_pis: (n_samples, n_targets, 2)
        y_pred_log, y_intervals_log = self.mapie.predict(X, alpha=self.alpha)

        target_names = ["yield_strength_mpa","tensile_strength_mpa","hardness_hv","elongation_pct"]
        result = {}
        for i, name in enumerate(target_names):
            result[name] = {
                "mean":  float(np.expm1(y_pred_log[0, i])),
                "lower": float(np.expm1(y_intervals_log[0, i, 0])),
                "upper": float(np.expm1(y_intervals_log[0, i, 1])),
            }
        return result

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AlloyUncertainty":
        return joblib.load(path)
