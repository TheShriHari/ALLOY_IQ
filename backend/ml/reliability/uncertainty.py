import numpy as np
from typing import Dict, Tuple
from loguru import logger

class AlloyConformalPredictor:
    """
    Split Conformal Prediction engine.
    Calibrates prediction intervals utilizing residuals of a validation/calibration split.
    Provides distribution-free coverage guarantees for property predictions.
    """
    def __init__(self):
        self.residuals = None
        self.calibrated_quantiles = {}

    def fit(self, y_true_calib: np.ndarray, y_pred_calib: np.ndarray):
        """
        Calibrates the nonconformity scores using the residual difference |y - y_pred|
        over the calibration dataset split.
        """
        logger.info("Calibrating AlloyConformalPredictor with {} samples.", len(y_true_calib))
        
        # Nonconformity score = absolute prediction error
        self.residuals = np.abs(y_true_calib - y_pred_calib)
        
        # Pre-calculate common significance quantiles: e.g. alpha = 0.10 (90%), 0.05 (95%)
        for alpha in [0.05, 0.10, 0.20]:
            self.calibrated_quantiles[alpha] = self._compute_conformal_quantile(alpha)
            
        logger.info(
            "Conformal quantiles computed - 95% Conf (alpha=0.05): {:.4f}, 90% Conf (alpha=0.10): {:.4f}",
            self.calibrated_quantiles.get(0.05),
            self.calibrated_quantiles.get(0.10)
        )
        return self

    def _compute_conformal_quantile(self, alpha: float) -> float:
        """
        Computes the exact conformal quantile with finite sample correction:
        q = np.percentile(residuals, 100 * (1 - alpha) * (n + 1) / n)
        """
        n = len(self.residuals)
        if n == 0:
            return 0.0
            
        # Target coverage with finite-sample inflation
        coverage_level = (1.0 - alpha) * (n + 1) / n
        # Bound coverage level between 0 and 1 to prevent indexing errors
        coverage_level = min(max(coverage_level, 0.0), 1.0)
        
        return float(np.percentile(self.residuals, coverage_level * 100))

    def predict_interval(self, y_pred: float, alpha: float = 0.10) -> Tuple[float, float, float]:
        """
        Calculates the calibrated prediction interval: [y_pred - q, y_pred + q].
        Returns: (lower_bound, upper_bound, interval_width)
        """
        if self.residuals is None:
            raise ValueError("AlloyConformalPredictor has not been calibrated/fitted!")
            
        # Fallback if specific alpha was not pre-calibrated
        quantile = self.calibrated_quantiles.get(alpha)
        if quantile is None:
            quantile = self._compute_conformal_quantile(alpha)
            
        lower_bound = y_pred - quantile
        upper_bound = y_pred + quantile
        width = 2.0 * quantile
        
        return lower_bound, upper_bound, width

    def validate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.10) -> float:
        """
        Computes empirical interval coverage over an independent test validation fold.
        Empirical coverage should ideally be >= 1 - alpha.
        """
        quantile = self.calibrated_quantiles.get(alpha)
        if quantile is None:
            quantile = self._compute_conformal_quantile(alpha)
            
        lower_bounds = y_pred - quantile
        upper_bounds = y_pred + quantile
        
        covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        coverage_pct = float(np.mean(covered))
        
        logger.info(
            "Empirical validation coverage at alpha={:.2f} is {:.2f}% (Target: {:.2f}%)",
            alpha,
            coverage_pct * 100.0,
            (1.0 - alpha) * 100.0
        )
        return coverage_pct
