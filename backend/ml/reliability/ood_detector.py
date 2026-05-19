import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from loguru import logger

class AlloyOODDetector:
    """
    Materials informatics OOD Detection engine.
    Computes Mahalanobis distance and Local Outlier Factor (LOF)
    over composition + processing feature spaces to evaluate covariate shift.
    """
    def __init__(self, regularization: float = 1e-5):
        self.regularization = regularization
        self.mu = None
        self.inv_cov = None
        self.lof = None
        self.mahalanobis_threshold_mild = 0.0
        self.mahalanobis_threshold_severe = 0.0

    def fit(self, X: np.ndarray):
        """
        Fits OOD models to training feature representations,
        computing multivariate means, covariance structures, and LOF matrices.
        """
        logger.info("Fitting AlloyOODDetector to feature matrix of shape {}", X.shape)
        
        # 1. Compute Mahalanobis Parameters
        self.mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        
        # Add Ridge regularization to prevent singular/ill-conditioned matrix inversion issues
        if cov.ndim == 0:
            cov = np.array([[cov]])
        identity = np.eye(cov.shape[0])
        regularized_cov = cov + self.regularization * identity
        
        try:
            self.inv_cov = np.linalg.inv(regularized_cov)
        except np.linalg.LinAlgError:
            self.inv_cov = np.linalg.pinv(regularized_cov)
            logger.warning("Covariance matrix inversion failed. Fallback to Pseudo-Inverse.")

        # Compute training distances to establish empirical thresholds (95th & 99th percentiles)
        train_dists = [self.compute_mahalanobis(x) for x in X]
        self.mahalanobis_threshold_mild = np.percentile(train_dists, 95.0)
        self.mahalanobis_threshold_severe = np.percentile(train_dists, 99.0)
        
        logger.info(
            "Calibrated Mahalanobis thresholds - Mild (95th): {:.4f}, Severe (99th): {:.4f}",
            self.mahalanobis_threshold_mild,
            self.mahalanobis_threshold_severe
        )

        # 2. Fit Local Outlier Factor (LOF)
        # Novelty parameter lets us run predict/score_samples on fresh test records
        self.lof = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1), novelty=True)
        self.lof.fit(X)
        
        return self

    def compute_mahalanobis(self, x: np.ndarray) -> float:
        """Computes the Mahalanobis distance of sample x to the training distribution."""
        diff = x - self.mu
        dist = np.sqrt(diff.T @ self.inv_cov @ diff)
        return float(dist)

    def evaluate_ood(self, x: np.ndarray) -> dict:
        """
        Evaluates OOD status of test vector x.
        Returns OOD scores, LOF metrics, and classification thresholds.
        """
        if self.mu is None or self.inv_cov is None or self.lof is None:
            raise ValueError("AlloyOODDetector has not been fitted to training data!")
            
        m_dist = self.compute_mahalanobis(x)
        
        # LOF returns negative outlier factor (higher = normal, lower/negative = outlier)
        # We standardise it so that positive numbers represent normal regions
        lof_score = float(self.lof.score_samples(x.reshape(1, -1))[0])
        is_lof_outlier = bool(self.lof.predict(x.reshape(1, -1))[0] == -1)

        is_mild = m_dist > self.mahalanobis_threshold_mild
        is_severe = m_dist > self.mahalanobis_threshold_severe

        return {
            "mahalanobis_distance": m_dist,
            "mahalanobis_threshold_mild": self.mahalanobis_threshold_mild,
            "mahalanobis_threshold_severe": self.mahalanobis_threshold_severe,
            "lof_score": lof_score,
            "is_lof_outlier": is_lof_outlier,
            "is_mild_ood": is_mild,
            "is_severe_ood": is_severe
        }
