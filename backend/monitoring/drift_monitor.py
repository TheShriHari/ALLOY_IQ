import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.stats import ks_2samp

class DriftMonitor:
    """
    Monitors statistical changes and data/concept drift in feature spaces,
    predictions distributions, alloy-family categories, and processing routes.
    """

    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        """
        Computes the Population Stability Index (PSI) between baseline and current populations.
        PSI < 0.1: No significant change.
        PSI 0.1 to 0.25: Moderate shift/drift.
        PSI > 0.25: Critical drift / high shift.
        """
        if len(baseline) == 0 or len(current) == 0:
            return 0.0

        # Create deciles/buckets based on baseline
        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(baseline, percentiles)
        buckets[0] = -np.inf
        buckets[-1] = np.inf

        # Calculate counts
        baseline_counts, _ = np.histogram(baseline, bins=buckets)
        current_counts, _ = np.histogram(current, bins=buckets)

        # Convert to percentages with epsilon smoothing
        eps = 1e-4
        b_pct = (baseline_counts / len(baseline)) + eps
        c_pct = (current_counts / len(current)) + eps

        # Re-normalize
        b_pct /= sum(b_pct)
        c_pct /= sum(c_pct)

        # Compute PSI
        psi_value = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
        return float(psi_value)

    @staticmethod
    def calculate_categorical_psi(baseline_cats: List[str], current_cats: List[str]) -> float:
        """Computes PSI across categorical classes with epsilon smoothing."""
        if not baseline_cats or not current_cats:
            return 0.0

        all_cats = list(set(baseline_cats + current_cats))
        
        # Calculate frequencies
        b_counts = {c: baseline_cats.count(c) for c in all_cats}
        c_counts = {c: current_cats.count(c) for c in all_cats}

        total_b = len(baseline_cats)
        total_c = len(current_cats)

        eps = 1e-4
        psi_value = 0.0
        for c in all_cats:
            b_pct = (b_counts[c] / total_b) + eps
            c_pct = (c_counts[c] / total_c) + eps
            psi_value += (c_pct - b_pct) * np.log(c_pct / b_pct)

        return float(psi_value)

    def detect_feature_drift(self, baseline_features: np.ndarray, current_features: np.ndarray) -> Dict[str, Any]:
        """
        Audits numerical feature arrays for statistical drift using KS-testing.
        baseline_features/current_features have shape (N, D).
        """
        if baseline_features.ndim == 1:
            baseline_features = baseline_features.reshape(-1, 1)
        if current_features.ndim == 1:
            current_features = current_features.reshape(-1, 1)

        num_feats = baseline_features.shape[1]
        drifted_features_count = 0
        feature_reports = []

        for idx in range(num_feats):
            b_feat = baseline_features[:, idx]
            c_feat = current_features[:, idx]

            # Run Kolmogorov-Smirnov test (2-sample)
            ks_stat, p_value = ks_2samp(b_feat, c_feat)
            psi = self.calculate_psi(b_feat, c_feat)

            # Standard p-value threshold is 0.05
            is_drifted = bool(p_value < 0.05 and psi > 0.25)
            if is_drifted:
                drifted_features_count += 1

            feature_reports.append({
                "feature_index": idx,
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "psi": float(psi),
                "drift_detected": is_drifted
            })

        overall_drift = drifted_features_count / num_feats > 0.30 if num_feats > 0 else False

        return {
            "drift_detected": overall_drift,
            "drifted_features_count": drifted_features_count,
            "total_features_count": num_feats,
            "feature_reports": feature_reports
        }

    def detect_prediction_drift(self, baseline_preds: np.ndarray, current_preds: np.ndarray) -> Dict[str, Any]:
        """Checks prediction output distributions (e.g. Yield/Tensile) for concept drift."""
        ks_stat, p_value = ks_2samp(baseline_preds, current_preds)
        psi = self.calculate_psi(baseline_preds, current_preds)

        return {
            "drift_detected": bool(p_value < 0.05 and psi > 0.25),
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "psi": float(psi)
        }

    def detect_alloy_family_drift(self, baseline_families: List[str], current_families: List[str]) -> Dict[str, Any]:
        """Audits composition classes shift."""
        psi = self.calculate_categorical_psi(baseline_families, current_families)
        return {
            "drift_detected": bool(psi > 0.25),
            "psi": psi
        }

    def detect_processing_route_drift(self, baseline_routes: List[str], current_routes: List[str]) -> Dict[str, Any]:
        """Audits processing route shifts."""
        psi = self.calculate_categorical_psi(baseline_routes, current_routes)
        return {
            "drift_detected": bool(psi > 0.25),
            "psi": psi
        }
