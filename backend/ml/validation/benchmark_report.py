import numpy as np
from typing import Dict, Any, List, Tuple
from loguru import logger

class BenchmarkReporter:
    """
    ML Benchmark Reporting Engine.
    Generates publication-ready metrics: R2, MAE, conformal predictive coverage,
    auto-refusal rates, and detailed alloy-family performance tables.
    """
    
    @staticmethod
    def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the standard coefficient of determination R²."""
        if len(y_true) < 2:
            return 0.0
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    @staticmethod
    def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Mean Absolute Error (MAE)."""
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Root Mean Squared Error (RMSE)."""
        if len(y_true) == 0:
            return 0.0
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        conformal_intervals: List[Tuple[float, float]],
        refusal_mask: List[bool],
        families: List[str],
        ood_mask: List[bool] = None
    ) -> Dict[str, Any]:
        """
        Processes model outcome arrays to compile statistical summaries.
        Applies refusal masking filters to exclude refused records from primary MAE/R2 calculations.
        """
        logger.info("Generating publication-ready ML Benchmark Report.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        refusal_mask = np.asarray(refusal_mask, dtype=bool)
        families = np.asarray(families)
        
        n_total = len(y_true)
        if ood_mask is None:
            ood_mask = np.zeros(n_total, dtype=bool)
        else:
            ood_mask = np.asarray(ood_mask, dtype=bool)
            
        n_refused = int(np.sum(refusal_mask))
        refusal_rate = float(n_refused / n_total) if n_total > 0 else 0.0
        
        n_ood = int(np.sum(ood_mask))
        ood_rate = float(n_ood / n_total) if n_total > 0 else 0.0
        
        # 1. Filter out refused predictions from standard regression metrics (decision-support gate)
        accepted_mask = ~refusal_mask
        y_true_acc = y_true[accepted_mask]
        y_pred_acc = y_pred[accepted_mask]
        
        global_r2 = self._compute_r2(y_true_acc, y_pred_acc)
        global_mae = self._compute_mae(y_true_acc, y_pred_acc)
        global_rmse = self._compute_rmse(y_true_acc, y_pred_acc)

        # 2. Compute Conformal Prediction Coverage
        # Check if true target falls in [lower, upper] interval
        covered_count = 0
        for idx, (lower, upper) in enumerate(conformal_intervals):
            if lower <= y_true[idx] <= upper:
                covered_count += 1
        coverage = float(covered_count / n_total) if n_total > 0 else 0.0

        # 3. Family-level Metrics Breakdown
        family_report = {}
        unique_f = set(families)
        for fam in unique_f:
            fam_clean = str(fam).strip().lower()
            fam_mask = families == fam
            
            # Sub-group splits
            y_true_f = y_true[fam_mask]
            y_pred_f = y_pred[fam_mask]
            ref_f = refusal_mask[fam_mask]
            ood_f = ood_mask[fam_mask]
            conf_f = [conformal_intervals[i] for i in range(n_total) if fam_mask[i]]
            
            acc_f_mask = ~ref_f
            
            # Per-family math
            fam_total = len(y_true_f)
            fam_refused = int(np.sum(ref_f))
            fam_ref_rate = float(fam_refused / fam_total) if fam_total > 0 else 0.0
            
            fam_ood = int(np.sum(ood_f))
            fam_ood_rate = float(fam_ood / fam_total) if fam_total > 0 else 0.0
            
            fam_r2 = self._compute_r2(y_true_f[acc_f_mask], y_pred_f[acc_f_mask])
            fam_mae = self._compute_mae(y_true_f[acc_f_mask], y_pred_f[acc_f_mask])
            fam_rmse = self._compute_rmse(y_true_f[acc_f_mask], y_pred_f[acc_f_mask])
            
            covered_f = 0
            for i, (lower, upper) in enumerate(conf_f):
                if lower <= y_true_f[i] <= upper:
                    covered_f += 1
            fam_cov = float(covered_f / fam_total) if fam_total > 0 else 0.0
            
            family_report[fam_clean] = {
                "samples": fam_total,
                "refusal_rate": fam_ref_rate,
                "ood_rate": fam_ood_rate,
                "r2": fam_r2,
                "mae": fam_mae,
                "rmse": fam_rmse,
                "conformal_coverage": fam_cov
            }
            
        report = {
            "total_samples": n_total,
            "refused_samples": n_refused,
            "refusal_rate": refusal_rate,
            "ood_rate": ood_rate,
            "global_accepted_r2": global_r2,
            "global_accepted_mae": global_mae,
            "global_accepted_rmse": global_rmse,
            "conformal_coverage": coverage,
            "family_breakdown": family_report
        }
        
        logger.info("Report compiled. Global Accepted R²: {:.3f}, MAE: {:.2f}, RMSE: {:.2f}, Coverage: {:.2f}%, Refusal: {:.2f}%", 
                    global_r2, global_mae, global_rmse, coverage * 100.0, refusal_rate * 100.0)
        return report
