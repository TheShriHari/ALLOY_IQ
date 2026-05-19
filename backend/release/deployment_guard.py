from typing import Dict, Any, List

class DeploymentBlockedException(Exception):
    """Exception raised when a release fails safety checks and cannot be deployed."""
    pass

class DeploymentGuard:
    """
    Enforces quality checks during deployment phase.
    Analyzes historical trends, current metrics, drift anomalies, and smoke test status,
    and blocks deployment immediately if regressions or spikes are detected.
    """
    def __init__(self, regression_limit_pct: float = 5.0):
        self.regression_limit_pct = regression_limit_pct

    def check_regression(self, active_metrics: Dict[str, float], candidate_metrics: Dict[str, float]) -> List[str]:
        """Blocks release if the candidate model's performance metrics degrade >5%."""
        violations = []
        
        # Compare R2 (higher is better)
        active_r2 = active_metrics.get("r2", 0.0)
        cand_r2 = candidate_metrics.get("r2", 0.0)
        if active_r2 > 0:
            deg_r2_pct = ((active_r2 - cand_r2) / active_r2) * 100
            if deg_r2_pct > self.regression_limit_pct:
                violations.append(f"R² performance degraded by {deg_r2_pct:.2f}% (Limit: {self.regression_limit_pct}%).")
                
        # Compare MAE (lower is better)
        active_mae = active_metrics.get("mae", 0.0)
        cand_mae = candidate_metrics.get("mae", 0.0)
        if active_mae > 0:
            deg_mae_pct = ((cand_mae - active_mae) / active_mae) * 100
            if deg_mae_pct > self.regression_limit_pct:
                violations.append(f"MAE degraded (increased) by {deg_mae_pct:.2f}% (Limit: {self.regression_limit_pct}%).")
                
        return violations

    def audit_telemetry_safety(
        self,
        metrics_snapshot: Dict[str, Any],
        drift_snapshot: Dict[str, Any],
        smoke_tests_passed: bool
    ) -> List[str]:
        """Audits current live metrics and checks for failures or drift."""
        violations = []

        # 1. Refusal rates spike check
        refusal_rate = metrics_snapshot.get("refusal_rate", 0.0)
        if refusal_rate > 0.20:
            violations.append(f"Safety Gate Blocked: Refusal rate is too high ({refusal_rate*100:.1f}% > 20.0%).")

        # 2. OOD rates spike check
        ood_rate = metrics_snapshot.get("ood_rate", 0.0)
        if ood_rate > 0.30:
            violations.append(f"Safety Gate Blocked: Out-of-Distribution rate is elevated ({ood_rate*100:.1f}% > 30.0%).")

        # 3. Statistical drift check
        psi = drift_snapshot.get("psi", 0.0)
        if psi > 0.25:
            violations.append(f"Safety Gate Blocked: Severe statistical data drift detected (PSI: {psi:.3f} > 0.250).")

        # 4. Smoke test execution
        if not smoke_tests_passed:
            violations.append("Safety Gate Blocked: Smoke testing suite failed on target environment.")

        return violations

    def verify_and_gate_deployment(
        self,
        active_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        metrics_snapshot: Dict[str, Any],
        drift_snapshot: Dict[str, Any],
        smoke_tests_passed: bool
    ):
        """Runs overall validations. Raises DeploymentBlockedException on any failures."""
        violations = []
        
        # 1. Performance regression check
        violations.extend(self.check_regression(active_metrics, candidate_metrics))
        
        # 2. Production telemetry & drift check
        violations.extend(self.audit_telemetry_safety(metrics_snapshot, drift_snapshot, smoke_tests_passed))
        
        if violations:
            msg = "Deployment blocked due to safety violations:\n" + "\n".join(f"- {v}" for v in violations)
            raise DeploymentBlockedException(msg)
            
        print("✓ All deployment gate safety verification criteria passed. Release is cleared for production deployment.")
