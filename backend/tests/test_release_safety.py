import pytest
import os
import shutil

from backend.release.pre_release_validator import PreReleaseValidator
from backend.release.deployment_guard import DeploymentGuard, DeploymentBlockedException
from backend.release.release_registry import ReleaseRegistry
from backend.release.smoke_tests import ReleaseSmokeTester

@pytest.fixture
def clean_registry():
    ledger_path = "backend/release/test_release_ledger.json"
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
    
    registry = ReleaseRegistry(ledger_path=ledger_path)
    try:
        yield registry
    finally:
        if os.path.exists(ledger_path):
            os.remove(ledger_path)

def test_pre_release_validation_thresholds():
    """Ensure PreReleaseValidator enforces minimum benchmark performance thresholds."""
    validator = PreReleaseValidator(min_r2=0.75, max_mae=25.0)
    
    # 1. Clean metrics
    good_metrics = {"r2": 0.80, "mae": 20.0}
    is_valid, warnings = validator.verify_benchmark_thresholds(good_metrics)
    assert is_valid is True
    assert len(warnings) == 0
    
    # 2. Corrupted R2
    bad_r2 = {"r2": 0.70, "mae": 20.0}
    is_valid_r, warnings_r = validator.verify_benchmark_thresholds(bad_r2)
    assert is_valid_r is False
    assert any("R² performance level too low" in w for w in warnings_r)
    
    # 3. Corrupted MAE
    bad_mae = {"r2": 0.80, "mae": 28.0}
    is_valid_m, warnings_m = validator.verify_benchmark_thresholds(bad_mae)
    assert is_valid_m is False
    assert any("MAE performance level too high" in w for w in warnings_m)


def test_deployment_blocking_regressions():
    """Assert DeploymentGuard blocks deployments if candidate model regresses >5%."""
    guard = DeploymentGuard(regression_limit_pct=5.0)
    
    active_metrics = {"r2": 0.80, "mae": 20.0}
    
    # 1. candidate under 5% regression (e.g. 2% drop in R2, 3% increase in MAE)
    candidate_ok = {"r2": 0.79, "mae": 20.4}
    violations_ok = guard.check_regression(active_metrics, candidate_ok)
    assert len(violations_ok) == 0
    
    # 2. candidate over 5% regression (e.g. 6% drop in R2)
    candidate_bad_r2 = {"r2": 0.75, "mae": 20.0}
    violations_bad = guard.check_regression(active_metrics, candidate_bad_r2)
    assert len(violations_bad) > 0
    assert "R² performance degraded" in violations_bad[0]


def test_deployment_blocking_telemetry_and_smoke_failures():
    """Assert DeploymentGuard blocks deployments on telemetry spikes, severe drift, or smoke failures."""
    guard = DeploymentGuard()
    
    active_metrics = {"r2": 0.80, "mae": 20.0}
    candidate_metrics = {"r2": 0.80, "mae": 20.0}
    
    # 1. Safe telemetry
    safe_metrics = {"refusal_rate": 0.05, "ood_rate": 0.10}
    safe_drift = {"psi": 0.08}
    
    # Assert it clears safely
    guard.verify_and_gate_deployment(
        active_metrics=active_metrics,
        candidate_metrics=candidate_metrics,
        metrics_snapshot=safe_metrics,
        drift_snapshot=safe_drift,
        smoke_tests_passed=True
    )
    
    # 2. Spiked refusal rate blocks deployment
    bad_refusal = {"refusal_rate": 0.25, "ood_rate": 0.10}
    with pytest.raises(DeploymentBlockedException) as exc_info:
        guard.verify_and_gate_deployment(
            active_metrics=active_metrics,
            candidate_metrics=candidate_metrics,
            metrics_snapshot=bad_refusal,
            drift_snapshot=safe_drift,
            smoke_tests_passed=True
        )
    assert "Refusal rate is too high" in str(exc_info.value)
    
    # 3. Failed smoke tests blocks deployment
    with pytest.raises(DeploymentBlockedException) as exc_info_s:
        guard.verify_and_gate_deployment(
            active_metrics=active_metrics,
            candidate_metrics=candidate_metrics,
            metrics_snapshot=safe_metrics,
            drift_snapshot=safe_drift,
            smoke_tests_passed=False  # Smoke failure
        )
    assert "Smoke testing suite failed" in str(exc_info_s.value)


def test_release_tracking_ledger(clean_registry):
    """Verify ReleaseRegistry saves release metadata and retrieves the latest version correctly."""
    registry = clean_registry
    
    registry.register_release(
        release_id="rel_1.0.0",
        model_hash="sha256_mock_1",
        dataset_hash="dataset_hash_1",
        git_commit="git_mock_commit_1",
        benchmark_summary={"r2": 0.82, "mae": 19.5}
    )
    
    registry.register_release(
        release_id="rel_1.1.0",
        model_hash="sha256_mock_2",
        dataset_hash="dataset_hash_2",
        git_commit="git_mock_commit_2",
        benchmark_summary={"r2": 0.85, "mae": 18.2}
    )
    
    latest = registry.get_latest_release()
    assert latest is not None
    assert latest["release_id"] == "rel_1.1.0"
    assert latest["model_hash"] == "sha256_mock_2"
    
    # Load by specific ID
    rel_first = registry.get_release("rel_1.0.0")
    assert rel_first["git_commit"] == "git_mock_commit_1"


def test_smoke_tester_execution():
    """Verify ReleaseSmokeTester covers FastAPI API endpoints, socket connections, and task routing checks."""
    tester = ReleaseSmokeTester()
    passed, logs = tester.execute_all_smoke_tests()
    
    assert passed is True, f"Smoke tests failed: {logs}"
    assert logs["websocket"]["passed"] is True, f"WS check failed: {logs['websocket']}"
    assert logs["celery"]["passed"] is True, f"Celery check failed: {logs['celery']}"
    assert logs["model_loading"]["passed"] is True, f"Model loading failed: {logs['model_loading']}"
