import pytest
import numpy as np
from datetime import datetime, timedelta

from backend.monitoring.metrics import MetricsCollector
from backend.monitoring.health_monitor import SystemHealthMonitor
from backend.monitoring.drift_monitor import DriftMonitor
from backend.monitoring.alert_manager import AlertManager, INFO, WARNING, CRITICAL
from backend.monitoring.dashboard_schema import DashboardBuilder
from backend.db.models import ModelTrainingJob

@pytest.fixture
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.db.models import Base
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

def test_metrics_aggregation_telemetry():
    """Ensure MetricsCollector records latencies, uncertainty width, and computes rates correctly."""
    collector = MetricsCollector()
    
    # Record counters
    collector.increment("predictions_total", 10)
    collector.increment("predictions_refused_total", 2)
    collector.increment("predictions_ood_total", 3)
    
    # Record latencies
    collector.record_latency("api_latency_seconds", 0.12)
    collector.record_latency("api_latency_seconds", 0.08)
    collector.record_latency("prediction_latency_seconds", 0.04)
    collector.record_latency("prediction_latency_seconds", 0.06)
    
    # Record uncertainties
    collector.record_uncertainty(45.0)
    collector.record_uncertainty(55.0)
    
    snapshot = collector.get_snapshot()
    
    assert snapshot["predictions_total"] == 10
    assert snapshot["predictions_refused_total"] == 2
    assert snapshot["predictions_ood_total"] == 3
    assert snapshot["refusal_rate"] == 0.20
    assert snapshot["ood_rate"] == 0.30
    assert 0.09 <= snapshot["api_latency_seconds_avg"] <= 0.11
    assert snapshot["conformal_width_avg"] == 50.0
    assert snapshot["conformal_width_max"] == 55.0


def test_drift_monitor_numerical_and_categorical():
    """Verify Kolmogorov-Smirnov (KS) and Population Stability Index (PSI) drift monitoring."""
    monitor = DriftMonitor()
    
    # 1. Numerical drift: baseline vs current
    np.random.seed(42)
    baseline = np.random.normal(loc=10.0, scale=1.0, size=1000)
    current_no_drift = np.random.normal(loc=10.0, scale=1.0, size=1000)
    current_with_drift = np.random.normal(loc=12.0, scale=1.0, size=1000)  # shifted mean
    
    # Run PSI
    psi_ok = monitor.calculate_psi(baseline, current_no_drift)
    psi_drift = monitor.calculate_psi(baseline, current_with_drift)
    
    assert psi_ok < 0.15
    assert psi_drift > 0.25
    
    # Run KS feature check
    res = monitor.detect_feature_drift(baseline.reshape(-1, 1), current_with_drift.reshape(-1, 1))
    assert res["drift_detected"] is True
    
    # 2. Categorical drift: alloy families
    base_cats = ["steel", "steel", "steel", "nickel", "nickel", "titanium"]
    curr_cats_ok = ["steel", "steel", "steel", "nickel", "nickel", "titanium"]
    curr_cats_drift = ["titanium", "titanium", "titanium", "titanium", "aluminum", "aluminum"]
    
    drift_ok = monitor.detect_alloy_family_drift(base_cats, curr_cats_ok)
    drift_bad = monitor.detect_alloy_family_drift(base_cats, curr_cats_drift)
    
    assert drift_ok["drift_detected"] is False
    assert drift_bad["drift_detected"] is True


def test_alert_manager_threshold_triggers():
    """Verify AlertManager triggers dynamic alerts across correct levels."""
    manager = AlertManager()
    
    # 1. Trigger OOD alert
    snapshot = {"predictions_total": 100, "predictions_refused_total": 5, "predictions_ood_total": 35, "refusal_rate": 0.05, "ood_rate": 0.35}
    manager.evaluate_metrics(snapshot)
    
    active_alerts = manager.get_active_alerts()
    assert len(active_alerts) == 1
    assert active_alerts[0]["event_type"] == "ood_spike"
    assert active_alerts[0]["tier"] == WARNING
    
    # 2. Trigger high refusal critical alert
    snapshot_critical = {"predictions_total": 100, "predictions_refused_total": 45, "predictions_ood_total": 5, "refusal_rate": 0.45, "ood_rate": 0.05}
    manager.evaluate_metrics(snapshot_critical)
    
    active_alerts_now = manager.get_active_alerts()
    assert len(active_alerts_now) == 2
    assert active_alerts_now[1]["event_type"] == "refusal_spike"
    assert active_alerts_now[1]["tier"] == CRITICAL


def test_health_monitor_heartbeats(db_session):
    """Assert SystemHealthMonitor correctly flags running background jobs with lost heartbeats."""
    # Insert a stale training job manually into DB
    job1 = ModelTrainingJob(
        id="job_healthy",
        alloy_family="steel",
        status="running",
        progress=0.5,
        heartbeat=datetime.utcnow()
    )
    job2 = ModelTrainingJob(
        id="job_stale",
        alloy_family="nickel",
        status="running",
        progress=0.2,
        heartbeat=datetime.utcnow() - timedelta(minutes=10)  # stale (exceeds 5 mins limit)
    )
    db_session.add(job1)
    db_session.add(job2)
    db_session.commit()
    
    monitor = SystemHealthMonitor(db_session=db_session)
    worker_check = monitor.check_worker_heartbeats(timeout_seconds=300)
    
    assert worker_check["status"] == "DEGRADED"
    assert worker_check["stale_jobs_count"] == 1
    assert "job_stale" in worker_check["stale_job_ids"]


def test_dashboard_builder_payload():
    """Verify DashboardBuilder compiles a fully compliant unified DashboardPayload schema."""
    metrics_snapshot = {
        "api_requests_total": 1000,
        "websocket_reconnects_total": 5,
        "checkpoint_recoveries_total": 2,
        "predictions_total": 500,
        "predictions_refused_total": 10,
        "predictions_ood_total": 15,
        "refusal_rate": 0.02,
        "ood_rate": 0.03,
        "api_latency_seconds_avg": 0.15,
        "prediction_latency_seconds_avg": 0.05,
        "conformal_width_avg": 52.0
    }
    
    health_diagnostics = {
        "overall_status": "UP",
        "timestamp": datetime.utcnow().isoformat(),
        "postgres": {"status": "UP"},
        "redis": {"status": "UP"},
        "celery_queue": {"backlog_count": 0},
        "worker_heartbeats": {"stale_jobs_count": 0}
    }
    
    audit_stats = {
        "total_audits": 500,
        "refusal_count": 10,
        "tier_distribution": {"LOW": 480, "MEDIUM": 5, "HIGH": 5, "REFUSE": 10},
        "flag_distribution": {"PHYSICS_VIOLATION": 10}
    }
    
    trend_series = [
        {
            "timestamp_hour": "2026-05-19T14:00:00Z",
            "refusal_rate": 0.02,
            "ood_rate": 0.03,
            "api_latency_avg": 0.15,
            "predictions_count": 500
        }
    ]
    
    payload = DashboardBuilder.compile_payload(
        metrics_snapshot=metrics_snapshot,
        health_diagnostics=health_diagnostics,
        audit_stats=audit_stats,
        trend_series=trend_series,
        alerts_count=1
    )
    
    # Assert type compliance
    assert payload.telemetry.api_requests_total == 1000
    assert payload.health.overall_status == "UP"
    assert payload.audit.total_audits == 500
    assert len(payload.trends) == 1
    assert payload.active_alerts_count == 1
