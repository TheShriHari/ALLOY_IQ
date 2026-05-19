from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class RealtimeTelemetry(BaseModel):
    """Schema representing current operational and statistical telemetry."""
    api_requests_total: int = Field(..., description="Total API requests counted.")
    websocket_reconnects_total: int = Field(..., description="Total socket reconnections.")
    checkpoint_recoveries_total: int = Field(..., description="Total rollback checkpoint recoveries.")
    predictions_total: int = Field(..., description="Total property predictions computed.")
    predictions_refused_total: int = Field(..., description="Total predictions intercepted/blocked.")
    predictions_ood_total: int = Field(..., description="Total predictions sitting in OOD space.")
    refusal_rate: float = Field(..., description="Prediction refusal rate (0 to 1).")
    ood_rate: float = Field(..., description="Incidence rate of OOD alloys.")
    api_latency_avg: float = Field(..., description="Average API request duration.")
    prediction_latency_avg: float = Field(..., description="Average prediction computation latency.")
    conformal_width_avg: float = Field(..., description="Average conformal intervals width.")

class HealthCheckStatus(BaseModel):
    """Schema representing diagnostics checks status."""
    overall_status: str = Field(..., description="UP, DEGRADED, or DOWN status.")
    timestamp: str = Field(..., description="Diagnostics ISO execution time.")
    postgres_status: str = Field(..., description="PostgreSQL status.")
    redis_status: str = Field(..., description="Redis status.")
    celery_backlog: int = Field(..., description="Pending tasks in celery queue.")
    stale_jobs: int = Field(..., description="Active training jobs with stale heartbeats.")

class AuditSummary(BaseModel):
    """Telemetry schema summarizing prediction reliability audits."""
    total_audits: int = Field(..., description="Total predictions audited.")
    refusal_count: int = Field(..., description="Refused predictions.")
    tier_distribution: Dict[str, int] = Field(..., description="Counts of LOW, MEDIUM, HIGH, and REFUSE risk tiers.")
    flag_distribution: Dict[str, int] = Field(..., description="Counts of triggered risk flags.")

class TrendMetricPoint(BaseModel):
    """Representing a historical aggregated telemetry record."""
    timestamp_hour: str = Field(..., description="ISO hourly timestamp.")
    refusal_rate: float
    ood_rate: float
    api_latency_avg: float
    predictions_count: int

class DashboardPayload(BaseModel):
    """Unified payload schema parsed by frontend web clients."""
    telemetry: RealtimeTelemetry
    health: HealthCheckStatus
    audit: AuditSummary
    trends: List[TrendMetricPoint]
    active_alerts_count: int

class DashboardBuilder:
    """Helper class to aggregate telemetry states and build dashboard payload snapshots."""
    
    @staticmethod
    def compile_payload(
        metrics_snapshot: Dict[str, Any],
        health_diagnostics: Dict[str, Any],
        audit_stats: Dict[str, Any],
        trend_series: List[Dict[str, Any]],
        alerts_count: int
    ) -> DashboardPayload:
        """Aggregates variables to output a fully validated DashboardPayload."""
        
        telemetry = RealtimeTelemetry(
            api_requests_total=metrics_snapshot.get("api_requests_total", 0),
            websocket_reconnects_total=metrics_snapshot.get("websocket_reconnects_total", 0),
            checkpoint_recoveries_total=metrics_snapshot.get("checkpoint_recoveries_total", 0),
            predictions_total=metrics_snapshot.get("predictions_total", 0),
            predictions_refused_total=metrics_snapshot.get("predictions_refused_total", 0),
            predictions_ood_total=metrics_snapshot.get("predictions_ood_total", 0),
            refusal_rate=metrics_snapshot.get("refusal_rate", 0.0),
            ood_rate=metrics_snapshot.get("ood_rate", 0.0),
            api_latency_avg=metrics_snapshot.get("api_latency_seconds_avg", 0.0),
            prediction_latency_avg=metrics_snapshot.get("prediction_latency_seconds_avg", 0.0),
            conformal_width_avg=metrics_snapshot.get("conformal_width_avg", 0.0)
        )
        
        health = HealthCheckStatus(
            overall_status=health_diagnostics.get("overall_status", "UP"),
            timestamp=health_diagnostics.get("timestamp", ""),
            postgres_status=health_diagnostics.get("postgres", {}).get("status", "UP"),
            redis_status=health_diagnostics.get("redis", {}).get("status", "UP"),
            celery_backlog=health_diagnostics.get("celery_queue", {}).get("backlog_count", 0),
            stale_jobs=health_diagnostics.get("worker_heartbeats", {}).get("stale_jobs_count", 0)
        )
        
        audit = AuditSummary(
            total_audits=audit_stats.get("total_audits", 0),
            refusal_count=audit_stats.get("refusal_count", 0),
            tier_distribution=audit_stats.get("tier_distribution", {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "REFUSE": 0}),
            flag_distribution=audit_stats.get("flag_distribution", {})
        )
        
        trends = []
        for pt in trend_series:
            trends.append(TrendMetricPoint(
                timestamp_hour=pt.get("timestamp_hour", ""),
                refusal_rate=pt.get("refusal_rate", 0.0),
                ood_rate=pt.get("ood_rate", 0.0),
                api_latency_avg=pt.get("api_latency_avg", 0.0),
                predictions_count=pt.get("predictions_count", 0)
            ))
            
        return DashboardPayload(
            telemetry=telemetry,
            health=health,
            audit=audit,
            trends=trends,
            active_alerts_count=alerts_count
        )
