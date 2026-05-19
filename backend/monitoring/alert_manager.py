import time
import threading
from typing import Dict, List, Any, Optional

# Standard tiers
INFO = "INFO"
WARNING = "WARNING"
CRITICAL = "CRITICAL"

class AlertManager:
    """
    Evaluates telemetry thresholds, manages severity rankings, and dispatches structured
    operational notifications regarding statistical drift, failures, and capacity backlogs.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.alerts: List[Dict[str, Any]] = []

    def trigger_alert(self, tier: str, event_type: str, message: str):
        """Creates and logs a new system alert."""
        alert_item = {
            "timestamp": time.time(),
            "tier": tier,
            "event_type": event_type,
            "message": message
        }
        with self._lock:
            self.alerts.append(alert_item)
            # Limit stored alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]
            
        print(f"[{tier}] {event_type.upper()}: {message}")

    def evaluate_metrics(self, snapshot: Dict[str, Any]):
        """Evaluates operational thresholds and raises alerts dynamically."""
        # 1. Refusal rates
        refusal_rate = snapshot.get("refusal_rate", 0.0)
        if refusal_rate > 0.20:
            self.trigger_alert(
                CRITICAL if refusal_rate > 0.40 else WARNING,
                "refusal_spike",
                f"Elevated model refusal rate detected: {refusal_rate*100:.1f}%."
            )

        # 2. OOD rates
        ood_rate = snapshot.get("ood_rate", 0.0)
        if ood_rate > 0.30:
            self.trigger_alert(
                CRITICAL if ood_rate > 0.50 else WARNING,
                "ood_spike",
                f"Elevated out-of-distribution (OOD) incidence rate: {ood_rate*100:.1f}%."
            )

    def evaluate_health(self, diagnostics: Dict[str, Any]):
        """Evaluates diagnostics reports and triggers operational alerts."""
        overall = diagnostics.get("overall_status", "UP")
        
        # 1. Core infrastructure downs
        if overall == "DOWN":
            pg_status = diagnostics.get("postgres", {}).get("status", "UP")
            redis_status = diagnostics.get("redis", {}).get("status", "UP")
            
            if pg_status == "DOWN":
                self.trigger_alert(CRITICAL, "db_failure", "PostgreSQL database instance is down or unreachable.")
            if redis_status == "DOWN":
                self.trigger_alert(CRITICAL, "redis_failure", "Redis broker/cache service is unreachable.")

        # 2. Queue delays
        backlog = diagnostics.get("celery_queue", {}).get("backlog_count", 0)
        if backlog > 10:
            self.trigger_alert(
                CRITICAL if backlog > 50 else WARNING,
                "queue_backlog",
                f"Celery queue backlog exceeds capacity limit. Pending: {backlog} tasks."
            )

        # 3. Lost heartbeats
        stale_count = diagnostics.get("worker_heartbeats", {}).get("stale_jobs_count", 0)
        if stale_count > 0:
            self.trigger_alert(
                WARNING,
                "stale_worker",
                f"Detected {stale_count} background worker training jobs with lost heartbeats."
            )

    def evaluate_drift(self, drift_results: Dict[str, Any], context: str = "feature"):
        """Evaluates statistical drift values and issues alerts."""
        psi = drift_results.get("psi", 0.0)
        drift_detected = drift_results.get("drift_detected", False)
        
        if psi > 0.25 or drift_detected:
            self.trigger_alert(
                CRITICAL if psi > 0.50 else WARNING,
                "drift_detected",
                f"Significant statistical drift detected in {context} distribution (PSI: {psi:.2f})."
            )

    def get_active_alerts(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries recent system alerts filtered optionally by tier."""
        with self._lock:
            if tier:
                return [a for a in self.alerts if a["tier"] == tier]
            return list(self.alerts)
