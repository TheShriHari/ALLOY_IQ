import time
import threading
from typing import Dict, List, Any, Optional

class MetricsCollector:
    """
    Thread-safe registry for recording, tracking, and aggregating
    production system, pipeline, and machine learning quality metrics.
    """
    def __init__(self):
        self._lock = threading.Lock()
        
        # Counters
        self.counters: Dict[str, int] = {
            "api_requests_total": 0,
            "websocket_reconnects_total": 0,
            "checkpoint_recoveries_total": 0,
            "predictions_total": 0,
            "predictions_refused_total": 0,
            "predictions_ood_total": 0
        }
        
        # Latency lists (for average & percentile snapshots)
        self.latencies: Dict[str, List[float]] = {
            "api_latency_seconds": [],
            "celery_queue_delay_seconds": [],
            "prediction_latency_seconds": []
        }
        
        # Quality values (e.g. conformal intervals widths)
        self.uncertainties: Dict[str, List[float]] = {
            "conformal_width_mpa": []
        }

    def increment(self, name: str, value: int = 1):
        """Increments a counter in a thread-safe manner."""
        with self._lock:
            if name in self.counters:
                self.counters[name] += value
            else:
                self.counters[name] = value

    def record_latency(self, name: str, value_seconds: float):
        """Appends a latency observation."""
        with self._lock:
            if name in self.latencies:
                self.latencies[name].append(value_seconds)
                # Keep max 10,000 observations to prevent memory growth
                if len(self.latencies[name]) > 10000:
                    self.latencies[name] = self.latencies[name][-5000:]

    def record_uncertainty(self, width: float):
        """Records conformal intervals width statistics."""
        with self._lock:
            self.uncertainties["conformal_width_mpa"].append(width)
            if len(self.uncertainties["conformal_width_mpa"]) > 10000:
                self.uncertainties["conformal_width_mpa"] = self.uncertainties["conformal_width_mpa"][-5000:]

    def get_snapshot(self) -> Dict[str, Any]:
        """Generates a complete telemetry aggregation snapshot."""
        with self._lock:
            snapshot = {}
            
            # 1. Fetch counters
            for k, v in self.counters.items():
                snapshot[k] = v
                
            # Compute rates/ratios
            preds = self.counters["predictions_total"]
            refused = self.counters["predictions_refused_total"]
            ood = self.counters["predictions_ood_total"]
            
            snapshot["refusal_rate"] = float(refused / preds) if preds > 0 else 0.0
            snapshot["ood_rate"] = float(ood / preds) if preds > 0 else 0.0
            
            # 2. Compute latency averages
            for name, vals in self.latencies.items():
                if vals:
                    snapshot[f"{name}_avg"] = float(sum(vals) / len(vals))
                    snapshot[f"{name}_max"] = float(max(vals))
                    snapshot[f"{name}_count"] = len(vals)
                else:
                    snapshot[f"{name}_avg"] = 0.0
                    snapshot[f"{name}_max"] = 0.0
                    snapshot[f"{name}_count"] = 0
            
            # 3. Compute uncertainty statistics
            unc_vals = self.uncertainties["conformal_width_mpa"]
            if unc_vals:
                avg_unc = sum(unc_vals) / len(unc_vals)
                snapshot["conformal_width_avg"] = float(avg_unc)
                snapshot["conformal_width_max"] = float(max(unc_vals))
                snapshot["conformal_width_min"] = float(min(unc_vals))
            else:
                snapshot["conformal_width_avg"] = 0.0
                snapshot["conformal_width_max"] = 0.0
                snapshot["conformal_width_min"] = 0.0

            return snapshot

# Global singleton collector
metrics_collector = MetricsCollector()
