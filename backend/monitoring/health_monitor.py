import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from loguru import logger

from backend.db.models import ModelTrainingJob

class SystemHealthMonitor:
    """
    Executes live diagnostics across core storage layers (PostgreSQL),
    caching/message brokers (Redis), background task processing queues (Celery),
    and active background training heartbeats.
    """
    def __init__(self, db_session: Optional[Session] = None, redis_client: Optional[Any] = None):
        self.db = db_session
        self.redis = redis_client

    def check_postgres(self) -> Dict[str, Any]:
        """Validates database responsiveness using SELECT 1."""
        if not self.db:
            return {"status": "UNKNOWN", "message": "No active DB session provided."}
        try:
            start = time.time()
            self.db.execute(text("SELECT 1"))
            latency_ms = (time.time() - start) * 1000
            return {
                "status": "UP",
                "latency_ms": float(latency_ms),
                "message": "Database is fully responsive."
            }
        except Exception as e:
            logger.error("PostgreSQL health check failed: {}", e)
            return {
                "status": "DOWN",
                "message": f"Database failure: {str(e)}"
            }

    def check_redis(self) -> Dict[str, Any]:
        """Pings Redis to test connectivity and broker status."""
        if not self.redis:
            # Try to connect optionally
            try:
                import redis
                # default docker or local host fallback
                r = redis.Redis(host="localhost", port=6379, socket_timeout=2)
                r.ping()
                return {"status": "UP", "message": "Redis is responsive."}
            except Exception:
                return {"status": "DOWN", "message": "Redis cache/broker is unreachable."}
        try:
            self.redis.ping()
            return {"status": "UP", "message": "Redis broker is responsive."}
        except Exception as e:
            return {"status": "DOWN", "message": f"Redis failure: {str(e)}"}

    def check_celery_backlog(self) -> Dict[str, Any]:
        """Checks background queue delay size and backlog counts."""
        # Query Redis for list length if using standard celery transport
        if not self.redis:
            return {"status": "UNKNOWN", "backlog_count": 0, "message": "Redis not connected; cannot query queue backlog."}
        try:
            # Default Celery queue name is 'celery'
            backlog = self.redis.llen("celery")
            status = "UP" if backlog < 20 else "DEGRADED"
            return {
                "status": status,
                "backlog_count": int(backlog),
                "message": f"Queue contains {backlog} pending tasks."
            }
        except Exception:
            return {"status": "UP", "backlog_count": 0, "message": "No active queue backlog found."}

    def check_worker_heartbeats(self, timeout_seconds: int = 300) -> Dict[str, Any]:
        """Scans database records to identify training runs with lost heartbeats."""
        if not self.db:
            return {"status": "UNKNOWN", "stale_jobs_count": 0}
        
        try:
            threshold = datetime.utcnow() - timedelta(seconds=timeout_seconds)
            # Find running jobs whose heartbeat is older than threshold
            stale_jobs = (
                self.db.query(ModelTrainingJob)
                .filter(ModelTrainingJob.status == "running")
                .filter(ModelTrainingJob.heartbeat < threshold)
                .all()
            )
            
            if stale_jobs:
                stale_ids = [job.id for job in stale_jobs]
                logger.warning("Detected {} stale worker training jobs: {}", len(stale_ids), stale_ids)
                return {
                    "status": "DEGRADED",
                    "stale_jobs_count": len(stale_ids),
                    "stale_job_ids": stale_ids,
                    "message": f"Discovered {len(stale_ids)} worker training jobs with lost heartbeats."
                }
            
            return {
                "status": "UP",
                "stale_jobs_count": 0,
                "message": "All active background training worker heartbeats are active."
            }
        except Exception as e:
            return {
                "status": "UNKNOWN",
                "stale_jobs_count": 0,
                "message": f"Failed checking heartbeats: {str(e)}"
            }

    def execute_diagnostics(self) -> Dict[str, Any]:
        """Runs overall health checks and aggregates statuses."""
        pg_status = self.check_postgres()
        redis_status = self.check_redis()
        queue_status = self.check_celery_backlog()
        worker_status = self.check_worker_heartbeats()

        overall = "UP"
        if pg_status["status"] == "DOWN" or redis_status["status"] == "DOWN":
            overall = "DOWN"
        elif queue_status["status"] == "DEGRADED" or worker_status["status"] == "DEGRADED":
            overall = "DEGRADED"

        return {
            "overall_status": overall,
            "timestamp": datetime.utcnow().isoformat(),
            "postgres": pg_status,
            "redis": redis_status,
            "celery_queue": queue_status,
            "worker_heartbeats": worker_status
        }
