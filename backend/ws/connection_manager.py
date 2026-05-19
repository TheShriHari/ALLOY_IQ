import asyncio
import json
import time
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
import redis.asyncio as aioredis
from sqlalchemy.orm import Session
from backend.db.models import InverseDesignJob
from backend.config.settings import settings

class WebSocketConnectionManager:
    """
    Stateless connection manager coordinating WebSocket clients and Redis Pub/Sub.
    Maintains zero active connection state in memory, allowing horizontal scaling
    across multiple web worker processes.
    """
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.heartbeat_interval = 12.0  # Send ping frame every 12 seconds
        
    async def handle_connection(self, websocket: WebSocket, job_id: str, db: Session):
        """
        Main lifecycle handler for a WebSocket connection.
        Recovers active state from DB, streams live updates from Redis pub/sub,
        and manages heartbeat/stale cleanup.
        """
        await websocket.accept()
        logger.info("WebSocket client connected for job: {}", job_id)
        
        # 1. Recovery Check (PostgreSQL/SQLite Fallback Source of Truth)
        try:
            job = db.query(InverseDesignJob).filter(InverseDesignJob.id == job_id).first()
            if not job:
                await websocket.send_json({"status": "error", "message": "Job not found"})
                await websocket.close()
                return
                
            # If the job is already done, send result and close cleanly
            if job.status == "done":
                logger.info("Job {} already complete. Sending final recovery state.", job_id)
                # Build final axes
                axes = []
                if job.targets and isinstance(job.targets, dict):
                    axes = list(job.targets.keys())
                elif job.targets and isinstance(job.targets, list):
                    axes = [t.get("property") for t in job.targets if t.get("property")]
                    
                await websocket.send_json({
                    "status": "complete",
                    "job_id": job_id,
                    "total_generations": job.n_generations,
                    "n_candidates": job.n_candidates or 0,
                    "result": {
                        "pareto_front": job.pareto_front or [],
                        "n_candidates": job.n_candidates or 0,
                        "objective_axes": axes
                    }
                })
                await websocket.close()
                return
                
            # If the job has errored, send error and close
            elif job.status == "error":
                logger.info("Job {} in error state. Sending final error recovery state.", job_id)
                await websocket.send_json({
                    "status": "error",
                    "job_id": job_id,
                    "message": job.error_msg or "Unknown background task error"
                })
                await websocket.close()
                return
                
            # Recovery Mode: Send the last stored database generation checkpoint
            # so the browser can immediately render current progress upon refresh/recovery
            else:
                logger.info("Job {} is active ({}). Initiating browser recovery payload.", job_id, job.status)
                await websocket.send_json({
                    "status": "running",
                    "job_id": job_id,
                    "generation": job.current_generation or 0,
                    "current_generation": job.current_generation or 0,
                    "total_generations": job.n_generations or 50,
                    "pareto_front": job.latest_pareto_front or [],
                    "message": "Reconnected. Recovered active job progress."
                })
                
        except Exception as err:
            logger.exception("Error checking job recovery state in DB: {}", err)
            try:
                await websocket.send_json({"status": "error", "message": "Failed to verify job status"})
                await websocket.close()
            except Exception:
                pass
            return

        # 2. Redis Pub/Sub Forwarding Loop & Heartbeat Handler
        redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
        pubsub = redis_client.pubsub()
        channel_name = f"job:progress:{job_id}"
        
        await pubsub.subscribe(channel_name)
        logger.info("Subscribed to Redis channel: {}", channel_name)
        
        last_ping_time = time.time()
        
        try:
            while True:
                now = time.time()
                
                # Heartbeat: Send ping payload to verify client socket liveness
                if now - last_ping_time >= self.heartbeat_interval:
                    try:
                        await websocket.send_json({"type": "ping", "timestamp": int(now)})
                        last_ping_time = now
                    except Exception:
                        logger.warning("WebSocket client silent disconnect detected for job {}", job_id)
                        break
                        
                # Read next transient message from Redis with short timeout
                try:
                    # Timeout of 1.0 second allows connection heartbeat check to execute regularly
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=1.0
                    )
                    
                    if message:
                        raw_data = message["data"]
                        payload = json.loads(raw_data)
                        
                        # Forward real-time message to client WebSocket
                        await websocket.send_json(payload)
                        
                        # Auto-disconnect once job terminal state is received
                        if payload.get("status") in ("complete", "error"):
                            logger.info("Job {} complete. Closing WebSocket cleanly.", job_id)
                            break
                            
                except asyncio.TimeoutError:
                    # Normal timeout; continue loop so heartbeat ping runs
                    continue
                    
        except WebSocketDisconnect:
            logger.info("Client disconnected from WebSocket for job {}", job_id)
        except Exception as e:
            logger.exception("Error running WebSocket broadcast loop for job {}: {}", job_id, e)
        finally:
            # 3. Clean up subscription resources & close connections
            try:
                logger.info("Cleaning up subscription resources for job {}", job_id)
                await pubsub.unsubscribe(channel_name)
                await pubsub.close()
                await redis_client.close()
            except Exception as clean_err:
                logger.error("Failed to clean up Redis client for job {}: {}", job_id, clean_err)
                
            try:
                await websocket.close()
            except Exception:
                pass
                
# Global instance
connection_manager = WebSocketConnectionManager()
