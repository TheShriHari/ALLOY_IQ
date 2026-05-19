"""
WebSocket endpoint for real-time inverse design optimization streaming.
Streams GenerationResult objects as JSON after each GA generation.

Frontend connects to: ws://localhost:8000/ws/jobs/{job_id}
Streams: one JSON message per generation in real-time.
"""

from fastapi import APIRouter, WebSocket, Depends
from sqlalchemy.orm import Session
from backend.ws.connection_manager import connection_manager
from backend.tasks.render_task import SessionLocal

router = APIRouter()

# Dependency for DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.websocket("/ws/jobs/{job_id}")
async def job_progress_websocket(websocket: WebSocket, job_id: str, db: Session = Depends(get_db)):
    """
    WebSocket handler for live progress tracking.
    Supports browser refresh, connection recovery, and multiple backend server nodes.
    """
    await connection_manager.handle_connection(websocket, job_id, db)
