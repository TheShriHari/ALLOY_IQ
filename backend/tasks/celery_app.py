from celery import Celery
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "alloyiq",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.tasks.render_task", "backend.tasks.inverse_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,   # results expire after 1 hour
    task_routes={
        "backend.tasks.render_task.*": {"queue": "render_queue"},
        "backend.tasks.inverse_task.*": {"queue": "inverse_queue"},
        "tasks.render_task.*": {"queue": "render_queue"},
        "tasks.inverse_task.*": {"queue": "inverse_queue"},
    },
)
