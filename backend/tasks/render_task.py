from backend.tasks.celery_app import celery_app
import subprocess
import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.models import RenderJob

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./alloy_iq.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

@celery_app.task(bind=True, max_retries=2, soft_time_limit=120)
def render_microstructure(self, job_id: str, composition: dict, predictions: dict):
    """
    Celery task: runs Blender headless render and saves output PNG.
    Executed in background — never blocks the FastAPI thread.
    """
    from blender.microstructure_bridge import estimate_phase_fractions, get_generator_path

    db = SessionLocal()
    try:
        # Update job status in DB
        job = db.query(RenderJob).filter(RenderJob.id == job_id).first()
        if job:
            job.status = "running"
            db.commit()

        # Compute phase fractions using metallurgical bridge heuristics
        phase_fractions = estimate_phase_fractions(composition, predictions)
        
        # Configure output paths
        output_dir = os.path.join("frontend", "public", "renders")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.abspath(os.path.join(output_dir, f"{job_id}.png"))

        # CLI inputs
        cli_args = {
            **phase_fractions,
            "output_path": output_path,
            "seed": 42
        }

        generator_script = get_generator_path()
        blender_bin = os.getenv("BLENDER_PATH", "blender")

        # Command: blender --background --python <script> -- <json_args>
        cmd = [
            blender_bin,
            "--background",
            "--python",
            generator_script,
            "--",
            json.dumps(cli_args)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90
        )

        if result.returncode != 0:
            raise Exception(f"Blender failed: {result.stderr or result.stdout}")

        # Update DB job to complete
        if job:
            job.status = "complete"
            job.image_url = f"/renders/{job_id}.png"
            db.commit()

        return {"status": "complete", "image_url": f"/renders/{job_id}.png"}

    except Exception as exc:
        db.rollback()
        # Update job status in DB to failed
        job = db.query(RenderJob).filter(RenderJob.id == job_id).first()
        if job:
            job.status = "failed"
            db.commit()
        # Retry logic
        try:
            self.retry(exc=exc, countdown=10)
        except Exception:
            raise exc
    finally:
        db.close()
