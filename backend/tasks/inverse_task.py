from backend.tasks.celery_app import celery_app
import os
import json
import time
from datetime import datetime
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import InverseDesignJob
from backend.inverse.optimizer import AlloyOptimizer, ObjectiveTarget, GenerationResult
from backend.cache.redis_client import r, redis_available
from backend.cache.checkpoint_service import checkpoint_service

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./alloy_iq.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def _normalize_targets(targets_data) -> list[ObjectiveTarget]:
    """
    Normalizes target specifications from either:
    1. A list of dicts: [{"property": "yield_strength_mpa", "direction": "maximize", "min_val": 900}]
    2. A dictionary mapping: {"yield_strength": [">", 900]}
    """
    normalized = []
    
    if isinstance(targets_data, list):
        for item in targets_data:
            if isinstance(item, dict):
                normalized.append(ObjectiveTarget(
                    property_name=item.get("property") or item.get("property_name"),
                    direction=item.get("direction", "maximize"),
                    min_val=item.get("min_val"),
                    max_val=item.get("max_val"),
                    weight=item.get("weight", 1.0)
                ))
    elif isinstance(targets_data, dict):
        for prop, spec in targets_data.items():
            if isinstance(spec, list) and len(spec) >= 2:
                op, limit_val = spec[0], spec[1]
                direction = "maximize" if op in (">", ">=") else "minimize"
                min_val = limit_val if direction == "maximize" else None
                max_val = limit_val if direction == "minimize" else None
                normalized.append(ObjectiveTarget(
                    property_name=prop,
                    direction=direction,
                    min_val=min_val,
                    max_val=max_val,
                    weight=1.0
                ))
            elif isinstance(spec, dict):
                normalized.append(ObjectiveTarget(
                    property_name=prop,
                    direction=spec.get("direction", "maximize"),
                    min_val=spec.get("min_val"),
                    max_val=spec.get("max_val"),
                    weight=spec.get("weight", 1.0)
                ))
                
    return normalized

def _normalize_constraints(constraints_data) -> dict[str, tuple[float, float]]:
    """
    Normalizes constraint specifications into element-key pairs.
    Example: {"Cr": [15, 25]} -> {"frac_Cr": (0.15, 0.25)}
    """
    normalized = {}
    if not constraints_data:
        return normalized
        
    for k, v in constraints_data.items():
        # Ensure standard key structure: frac_ELEMENT
        key = k if k.startswith("frac_") else f"frac_{k}"
        if isinstance(v, list) and len(v) >= 2:
            min_val = v[0] / 100.0 if v[0] > 1.0 else v[0]
            max_val = v[1] / 100.0 if v[1] > 1.0 else v[1]
            normalized[key] = (float(min_val), float(max_val))
        elif isinstance(v, tuple) and len(v) >= 2:
            min_val = v[0] / 100.0 if v[0] > 1.0 else v[0]
            max_val = v[1] / 100.0 if v[1] > 1.0 else v[1]
            normalized[key] = (float(min_val), float(max_val))
            
    return normalized

@celery_app.task(bind=True, max_retries=1, time_limit=300, soft_time_limit=240)
def run_inverse_optimization(self, job_id: str):
    """
    Celery task that executes NSGA-II optimization, publishes generation
    milestones to Redis, and stores checkpoints in Postgres/SQLite database.
    Now supports rollback-safe cryptographic checkpoint verification.
    """
    logger.info("Initializing inverse design job: {}", job_id)
    db = SessionLocal()
    
    try:
        # 1. Fetch job row
        job = db.query(InverseDesignJob).filter(InverseDesignJob.id == job_id).first()
        if not job:
            logger.error("Job {} not found in database.", job_id)
            return {"status": "error", "message": "Job not found"}
            
        # 2. Check for rollback-safe recovery checkpoint
        recovered = checkpoint_service.rollback_recovery(db, job_id)
        recovered_state = None
        latest_checksum = None
        start_generation = 0
        
        if recovered:
            recovered_gen, recovered_state, latest_checksum = recovered
            start_generation = recovered_gen
            logger.info("Recovered checkpoint for job {} at generation {}", job_id, start_generation)
            
        # 3. Update status to running
        job.status = "running"
        job.current_generation = start_generation
        db.commit()
        
        # 4. Parse and normalize inputs
        normalized_targets = _normalize_targets(job.targets)
        normalized_constraints = _normalize_constraints(job.constraints)
        
        if not normalized_targets:
            raise ValueError("No targets specified or target formatting is invalid.")
            
        # 5. Initialize optimizer
        optimizer = AlloyOptimizer(
            targets=normalized_targets,
            constraints=normalized_constraints,
            alloy_family=job.alloy_family
        )
        
        n_gen = job.n_generations or 50
        pop_size = job.pop_size or 100
        
        logger.info("Starting GA: job={}, generations={}, pop_size={}", job_id, n_gen, pop_size)
        
        # 6. Broadcast starting/recovery signal
        start_payload = {
            "status": "running",
            "job_id": job_id,
            "generation": start_generation,
            "current_generation": start_generation,
            "total_generations": n_gen,
            "message": "Optimization started." if start_generation == 0 else f"Recovered from generation {start_generation}."
        }
        if redis_available:
            r.publish(f"job:progress:{job_id}", json.dumps(start_payload))
            
        # 7. Run optimizer generation-by-generation
        start_time = time.time()
        
        for gen_result in optimizer.run(n_generations=n_gen, pop_size=pop_size, initial_state=recovered_state):
            # Checkpoint DB and write binary state every 5 generations to limit DB write amplification
            if gen_result.generation % 5 == 0 or gen_result.generation == n_gen:
                try:
                    # Commit lightweight DB generation checkpoint
                    job.current_generation = gen_result.generation
                    job.latest_pareto_front = gen_result.pareto_front
                    db.commit()
                    
                    # Save binary checkpoint Msgpack + Zstd payload (previous_checksum links cryptographic chain)
                    if hasattr(gen_result, "checkpoint_state"):
                        cp = checkpoint_service.save_checkpoint(
                            db=db,
                            job_id=job_id,
                            generation=gen_result.generation,
                            state_data=gen_result.checkpoint_state,
                            previous_checksum=latest_checksum
                        )
                        latest_checksum = cp.checksum
                        
                        # Prune intermediate checkpoint assets to prevent disk bloat
                        checkpoint_service.cleanup_stale_checkpoints(db=db, job_id=job_id, keep_last=3)
                        
                except Exception as cp_err:
                    logger.warning("Checkpointing failed for job {} gen {}: {}", job_id, gen_result.generation, cp_err)
            
            # Formulate progress message
            progress_payload = {
                "status": "running",
                "job_id": job_id,
                "generation": gen_result.generation,
                "current_generation": gen_result.generation,
                "total_generations": n_gen,
                "best_fitness": gen_result.best_fitness,
                "pareto_front": gen_result.pareto_front,
                "elapsed_seconds": gen_result.elapsed_seconds,
                "constraint_violation_rate": gen_result.constraint_violation_rate
            }
            
            # Stream high-frequency milestones to Redis Pub/Sub
            if redis_available:
                r.publish(f"job:progress:{job_id}", json.dumps(progress_payload))
                
        # 8. Success persistence
        final_job = db.query(InverseDesignJob).filter(InverseDesignJob.id == job_id).first()
        if final_job:
            final_job.status = "done"
            final_job.pareto_front = gen_result.pareto_front
            final_job.latest_pareto_front = gen_result.pareto_front
            final_job.n_candidates = len(gen_result.pareto_front)
            final_job.current_generation = n_gen
            final_job.completed_at = datetime.utcnow()
            db.commit()
            
        # Send final completion message to subscribers
        complete_payload = {
            "status": "complete",
            "job_id": job_id,
            "total_generations": n_gen,
            "n_candidates": len(gen_result.pareto_front),
            "result": {
                "pareto_front": gen_result.pareto_front,
                "n_candidates": len(gen_result.pareto_front),
                "objective_axes": [t.property_name for t in normalized_targets]
            }
        }
        if redis_available:
            r.publish(f"job:progress:{job_id}", json.dumps(complete_payload))
            
        logger.info("Successfully finished job: {}", job_id)
        return {"status": "complete", "job_id": job_id}
        
    except Exception as exc:
        db.rollback()
        logger.exception("Error running inverse optimization for job {}", job_id)
        
        # Mark failure state in Database
        try:
            failed_job = db.query(InverseDesignJob).filter(InverseDesignJob.id == job_id).first()
            if failed_job:
                failed_job.status = "error"
                failed_job.error_msg = str(exc)
                db.commit()
        except Exception as db_exc_err:
            logger.error("Failed to mark job failure in DB: {}", db_exc_err)
            
        # Broadcast failure payload
        err_payload = {
            "status": "error",
            "job_id": job_id,
            "message": str(exc)
        }
        if redis_available:
            r.publish(f"job:progress:{job_id}", json.dumps(err_payload))
            
        raise exc
    finally:
        db.close()
