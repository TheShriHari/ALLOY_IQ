import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session
from loguru import logger

from backend.db.models import ModelTrainingJob
from backend.ml.lifecycle.trainer import ModelTrainer
from backend.ml.lifecycle.model_registry import ModelRegistry

# Ensure celery can be loaded optionally so tests don't strictly require a running broker
try:
    from backend.tasks.celery_app import celery_app
except ImportError:
    class MockCelery:
        def task(self, *args, **kwargs):
            return lambda fn: fn
    celery_app = MockCelery()

class TrainingScheduler:
    """
    Manages scheduling background Celery training runs, issuing heartbeat updates,
    and enabling resumable fold recovery checks.
    """
    def __init__(self, jobs_dir: str = "./data/jobs"):
        self.jobs_dir = os.path.abspath(jobs_dir)
        os.makedirs(self.jobs_dir, exist_ok=True)

    def get_checkpoint_path(self, job_id: str, fold: int) -> str:
        return os.path.join(self.jobs_dir, f"{job_id}_fold_{fold}.json")

    def save_fold_checkpoint(self, job_id: str, fold: int, checkpoint_metrics: Dict[str, Any]):
        """Caches intermediate fold metrics locally to enable resume recovery."""
        path = self.get_checkpoint_path(job_id, fold)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_metrics, f, indent=4)
        logger.info("Saved local fold checkpoint for job {} fold {}", job_id, fold)

    def load_fold_checkpoint(self, job_id: str, fold: int) -> Optional[Dict[str, Any]]:
        """Loads cached fold checkpoint if present."""
        path = self.get_checkpoint_path(job_id, fold)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def execute_training_job(
        self,
        job_id: str,
        df: pd.DataFrame,
        features: List[str],
        target_columns: List[str],
        group_column: str,
        coverage: str = "rich",
        n_splits: int = 3,
        db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Executes model training with progress heartbeat updates and fold recovery.
        """
        logger.info("Executing training job {} for alloy family", job_id)
        
        # 1. Update DB to RUNNING
        if db_session:
            job_rec = db_session.query(ModelTrainingJob).filter_by(id=job_id).first()
            if job_rec:
                job_rec.status = "running"
                job_rec.progress = 0.0
                job_rec.heartbeat = datetime.utcnow()
                db_session.commit()

        trainer = ModelTrainer(confidence=0.90)
        registry = ModelRegistry()

        fold_checkpoints = []

        # Heartbeat helper
        def update_heartbeat(fold_idx: int, checkpoint_data: Dict[str, Any]):
            self.save_fold_checkpoint(job_id, fold_idx, checkpoint_data)
            progress_pct = float(fold_idx + 1) / n_splits
            logger.info("Heartbeat: Job {} fold {} finished (progress={:.1f}%)", job_id, fold_idx, progress_pct * 100)
            
            if db_session:
                rec = db_session.query(ModelTrainingJob).filter_by(id=job_id).first()
                if rec:
                    rec.progress = progress_pct
                    rec.heartbeat = datetime.utcnow()
                    db_session.commit()

        # Resumable GroupKFold execution loop
        try:
            # We build our GroupKFold folds manually to check checkpoints
            from sklearn.model_selection import GroupKFold
            trainer.verify_features(df, features)
            
            X = df[features]
            y = df[target_columns]
            groups = df[group_column]
            gkf = GroupKFold(n_splits=n_splits)

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
                # Check for cached checkpoint
                cached = self.load_fold_checkpoint(job_id, fold)
                if cached is not None:
                    logger.info("Resuming Job {} fold {}: Loaded cached checkpoint metrics.", job_id, fold)
                    fold_checkpoints.append(cached)
                    continue

                # Not cached: run fit
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                from backend.ml.lifecycle.ensemble_manager import EnsembleManager
                fold_manager = EnsembleManager(coverage=coverage, confidence=0.90)
                
                cal_split = int(len(X_tr) * 0.20)
                if cal_split < 2:
                    cal_split = 2
                X_tr_fit, y_tr_fit = X_tr.iloc[:-cal_split], y_tr.iloc[:-cal_split]
                X_tr_cal, y_tr_cal = X_tr.iloc[-cal_split:], y_tr.iloc[-cal_split:]

                fold_manager.fit_and_calibrate(X_tr_fit, y_tr_fit, X_tr_cal, y_tr_cal)

                y_pred_log = fold_manager.pipeline.predict(X_val)
                import numpy as np
                y_pred = np.expm1(y_pred_log)

                fold_metrics = {}
                for idx, target in enumerate(target_columns):
                    ss_res = np.sum((y_val.iloc[:, idx] - y_pred[:, idx]) ** 2)
                    ss_tot = np.sum((y_val.iloc[:, idx] - y_val.iloc[:, idx].mean()) ** 2)
                    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    mae = float(np.mean(np.abs(y_val.iloc[:, idx] - y_pred[:, idx])))
                    fold_metrics[f"{target}_r2"] = float(r2)
                    fold_metrics[f"{target}_mae"] = float(mae)

                chk = {
                    "fold": fold,
                    "metrics": fold_metrics
                }
                fold_checkpoints.append(chk)
                update_heartbeat(fold, chk)

            # Fit production model
            logger.info("Fitting final production ensemble.")
            prod_manager = EnsembleManager(coverage=coverage, confidence=0.90)
            cal_split = int(len(X) * 0.20)
            if cal_split < 2:
                cal_split = 2
            X_fit, y_fit = X.iloc[:-cal_split], y.iloc[:-cal_split]
            X_cal, y_cal = X.iloc[-cal_split:], y.iloc[-cal_split:]

            prod_manager.fit_and_calibrate(X_fit, y_fit, X_cal, y_cal)

            import numpy as np
            prod_pred_log = prod_manager.pipeline.predict(X_cal)
            prod_pred = np.expm1(prod_pred_log)

            global_metrics = {
                "n_train_total": len(df),
                "fold_checkpoints": fold_checkpoints,
            }
            for idx, target in enumerate(target_columns):
                ss_res = np.sum((y_cal.iloc[:, idx] - prod_pred[:, idx]) ** 2)
                ss_tot = np.sum((y_cal.iloc[:, idx] - y_cal.iloc[:, idx].mean()) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                mae = float(np.mean(np.abs(y_cal.iloc[:, idx] - prod_pred[:, idx])))
                global_metrics[f"{target}_r2"] = float(r2)
                global_metrics[f"{target}_mae"] = float(mae)

            # Serialize model bytes for registration
            import io
            import joblib
            pipe_buffer = io.BytesIO()
            joblib.dump(prod_manager.pipeline, pipe_buffer)
            pipe_bytes = pipe_buffer.getvalue()

            conf_bytes = b""
            if prod_manager.conformal:
                conf_buffer = io.BytesIO()
                prod_manager.conformal.save(conf_buffer)
                conf_bytes = conf_buffer.getvalue()

            dataset_hash = trainer.registry.compute_dataset_hash(df)
            feature_hash = trainer.registry.compute_feature_hash(features)

            # Log to model registry
            alloy_family = str(df.iloc[0].get("alloy_family", "steel")).strip().lower()
            registry.register_model(
                model_id=f"job_{job_id}",
                pipeline_bytes=pipe_bytes,
                conformal_bytes=conf_bytes,
                feature_hash=feature_hash,
                dataset_hash=dataset_hash,
                metrics=global_metrics,
                alloy_family=alloy_family,
                status="validated",
                db_session=db_session
            )

            # Mark as complete in DB
            if db_session:
                rec = db_session.query(ModelTrainingJob).filter_by(id=job_id).first()
                if rec:
                    rec.status = "complete"
                    rec.progress = 1.0
                    rec.heartbeat = datetime.utcnow()
                    db_session.commit()

            # Clean local fold checkpoints on success
            for fold in range(n_splits):
                p = self.get_checkpoint_path(job_id, fold)
                if os.path.exists(p):
                    os.remove(p)

            logger.info("Training job {} successfully completed.", job_id)
            return global_metrics

        except Exception as e:
            logger.error("Job {} failed with exception: {}", job_id, e)
            if db_session:
                rec = db_session.query(ModelTrainingJob).filter_by(id=job_id).first()
                if rec:
                    rec.status = "failed"
                    rec.error_msg = str(e)
                    rec.heartbeat = datetime.utcnow()
                    db_session.commit()
            raise e


@celery_app.task(name="backend.ml.lifecycle.training_scheduler.train_model_task")
def train_model_task(
    job_id: str,
    csv_data_str: str,
    features: List[str],
    target_columns: List[str],
    group_column: str,
    coverage: str = "rich",
    n_splits: int = 3
) -> Dict[str, Any]:
    """Background Celery task to execute resumable GroupKFold training runs."""
    import pandas as pd
    from io import StringIO
    from backend.db.session import SessionLocal

    df = pd.read_csv(StringIO(csv_data_str))
    db_session = SessionLocal()
    scheduler = TrainingScheduler()
    
    try:
        metrics = scheduler.execute_training_job(
            job_id=job_id,
            df=df,
            features=features,
            target_columns=target_columns,
            group_column=group_column,
            coverage=coverage,
            n_splits=n_splits,
            db_session=db_session
        )
        return metrics
    finally:
        db_session.close()
