import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from loguru import logger

from backend.db.models import ModelRegistryEntry

class ModelRegistry:
    """
    Manages registering production models, tracking their verification metrics,
    features compatibility hashes, and status states (candidate -> validated -> active).
    """
    def __init__(self, registry_dir: str = "./data/models"):
        self.registry_dir = os.path.abspath(registry_dir)
        self.registry_json = os.path.join(self.registry_dir, "registry.json")
        os.makedirs(self.registry_dir, exist_ok=True)

    def register_model(
        self,
        model_id: str,
        pipeline_bytes: bytes,
        conformal_bytes: bytes,
        feature_hash: str,
        dataset_hash: str,
        metrics: Dict[str, Any],
        alloy_family: str,
        status: str = "candidate",
        db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Saves the binary pipeline + conformal weights to local model storage,
        writes metadata to the local JSON ledger, and inserts into DB registry.
        """
        # Save model binaries
        model_path = os.path.join(self.registry_dir, f"{model_id}")
        
        # Calculate combined model hash
        hasher = hashlib.sha256()
        hasher.update(pipeline_bytes)
        hasher.update(conformal_bytes)
        model_hash = hasher.hexdigest()

        # Write binaries
        with open(f"{model_path}_stack.pkl", "wb") as f:
            f.write(pipeline_bytes)
        with open(f"{model_path}_conformal.pkl", "wb") as f:
            f.write(conformal_bytes)

        # Build entry
        entry = {
            "model_id": model_id,
            "model_hash": model_hash,
            "feature_hash": feature_hash,
            "dataset_hash": dataset_hash,
            "metrics": metrics,
            "alloy_family": alloy_family,
            "created_at": datetime.utcnow().isoformat(),
            "status": status
        }

        # Save to local JSON ledger
        history = []
        if os.path.exists(self.registry_json):
            try:
                with open(self.registry_json, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        history = json.loads(content)
            except Exception as e:
                logger.error("Failed to read models registry JSON: {}", e)

        # Overwrite if exists, otherwise append
        history = [h for h in history if h["model_id"] != model_id]
        history.append(entry)

        with open(self.registry_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        logger.info("Registered model {} locally with status '{}'", model_id, status)

        # Persist to database ModelRegistryEntry
        if db_session:
            db_entry = ModelRegistryEntry(
                model_id=model_id,
                model_hash=model_hash,
                feature_hash=feature_hash,
                dataset_hash=dataset_hash,
                metrics=metrics,
                status=status
            )
            # Merge to support updates/saves cleanly
            db_session.merge(db_entry)
            db_session.commit()
            logger.info("Persisted ModelRegistryEntry to DB.")

        return entry

    def update_model_status(
        self,
        model_id: str,
        status: str,
        db_session: Optional[Session] = None
    ):
        """Updates status locally and in the SQL database."""
        # Local JSON ledger update
        if os.path.exists(self.registry_json):
            try:
                with open(self.registry_json, "r", encoding="utf-8") as f:
                    history = json.loads(f.read())
                
                updated = False
                for h in history:
                    if h["model_id"] == model_id:
                        h["status"] = status
                        updated = True
                
                if updated:
                    with open(self.registry_json, "w", encoding="utf-8") as f:
                        json.dump(history, f, indent=4)
                    logger.info("Updated model status locally for {} to '{}'", model_id, status)
            except Exception as e:
                logger.error("Failed to update status in JSON ledger: {}", e)

        # SQL update
        if db_session:
            db_entry = db_session.query(ModelRegistryEntry).filter_by(model_id=model_id).first()
            if db_entry:
                db_entry.status = status
                db_session.commit()
                logger.info("Updated model status in DB for {} to '{}'", model_id, status)

    def get_latest_validated_model(
        self,
        alloy_family: str,
        db_session: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """Queries the latest validated or active model."""
        if db_session:
            # Query from DB
            db_records = (
                db_session.query(ModelRegistryEntry)
                .filter(ModelRegistryEntry.status.in_(["validated", "active"]))
                .order_by(ModelRegistryEntry.created_at.desc())
                .all()
            )
            # Find the first one matching local files of that alloy family
            for r in db_records:
                local_info = self.get_model_info(r.model_id)
                if local_info and local_info.get("alloy_family") == alloy_family:
                    return {
                        "model_id": r.model_id,
                        "model_hash": r.model_hash,
                        "feature_hash": r.feature_hash,
                        "dataset_hash": r.dataset_hash,
                        "metrics": r.metrics,
                        "status": r.status,
                        "alloy_family": alloy_family
                    }
        
        # Fallback to local JSON ledger
        if os.path.exists(self.registry_json):
            try:
                with open(self.registry_json, "r", encoding="utf-8") as f:
                    history = json.loads(f.read())
                
                # Filter by status and family, sorting by created_at desc
                matches = [
                    h for h in history 
                    if h.get("alloy_family") == alloy_family and h.get("status") in ["validated", "active"]
                ]
                if matches:
                    matches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                    return matches[0]
            except Exception as e:
                logger.error("Error reading JSON registry ledger: {}", e)

        return None

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Reads local info dict for a model."""
        if os.path.exists(self.registry_json):
            try:
                with open(self.registry_json, "r", encoding="utf-8") as f:
                    history = json.loads(f.read())
                for h in history:
                    if h["model_id"] == model_id:
                        return h
            except Exception:
                pass
        return None
