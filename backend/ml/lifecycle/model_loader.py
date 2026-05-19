import os
import json
import joblib
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from loguru import logger

from backend.ml.lifecycle.model_registry import ModelRegistry
from backend.ml.lifecycle.ensemble_manager import EnsembleManager

class ModelLoader:
    """
    Manages high-integrity inference model loading.
    Validates feature signature hashes and implements recursive rollback fallbacks on deserialization failure.
    """
    def __init__(self, registry_dir: str = "./data/models"):
        self.registry = ModelRegistry(registry_dir=registry_dir)

    @staticmethod
    def compute_feature_hash(features: List[str]) -> str:
        """Helper to compute feature hash consistently."""
        feature_str = ",".join(sorted(str(f).strip().lower() for f in features))
        return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()

    def load_active_model(
        self,
        alloy_family: str,
        current_features: List[str],
        db_session: Optional[Session] = None
    ) -> EnsembleManager:
        """
        Loads the latest validated/active model for the family.
        Verifies feature signature hashes. If loading fails, rolls back to previous valid models.
        """
        # Find all validated/active candidate models from history
        models_history = []
        if db_session:
            from backend.db.models import ModelRegistryEntry
            records = (
                db_session.query(ModelRegistryEntry)
                .filter(ModelRegistryEntry.status.in_(["validated", "active"]))
                .order_by(ModelRegistryEntry.created_at.desc())
                .all()
            )
            for r in records:
                info = self.registry.get_model_info(r.model_id)
                if info and info.get("alloy_family") == alloy_family:
                    models_history.append(info)
        
        # Merge with local registry ledger history if empty
        if not models_history and os.path.exists(self.registry.registry_json):
            try:
                with open(self.registry.registry_json, "r", encoding="utf-8") as f:
                    history = json.loads(f.read())
                models_history = [
                    h for h in history 
                    if h.get("alloy_family") == alloy_family and h.get("status") in ["validated", "active"]
                ]
                models_history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            except Exception:
                pass

        if not models_history:
            raise ValueError(f"No validated or active models found for alloy family '{alloy_family}'.")

        # Compute current input feature signature hash
        current_hash = self.compute_feature_hash(current_features)

        # Loop through history (rollback chain fallback)
        for idx, model_info in enumerate(models_history):
            model_id = model_info["model_id"]
            expected_hash = model_info["feature_hash"]

            # 1. Feature signature hash validation check
            if current_hash != expected_hash:
                logger.error(
                    "Feature hash mismatch for model {}. Expected: {}, Got: {}. Rejecting incompatible inputs.",
                    model_id, expected_hash, current_hash
                )
                if idx == len(models_history) - 1:
                    raise ValueError(
                        f"Feature signature mismatch on all available models for family '{alloy_family}'."
                    )
                # Proceed to try the next model in the rollback chain
                continue

            # 2. Deserialization attempt
            try:
                base_path = os.path.join(self.registry.registry_dir, model_id)
                logger.info("Attempting to load model {}...", model_id)
                
                manager = EnsembleManager()
                manager.load(base_path)
                
                logger.info("Successfully loaded active model: {}", model_id)
                return manager

            except Exception as e:
                logger.error(
                    "Failed to deserialize model {}: {}. Attempting rollback recovery.",
                    model_id, e
                )
                if idx == len(models_history) - 1:
                    raise RuntimeError(
                        f"All validated models for family '{alloy_family}' failed to load."
                    )

        raise ValueError(f"Could not load any compatible validated models for family '{alloy_family}'.")
