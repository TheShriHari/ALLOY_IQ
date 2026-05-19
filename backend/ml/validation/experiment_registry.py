import os
import json
import hashlib
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from backend.db.models import ExperimentRun
from loguru import logger

class ExperimentRegistry:
    """
    Cryptographic Experiment Registry.
    Tracks SHA-256 hashes of datasets, feature configurations, and model binaries
    alongside performance metrics to enforce statistical transparency and prevent cherry-picking.
    """
    def __init__(self, registry_path: str = "./data/experiments/registry.json"):
        # Put under workspace path
        self.registry_path = os.path.abspath(registry_path)
        self._ensure_directory()

    def _ensure_directory(self):
        """Creates the parent registry folders if they don't exist."""
        parent_dir = os.path.dirname(self.registry_path)
        os.makedirs(parent_dir, exist_ok=True)

    @staticmethod
    def compute_dataset_hash(df: pd.DataFrame) -> str:
        """Computes deterministic SHA-256 hash over the values of a pandas DataFrame."""
        # Convert DataFrame to a canonical CSV byte sequence for consistent hashing
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(csv_bytes).hexdigest()

    @staticmethod
    def compute_feature_hash(features: List[str]) -> str:
        """Computes deterministic SHA-256 hash over sorted feature column lists."""
        feature_str = ",".join(sorted(str(f).strip().lower() for f in features))
        return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_model_hash(model_bytes: bytes) -> str:
        """Computes SHA-256 hash over serialized model weights/byte arrays."""
        return hashlib.sha256(model_bytes).hexdigest()

    def log_run(
        self,
        train_df: pd.DataFrame,
        features: List[str],
        model_bytes: bytes,
        metrics: Dict[str, Any],
        training_config: Dict[str, Any] = None,
        db_session: Session = None
    ) -> Dict[str, Any]:
        """
        Records an experiment execution into the cryptographic ledger file and optionally the database.
        Returns the constructed run payload.
        """
        logger.info("Recording experiment run into cryptographic registry at {}", self.registry_path)
        
        # Calculate hashes
        dataset_hash = self.compute_dataset_hash(train_df)
        feature_hash = self.compute_feature_hash(features)
        model_hash = self.compute_model_hash(model_bytes)
        
        config_payload = training_config or {}
        
        run_entry = {
            "run_id": hashlib.sha256(f"{dataset_hash}-{feature_hash}-{model_hash}".encode("utf-8")).hexdigest()[:16],
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_hash": dataset_hash,
            "feature_hash": feature_hash,
            "model_hash": model_hash,
            "training_config": config_payload,
            "metrics": metrics
        }
        
        # Read existing history
        history = []
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        history = json.loads(content)
            except Exception as e:
                logger.error("Error reading experiment registry: {}", e)
                
        # Append and save
        history.append(run_entry)
        
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
            logger.info("Successfully recorded experiment Run ID: {}", run_entry["run_id"])
        except Exception as e:
            logger.error("Failed to write to experiment registry: {}", e)
            raise e
            
        # Optional SQLAlchemy database persistence
        if db_session:
            db_run = ExperimentRun(
                dataset_hash=dataset_hash,
                feature_hash=feature_hash,
                model_hash=model_hash,
                metrics_path=self.registry_path
            )
            db_session.add(db_run)
            db_session.commit()
            logger.info("Persisted ExperimentRun record to DB.")
            
        return run_entry

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Retrieves list of all logged runs in the ledger."""
        if not os.path.exists(self.registry_path):
            return []
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except Exception as e:
            logger.error("Error reading registry database: {}", e)
        return []

