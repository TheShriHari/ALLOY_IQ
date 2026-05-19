import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple
from sqlalchemy.orm import Session
from backend.db.models import BlindValidationTrial
from loguru import logger

class BlindTrialRegistry:
    """
    Manages immutable trial registration and cryptographic locking.
    Generates SHA-256 locks of experimental compositions and predictions prior to physical lab synthesis.
    """
    def __init__(self, db: Session):
        self.db = db

    def generate_verification_hash(
        self,
        composition: Dict[str, float],
        processing: Dict[str, Any],
        predictions: Dict[str, float],
        conformal_bounds: Dict[str, Tuple[float, float]]
    ) -> str:
        """Computes a secure, deterministic SHA-256 hash of the complete prediction snapshot."""
        # Use sorted key mapping to ensure deterministic serialization
        payload = {
            "composition": sorted(composition.items()),
            "processing": sorted((k, str(v)) for k, v in processing.items()),
            "predictions": sorted(predictions.items()),
            "conformal_bounds": sorted((k, list(v)) for k, v in conformal_bounds.items())
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def register_blind_trial(
        self,
        composition: Dict[str, float],
        processing: Dict[str, Any],
        predictions: Dict[str, float],
        conformal_bounds: Dict[str, Tuple[float, float]]
    ) -> BlindValidationTrial:
        """
        Creates an immutable, cryptographically locked laboratory validation trial.
        Enforces validation locking before synthesis, preventing design tampering.
        """
        # Calculate secure signature
        locked_hash = self.generate_verification_hash(composition, processing, predictions, conformal_bounds)
        
        trial = BlindValidationTrial(
            alloy_composition=composition,
            processing_route=processing,
            predicted_properties=predictions,
            prediction_interval=conformal_bounds,
            locked_hash=locked_hash,
            lab_status="locked",
            created_at=datetime.utcnow()
        )
        
        self.db.add(trial)
        self.db.commit()
        self.db.refresh(trial)
        
        logger.info("Registered blind experimental trial. ID: {}, Hash Lock: {}", trial.experiment_id, locked_hash)
        return trial

    def verify_trial_integrity(self, experiment_id: str) -> bool:
        """Audits trial database record against locked_hash to detect unauthorized modification."""
        trial = self.db.query(BlindValidationTrial).filter(BlindValidationTrial.experiment_id == experiment_id).first()
        if not trial:
            logger.warning("Integrity check bypassed: Trial {} not found.", experiment_id)
            return False
            
        current_hash = self.generate_verification_hash(
            composition=trial.alloy_composition,
            processing=trial.processing_route,
            predictions=trial.predicted_properties,
            conformal_bounds=trial.prediction_interval
        )
        
        if current_hash != trial.locked_hash:
            logger.error("Tampering Alert! Registry record modified. Expected hash {}, got {}", trial.locked_hash, current_hash)
            return False
            
        logger.info("Registry integrity verified for trial: {}", experiment_id)
        return True
