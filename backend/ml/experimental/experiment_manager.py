from typing import Dict, Any, Tuple
from sqlalchemy.orm import Session
from backend.ml.experimental.blind_trial_registry import BlindTrialRegistry
from backend.ml.experimental.synthesis_tracker import SynthesisTracker
from backend.ml.experimental.lab_result_ingestion import LabResultIngestor
from backend.ml.experimental.validation_report import ValidationReportGenerator

class ExperimentalValidationManager:
    """
    Unified manager orchestrating the Laboratory Blind Validation pipeline.
    Ensures safe locking of predictions before synthesis, tracks physical operations,
    ingests material properties, and outputs verified reports.
    """
    def __init__(self, db: Session):
        self.db = db
        self.registry = BlindTrialRegistry(db)
        self.tracker = SynthesisTracker(db)
        self.ingestor = LabResultIngestor(db)
        self.reporter = ValidationReportGenerator(db)

    def register_prediction_trial(
        self,
        composition: Dict[str, float],
        processing: Dict[str, Any],
        predictions: Dict[str, float],
        conformal_bounds: Dict[str, Tuple[float, float]]
    ) -> str:
        """Saves predictions and creates the initial validation lock."""
        trial = self.registry.register_blind_trial(
            composition=composition,
            processing=processing,
            predictions=predictions,
            conformal_bounds=conformal_bounds
        )
        return trial.experiment_id

    def lock_synthesis_start(self, experiment_id: str, operator: str, specimen_id: str, notes: str = "") -> bool:
        """Transitions trial status to 'synthesizing'."""
        # 1. Audit integrity first to prevent tampering
        if not self.registry.verify_trial_integrity(experiment_id):
            raise ValueError(f"Integrity check failed: prediction record has been modified for trial {experiment_id}!")
            
        return self.tracker.start_synthesis(experiment_id, operator, specimen_id, notes)

    def ingest_measurements(self, experiment_id: str, raw_measurements: Dict[str, float]) -> bool:
        """Ingests mechanical measurements and updates status to 'completed'."""
        # 1. Audit integrity first to prevent tampering
        if not self.registry.verify_trial_integrity(experiment_id):
            raise ValueError(f"Integrity check failed: prediction record has been modified for trial {experiment_id}!")
            
        return self.ingestor.ingest_experimental_results(experiment_id, raw_measurements)

    def verify_and_generate_report(self) -> Dict[str, Any]:
        """Audits overall database trials integrity and returns aggregated validations metrics."""
        # Check integrity on all trials first
        from backend.db.models import BlindValidationTrial
        trials = self.db.query(BlindValidationTrial).all()
        for t in trials:
            if not self.registry.verify_trial_integrity(t.experiment_id):
                raise ValueError(f"Security Tampering Block: Record integrity violation for trial {t.experiment_id}!")
                
        return self.reporter.generate_overall_validation_report()
