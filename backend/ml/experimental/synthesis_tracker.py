from datetime import datetime
from sqlalchemy.orm import Session
from backend.db.models import BlindValidationTrial
from loguru import logger

class SynthesisTracker:
    """
    Manages physical synthesis steps for registered trials.
    Transitions trials from 'locked' to 'synthesizing' and logs metallurgical logs.
    """
    def __init__(self, db: Session):
        self.db = db

    def start_synthesis(self, experiment_id: str, operator: str, specimen_id: str, notes: str = "") -> bool:
        """Flags the trial as currently in synthesis in the physical lab."""
        trial = self.db.query(BlindValidationTrial).filter(BlindValidationTrial.experiment_id == experiment_id).first()
        if not trial:
            logger.error("Synthesis tracking failure: Trial {} not found.", experiment_id)
            return False
            
        if trial.lab_status != "locked":
            logger.warning("Synthesis already initiated or completed for trial: {}", experiment_id)
            return False
            
        trial.lab_status = "synthesizing"
        trial.operator = operator
        trial.specimen_id = specimen_id
        trial.synthesis_date = datetime.utcnow()
        trial.process_notes = notes
        
        self.db.commit()
        logger.info("Synthesis started by operator '{}' for specimen: {}", operator, specimen_id)
        return True

    def log_process_update(self, experiment_id: str, note_append: str) -> bool:
        """Appends technical observations or thermal log records to process_notes."""
        trial = self.db.query(BlindValidationTrial).filter(BlindValidationTrial.experiment_id == experiment_id).first()
        if not trial:
            return False
            
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        formatted_append = f"\n[{timestamp}] {note_append}"
        
        if trial.process_notes:
            trial.process_notes += formatted_append
        else:
            trial.process_notes = formatted_append
            
        self.db.commit()
        return True
