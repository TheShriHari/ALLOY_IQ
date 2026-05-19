from sqlalchemy.orm import Session
from backend.db.models import PredictionAudit
from loguru import logger
from typing import Optional

class PredictionAuditLogger:
    """
    Service layer to persist and query prediction audit footprints
    in the database.
    """
    @staticmethod
    def log_audit(
        db_session: Session,
        job_id: str,
        risk_tier: str,
        uncertainty_width: float,
        ood_score: float,
        refusal_reason: Optional[str] = None
    ) -> BaseException | PredictionAudit:
        """
        Creates and commits a new PredictionAudit record in the database.
        """
        logger.info(
            "Persisting prediction audit for Job: {} (Tier: {}, OOD: {:.2f}, Width: {:.2f})",
            job_id, risk_tier, ood_score, uncertainty_width
        )
        
        audit_record = PredictionAudit(
            job_id=job_id,
            risk_tier=risk_tier,
            uncertainty_width=uncertainty_width,
            ood_score=ood_score,
            refusal_reason=refusal_reason
        )
        
        try:
            db_session.add(audit_record)
            db_session.commit()
            db_session.refresh(audit_record)
            return audit_record
        except Exception as e:
            db_session.rollback()
            logger.error("Failed to persist prediction audit record: {}", e)
            raise e

    @staticmethod
    def get_audit(db_session: Session, job_id: str) -> Optional[PredictionAudit]:
        """
        Retrieves the auditing history record for a specific job_id.
        """
        return db_session.query(PredictionAudit).filter_by(job_id=job_id).first()
