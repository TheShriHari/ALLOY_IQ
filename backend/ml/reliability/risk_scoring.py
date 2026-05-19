from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.ml.reliability.ood_detector import AlloyOODDetector
from backend.ml.reliability.uncertainty import AlloyConformalPredictor
from backend.ml.reliability.nearest_neighbors import AlloyEvidenceFinder
from backend.ml.reliability.prediction_audit import PredictionAuditLogger
from loguru import logger

class AlloyPredictionReliabilitySystem:
    """
    ML Reliability Decision-Support Gateway.
    Combines OOD detectors, conformal predictors, and nearest-neighbor finders
    to assign risk scores (LOW, MEDIUM, HIGH, REFUSE) and record SQL audits.
    """
    def __init__(
        self,
        ood_detector: AlloyOODDetector,
        conformal_predictor: AlloyConformalPredictor,
        evidence_finder: AlloyEvidenceFinder,
        max_allowed_conformal_width: float = 200.0,
        max_allowed_neighbor_distance: float = 0.35
    ):
        self.ood_detector = ood_detector
        self.conformal_predictor = conformal_predictor
        self.evidence_finder = evidence_finder
        self.max_allowed_conformal_width = max_allowed_conformal_width
        self.max_allowed_neighbor_distance = max_allowed_neighbor_distance

    def audit_prediction(
        self,
        job_id: str,
        features: dict,
        feature_array: Any,
        prediction_val: float,
        db_session: Session = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive reliability auditing on a prediction request.
        Outputs risk tiers, flags, evidence matches, conformal intervals, and logs to SQLite/Postgres.
        """
        logger.info("Executing prediction reliability audit for Job ID: {}", job_id)

        # 1. Run OOD Evaluation
        ood_res = self.ood_detector.evaluate_ood(feature_array)
        m_dist = ood_res["mahalanobis_distance"]

        # 2. Run Conformal Prediction Interval calculation (90% Confidence)
        lower_b, upper_b, width = self.conformal_predictor.predict_interval(prediction_val, alpha=0.10)

        # 3. Retrieve Nearest-Neighbor Evidence
        neighbors = self.evidence_finder.find_evidence(feature_array)
        closest_neighbor_dist = neighbors[0]["distance"] if neighbors else 1.0

        # 4. Evaluate Risk Flags
        flags = []
        if ood_res["is_mild_ood"] or ood_res["is_severe_ood"] or ood_res["is_lof_outlier"]:
            flags.append("OOD")
            
        family = str(features.get("alloy_family") or "").strip().lower()
        # Trigger SPARSE_FAMILY if we have an unknown/rare family representation
        if family not in ("steel", "hea", "aluminum"):
            flags.append("SPARSE_FAMILY")
            
        # Trigger MISSING_PROCESSING if annealing temp is zero when a treatment was explicitly category-listed
        ht_cat = str(features.get("heat_treatment_category") or "").strip().lower()
        temp = features.get("annealing_temperature")
        if ht_cat not in ("none", "as_cast", "") and (temp is None or temp <= 0):
            flags.append("MISSING_PROCESSING")
            
        if closest_neighbor_dist > 0.20:
            flags.append("DISTANT_NEIGHBORS")
            
        if m_dist > ood_res["mahalanobis_threshold_mild"] or closest_neighbor_dist > 0.15:
            flags.append("LOW_CONFIDENCE")

        # 5. Enforce Prediction Refusal Logic
        refusal_reason = None
        risk_tier = "LOW"

        if ood_res["is_severe_ood"]:
            risk_tier = "REFUSE"
            refusal_reason = f"Severe out-of-distribution feature detected (Mahalanobis distance: {m_dist:.2f})"
        elif closest_neighbor_dist > self.max_allowed_neighbor_distance:
            risk_tier = "REFUSE"
            refusal_reason = f"No close empirical alloy matches in training database (Closest distance: {closest_neighbor_dist:.3f})"
        elif width > self.max_allowed_conformal_width:
            risk_tier = "REFUSE"
            refusal_reason = f"Conformal prediction interval width exceeds reliability limits (Width: {width:.1f})"
        else:
            # Assign remaining risk tiers based on flag severity
            if "OOD" in flags or "LOW_CONFIDENCE" in flags:
                risk_tier = "HIGH"
            elif flags:
                risk_tier = "MEDIUM"

        audit_payload = {
            "job_id": job_id,
            "risk_tier": risk_tier,
            "risk_flags": flags,
            "uncertainty_width": width,
            "prediction_interval": (lower_b, upper_b),
            "ood_score": m_dist,
            "refusal_reason": refusal_reason,
            "nearest_neighbors": neighbors
        }

        # 6. Optional SQLAlchemy Database Session Logging
        if db_session:
            PredictionAuditLogger.log_audit(
                db_session=db_session,
                job_id=job_id,
                risk_tier=risk_tier,
                uncertainty_width=width,
                ood_score=m_dist,
                refusal_reason=refusal_reason
            )

        return audit_payload
