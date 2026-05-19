import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.models import Base, PredictionAudit
from backend.ml.reliability import (
    AlloyOODDetector,
    AlloyConformalPredictor,
    AlloyEvidenceFinder,
    AlloyPredictionReliabilitySystem,
    PredictionAuditLogger
)

# In-memory database setup for clean auditing execution tests
@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_ood_detector_edge_cases():
    """Verify Mahalanobis distance calibration, LOF outliers, and severe OOD triggers."""
    detector = AlloyOODDetector()
    
    # Generate mock training dataset (e.g. 50 Fe-C alloy trials, 3 features: Fe, C, Mn)
    np.random.seed(42)
    X_train = np.random.normal(loc=[0.90, 0.05, 0.05], scale=[0.01, 0.005, 0.005], size=(50, 3))
    
    detector.fit(X_train)
    
    # 1. Evaluate normal in-distribution input
    x_normal = np.array([0.90, 0.05, 0.05])
    normal_res = detector.evaluate_ood(x_normal)
    assert not normal_res["is_severe_ood"]
    assert not normal_res["is_mild_ood"]
    
    # 2. Evaluate extreme OOD outlier
    x_outlier = np.array([0.10, 0.80, 0.10])
    outlier_res = detector.evaluate_ood(x_outlier)
    assert outlier_res["is_severe_ood"]
    assert outlier_res["mahalanobis_distance"] > detector.mahalanobis_threshold_severe


def test_conformal_uncertainty_coverage():
    """Assert mathematical split conformal prediction interval bounds and coverage rates."""
    predictor = AlloyConformalPredictor()
    
    # Setup mock validation residuals
    np.random.seed(123)
    y_true_calib = np.random.normal(loc=400.0, scale=30.0, size=100)
    # Model predictions have mild normal errors
    y_pred_calib = y_true_calib + np.random.normal(loc=0.0, scale=15.0, size=100)
    
    predictor.fit(y_true_calib, y_pred_calib)
    
    # Verify precalculated quantiles exist
    assert 0.10 in predictor.calibrated_quantiles
    assert 0.05 in predictor.calibrated_quantiles
    
    # Calculate interval for a new prediction
    y_pred = 410.0
    lower_b, upper_b, width = predictor.predict_interval(y_pred, alpha=0.10)
    assert lower_b < y_pred < upper_b
    assert width > 0.0
    
    # Evaluate empirical coverage on test set
    y_true_test = np.random.normal(loc=400.0, scale=30.0, size=100)
    y_pred_test = y_true_test + np.random.normal(loc=0.0, scale=15.0, size=100)
    
    coverage = predictor.validate_coverage(y_true_test, y_pred_test, alpha=0.10)
    # Inductive conformal bounds guarantees empirical coverage around ~90% (e.g. >= 80% under standard noise)
    assert coverage >= 0.80


def test_nearest_neighbor_evidence():
    """Ensure top-5 closest empirical matches return with matching metadata DOIs and processing routes."""
    finder = AlloyEvidenceFinder()
    
    np.random.seed(7)
    X_train = np.random.normal(loc=0.5, scale=0.1, size=(20, 2))
    
    metadata = []
    for i in range(20):
        metadata.append({
            "composition": {"Fe": 0.5 + 0.01*i, "C": 0.5 - 0.01*i},
            "paper_doi": f"10.1016/j.alloy.2026.0{i}",
            "heat_treatment_category": "anneal",
            "cooling_method": "furnace",
            "manufacturing_route": "wrought",
            "annealing_temperature": 850.0
        })
        
    finder.fit(X_train, metadata)
    
    # Search for nearest neighbors of normal composition representation
    x_query = np.array([0.5, 0.5])
    neighbors = finder.find_evidence(x_query)
    
    assert len(neighbors) == 5
    # Verify ranking sequences
    assert neighbors[0]["rank"] == 1
    assert neighbors[4]["rank"] == 5
    assert neighbors[0]["distance"] <= neighbors[4]["distance"]
    # Check populated fields
    assert "10.1016/j.alloy.2026.0" in neighbors[0]["paper_doi"]
    assert "WROUGHT -> ANNEAL @ 850.0C -> FURNACE" in neighbors[0]["processing_route"]


def test_prediction_reliability_system_refusal(db_session):
    """Confirm automatic refusals for severe OOD, large intervals, or distant neighbors."""
    # 1. Setup sub-elements
    np.random.seed(99)
    X_train = np.random.normal(loc=[0.8, 0.2], scale=0.01, size=(50, 2))
    
    ood_detector = AlloyOODDetector().fit(X_train)
    
    y_true_cal = np.random.normal(loc=500.0, scale=20.0, size=50)
    y_pred_cal = y_true_cal + np.random.normal(loc=0.0, scale=10.0, size=50)
    conformal_pred = AlloyConformalPredictor().fit(y_true_cal, y_pred_cal)
    
    metadata = [{"composition": {"Fe": 0.8, "C": 0.2}, "paper_doi": "10.1000/test"} for _ in range(50)]
    evidence_finder = AlloyEvidenceFinder().fit(X_train, metadata)
    
    # Initialize high-integrity reliability system with strict limits
    reliability_sys = AlloyPredictionReliabilitySystem(
        ood_detector=ood_detector,
        conformal_predictor=conformal_pred,
        evidence_finder=evidence_finder,
        max_allowed_conformal_width=50.0, # extremely strict width constraint
        max_allowed_neighbor_distance=0.25
    )

    # CASE A: Normal, low-risk query
    x_normal = np.array([0.8, 0.2])
    audit_normal = reliability_sys.audit_prediction(
        job_id="job_normal_001",
        features={"alloy_family": "steel", "heat_treatment_category": "none"},
        feature_array=x_normal,
        prediction_val=505.0,
        db_session=db_session
    )
    assert audit_normal["risk_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert audit_normal["refusal_reason"] is None
    
    # Verify normal entry was successfully saved in DB
    db_rec_normal = db_session.query(PredictionAudit).filter_by(job_id="job_normal_001").first()
    assert db_rec_normal is not None
    assert db_rec_normal.risk_tier == audit_normal["risk_tier"]

    # CASE B: Refusal due to Severe OOD
    x_severe_ood = np.array([0.2, 0.8]) # highly skewed
    audit_ood = reliability_sys.audit_prediction(
        job_id="job_ood_002",
        features={"alloy_family": "steel"},
        feature_array=x_severe_ood,
        prediction_val=505.0,
        db_session=db_session
    )
    assert audit_ood["risk_tier"] == "REFUSE"
    assert "out-of-distribution" in audit_ood["refusal_reason"]

    # CASE C: Refusal due to excessive conformal width
    # If we request conformal check with tight constraints, width of calibration (~30+)
    # let's trigger it by reducing max allowed width to extremely small values (e.g. 5.0)
    reliability_sys.max_allowed_conformal_width = 5.0
    audit_width = reliability_sys.audit_prediction(
        job_id="job_width_003",
        features={"alloy_family": "steel"},
        feature_array=x_normal,
        prediction_val=505.0,
        db_session=db_session
    )
    assert audit_width["risk_tier"] == "REFUSE"
    assert "conformal prediction interval width" in audit_width["refusal_reason"].lower()


def test_prediction_audit_logger(db_session):
    """Verify that PredictionAuditLogger correctly saves, commits, and queries audit logs."""
    # Write a test log entry
    log_rec = PredictionAuditLogger.log_audit(
        db_session=db_session,
        job_id="job_logger_101",
        risk_tier="HIGH",
        uncertainty_width=88.5,
        ood_score=2.15,
        refusal_reason="Sample OOD warning"
    )
    
    assert log_rec.id is not None
    assert log_rec.job_id == "job_logger_101"
    assert log_rec.risk_tier == "HIGH"
    
    # Query back the record
    queried = PredictionAuditLogger.get_audit(db_session, "job_logger_101")
    assert queried is not None
    assert queried.risk_tier == "HIGH"
    assert queried.uncertainty_width == 88.5
    assert queried.ood_score == 2.15
    assert queried.refusal_reason == "Sample OOD warning"

