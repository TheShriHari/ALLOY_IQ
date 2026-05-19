import pytest
import os
from datetime import datetime

from backend.ml.experimental.experiment_manager import ExperimentalValidationManager
from backend.ml.experimental.blind_trial_registry import BlindTrialRegistry
from backend.ml.experimental.synthesis_tracker import SynthesisTracker
from backend.ml.experimental.lab_result_ingestion import LabResultIngestor
from backend.ml.experimental.validation_report import ValidationReportGenerator
from backend.db.models import BlindValidationTrial

@pytest.fixture
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.db.models import Base
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

def test_prediction_locking_and_registration(db_session):
    """Ensure experimental trial registers predictions and locks them with SHA-256."""
    manager = ExperimentalValidationManager(db_session)
    
    comp = {"Fe": 60.0, "Ni": 20.0, "Cr": 20.0}
    proc = {"heat_treatment_category": "annealed", "annealing_temperature": 1050.0}
    preds = {"yield_strength": 350.0, "tensile_strength": 650.0}
    bounds = {"yield_strength": (320.0, 380.0), "tensile_strength": (600.0, 700.0)}
    
    exp_id = manager.register_prediction_trial(comp, proc, preds, bounds)
    assert exp_id is not None
    
    trial = db_session.query(BlindValidationTrial).filter(BlindValidationTrial.experiment_id == exp_id).first()
    assert trial is not None
    assert trial.lab_status == "locked"
    assert trial.locked_hash is not None
    
    # Confirm integrity checks pass
    assert manager.registry.verify_trial_integrity(exp_id) is True


def test_tamper_detection(db_session):
    """Assert that modifying registered database prediction metrics triggers integrity block."""
    manager = ExperimentalValidationManager(db_session)
    
    comp = {"Fe": 70.0, "Cr": 30.0}
    proc = {"heat_treatment_category": "as-cast"}
    preds = {"hardness": 220.0}
    bounds = {"hardness": (200.0, 240.0)}
    
    exp_id = manager.register_prediction_trial(comp, proc, preds, bounds)
    
    # Modify records in DB directly (simulation of malicious tampering)
    trial = db_session.query(BlindValidationTrial).filter(BlindValidationTrial.experiment_id == exp_id).first()
    trial.predicted_properties = {"hardness": 150.0}  # Tampered value!
    db_session.commit()
    
    # Assert verification blocks and flags tampering
    assert manager.registry.verify_trial_integrity(exp_id) is False
    
    # Assert manager raises block exception on operations
    with pytest.raises(ValueError) as exc_info:
        manager.lock_synthesis_start(exp_id, "Operator A", "spec_1")
    assert "Integrity check failed" in str(exc_info.value)


def test_lab_ingestion_validations(db_session):
    """Verify that physical validation properties parse correctly and reject impossible bounds."""
    manager = ExperimentalValidationManager(db_session)
    
    comp = {"Al": 90.0, "Cu": 10.0}
    proc = {"heat_treatment_category": "aged"}
    preds = {"yield_strength": 120.0, "tensile_strength": 180.0}
    bounds = {"yield_strength": (100.0, 140.0), "tensile_strength": (160.0, 200.0)}
    
    exp_id = manager.register_prediction_trial(comp, proc, preds, bounds)
    manager.lock_synthesis_start(exp_id, "Operator B", "spec_2", "Aged at 150C for 12 hours.")
    
    # 1. Reject invalid yield (too large, above uts check)
    invalid_measurements = {"yield_strength": 250.0, "tensile_strength": 180.0}
    with pytest.raises(ValueError) as exc_info:
        manager.ingest_measurements(exp_id, invalid_measurements)
    assert "Physical Conflict: Ingested Yield Strength" in str(exc_info.value)

    # 2. Ingest clean measurements
    valid_measurements = {
        "yield_strength": 115.0,
        "tensile_strength": 178.0,
        "elongation": 18.0,
        "hardness": 95.0,
        "density": 2.7
    }
    success = manager.ingest_measurements(exp_id, valid_measurements)
    assert success is True
    
    trial = db_session.query(BlindValidationTrial).filter(BlindValidationTrial.experiment_id == exp_id).first()
    assert trial.lab_status == "completed"
    assert trial.measured_properties["elongation"] == 18.0


def test_three_novel_alloy_trials_validation_report(db_session):
    """Verify registry support for at least 3 novel alloy compositions and correct metrics validation report."""
    manager = ExperimentalValidationManager(db_session)
    
    # Register 3 distinct novel compositions
    novel_alloys = [
        (
            {"Ti": 50.0, "Al": 45.0, "Nb": 5.0},
            {"yield_strength": 480.0},
            {"yield_strength": (450.0, 510.0)},
            {"yield_strength": 490.0, "tensile_strength": 580.0} # measured properties
        ),
        (
            {"Fe": 50.0, "Mn": 30.0, "Al": 10.0, "C": 10.0},
            {"yield_strength": 850.0},
            {"yield_strength": (800.0, 900.0)},
            {"yield_strength": 870.0, "tensile_strength": 980.0} # measured properties
        ),
        (
            {"Co": 30.0, "Cr": 30.0, "Fe": 20.0, "Ni": 20.0},
            {"yield_strength": 600.0},
            {"yield_strength": (580.0, 620.0)},
            {"yield_strength": 650.0, "tensile_strength": 780.0} # measured properties (outside interval)
        )
    ]
    
    exp_ids = []
    for comp, preds, bounds, measured in novel_alloys:
        eid = manager.register_prediction_trial(comp, {"heat_treatment_category": "as-rolled"}, preds, bounds)
        exp_ids.append((eid, measured))
        
    # Start synthesis and ingest results
    for index, (eid, measured) in enumerate(exp_ids):
        manager.lock_synthesis_start(eid, "Lab Admin", f"alloy_novel_{index}")
        manager.ingest_measurements(eid, measured)
        
    # Generate overall validation report
    report = manager.verify_and_generate_report()
    
    assert report["status"] == "ready"
    assert report["completed_count"] == 3
    
    # 2 trials (TiAlNb & FeMnAlC) are within bounds. 1 (CoCrFeNi) is outside bounds (650 > 620).
    # Conformal coverage should be 2/3 = 66.6%
    analytics = report["property_analytics"]["yield_strength"]
    assert analytics["sample_count"] == 3
    assert abs(analytics["conformal_coverage_rate"] - 0.666) < 0.01
    
    # Confirm mean absolute error calculation:
    # absolute errors: abs(480-490) = 10, abs(850-870) = 20, abs(600-650) = 50.
    # MAE = (10 + 20 + 50) / 3 = 26.66 MPa
    assert abs(analytics["mean_absolute_error"] - 26.66) < 0.01
