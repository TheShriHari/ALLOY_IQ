import pytest
import os
import shutil
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.models import Base, ExperimentRun
from backend.ml.validation import (
    BlindValidator,
    DataLeakageAuditor,
    BenchmarkReporter,
    ExperimentRegistry
)

@pytest.fixture
def temp_dir():
    """Temporary directory for experiment registry files during test runs."""
    path = "./temp_test_registry"
    os.makedirs(path, exist_ok=True)
    yield path
    # Cleanup after tests
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def db_session():
    """Clean SQLite in-memory session for model testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()



def test_blind_validator_partitions():
    """Verify that BlindValidator successfully isolates unseen alloy families, DOIs, and routes."""
    # Generate mock dataset
    data = {
        "alloy_family": ["steel", "steel", "hea", "hea", "aluminum"],
        "paper_doi": ["10.1000/a", "10.1000/a", "10.1000/b", "10.1000/c", "10.1000/d"],
        "manufacturing_route": ["wrought", "cast", "wrought", "cast", "wrought"],
        "value": [400, 420, 510, 520, 310]
    }
    df = pd.DataFrame(data)
    
    # 1. Holdout unseen family
    train_fam, test_fam = BlindValidator.holdout_by_family(df, "hea")
    assert len(test_fam) == 2
    assert len(train_fam) == 3
    assert "hea" not in train_fam["alloy_family"].values
    assert "hea" in test_fam["alloy_family"].values

    # 2. Holdout unseen DOIs
    train_doi, test_doi = BlindValidator.holdout_by_doi_groups(df, ["10.1000/a", "10.1000/d"])
    assert len(test_doi) == 3
    assert len(train_doi) == 2
    assert "10.1000/b" in train_doi["paper_doi"].values

    # 3. Holdout unseen manufacturing route
    train_route, test_route = BlindValidator.holdout_by_processing_route(df, "cast")
    assert len(test_route) == 2
    assert len(train_route) == 3
    assert "cast" not in train_route["manufacturing_route"].values


def test_leakage_auditor_detection():
    """Ensure DataLeakageAuditor catches duplicate compositions, DOI overlaps, and group leaks."""
    auditor = DataLeakageAuditor()
    
    # Setup clean splits
    train_data = pd.DataFrame({
        "composition": [{"Fe": 0.9, "C": 0.1}, {"Fe": 0.8, "C": 0.2}],
        "paper_doi": ["10.1000/1", "10.1000/2"],
        "research_group_id": ["group_a", "group_b"],
        "alloy_family": ["steel", "steel"]
    })
    
    test_data_clean = pd.DataFrame({
        "composition": [{"Fe": 0.7, "C": 0.3}],
        "paper_doi": ["10.1000/3"],
        "research_group_id": ["group_c"],
        "alloy_family": ["aluminum"]
    })
    
    # Audit normal split (should have no leakage or overlap risk)
    report_clean = auditor.audit_split(train_data, test_data_clean)
    assert report_clean["has_leakage"] is False
    assert report_clean["train_test_overlap_risk"]["detected"] is False
    
    # Introduce composition overlap leakage
    test_data_leaked_comp = pd.DataFrame({
        "composition": [{"Fe": 0.9, "C": 0.1}], # identical to train row 0
        "paper_doi": ["10.1000/3"],
        "research_group_id": ["group_c"]
    })
    report_comp = auditor.audit_split(train_data, test_data_leaked_comp)
    assert report_comp["has_leakage"] is True
    assert report_comp["composition_leakage"]["detected"] is True
    assert report_comp["composition_leakage"]["overlap_count"] == 1

    # Introduce DOI leakage
    test_data_leaked_doi = pd.DataFrame({
        "composition": [{"Fe": 0.7, "C": 0.3}],
        "paper_doi": ["10.1000/2"], # identical to train row 1
        "research_group_id": ["group_c"]
    })
    report_doi = auditor.audit_split(train_data, test_data_leaked_doi)
    assert report_doi["has_leakage"] is True
    assert report_doi["doi_leakage"]["detected"] is True
    assert report_doi["doi_leakage"]["overlap_count"] == 1

    # Introduce L1 composition proximity overlap risk (differing by less than 2wt% e.g. 1wt%)
    test_data_overlap_risk = pd.DataFrame({
        "composition": [{"Fe": 0.895, "C": 0.105}], # differing by 0.005 + 0.005 = 0.01 L1 deviation
        "paper_doi": ["10.1000/4"],
        "research_group_id": ["group_d"]
    })
    report_overlap = auditor.audit_split(train_data, test_data_overlap_risk)
    assert report_overlap["train_test_overlap_risk"]["detected"] is True
    assert report_overlap["train_test_overlap_risk"]["high_similarity_count"] == 1


def test_benchmark_reporter_metrics():
    """Verify statistical MAE, RMSE, R², conformal coverage, refusal and OOD rates."""
    reporter = BenchmarkReporter()
    
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([105.0, 195.0, 290.0, 410.0, 650.0]) # Row 4 has high deviation
    
    conformal_intervals = [
        (90.0, 110.0),   # covers 100.0
        (180.0, 220.0),  # covers 200.0
        (280.0, 320.0),  # covers 300.0
        (380.0, 420.0),  # covers 400.0
        (480.0, 520.0)   # covers 500.0
    ]
    
    # Simulate a refusal on the last element (Row 4) due to high uncertainty or OOD
    refusal_mask = [False, False, False, False, True]
    families = ["steel", "steel", "hea", "hea", "aluminum"]
    ood_mask = [False, False, True, False, False] # Row 2 is OOD
    
    report = reporter.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        conformal_intervals=conformal_intervals,
        refusal_mask=refusal_mask,
        families=families,
        ood_mask=ood_mask
    )
    
    # Accepted calculations exclude the refused index 4
    # Expected global accepted MAE = average(|-5|, |+5|, |+10|, |-10|) = 7.5 MPa
    assert report["global_accepted_mae"] == 7.5
    assert report["global_accepted_rmse"] > 7.0
    # Since y_true and y_pred accepted perfectly map to ±10 range, R² should be highly positive (> 0.95)
    assert report["global_accepted_r2"] > 0.90
    
    assert report["refusal_rate"] == 0.20 # 1 out of 5 refused
    assert report["ood_rate"] == 0.20 # 1 out of 5 is OOD
    assert report["conformal_coverage"] == 1.0 # all 5 true targets fall within intervals
    
    # Check family metrics
    steel_metrics = report["family_breakdown"]["steel"]
    assert steel_metrics["samples"] == 2
    assert steel_metrics["mae"] == 5.0 # average of |-5| and |+5|
    assert steel_metrics["rmse"] == 5.0
    assert steel_metrics["ood_rate"] == 0.0


def test_experiment_registry_serialization(temp_dir, db_session):
    """Confirm SHA-256 consistency and database persistence within the ExperimentRegistry."""
    registry_file = os.path.join(temp_dir, "registry.json")
    registry = ExperimentRegistry(registry_path=registry_file)
    
    df = pd.DataFrame({"Fe": [0.8, 0.7], "C": [0.2, 0.3]})
    features = ["Fe", "C"]
    model_bytes = b"fake_serialized_ensemble_bytes"
    metrics = {"r2": 0.92, "mae": 12.5, "coverage": 0.90}
    
    # Log run
    run = registry.log_run(
        train_df=df,
        features=features,
        model_bytes=model_bytes,
        metrics=metrics,
        training_config={"epochs": 100},
        db_session=db_session
    )
    
    assert run["run_id"] is not None
    assert run["dataset_hash"] == registry.compute_dataset_hash(df)
    assert run["feature_hash"] == registry.compute_feature_hash(features)
    assert run["model_hash"] == registry.compute_model_hash(model_bytes)
    assert run["training_config"]["epochs"] == 100
    
    # Read history back from ledger file
    runs = registry.get_all_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run["run_id"]
    assert runs[0]["metrics"]["r2"] == 0.92
    assert runs[0]["training_config"]["epochs"] == 100
    
    # Verify SQL persistence
    db_rec = db_session.query(ExperimentRun).first()
    assert db_rec is not None
    assert db_rec.dataset_hash == run["dataset_hash"]
    assert db_rec.feature_hash == run["feature_hash"]
    assert db_rec.model_hash == run["model_hash"]
    assert db_rec.metrics_path == os.path.abspath(registry_file)

