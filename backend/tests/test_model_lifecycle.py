import os
import io
import shutil
import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import joblib

from backend.db.models import Base, ModelRegistryEntry, ModelTrainingJob
from backend.ml.lifecycle.ensemble_manager import EnsembleManager
from backend.ml.lifecycle.trainer import ModelTrainer
from backend.ml.lifecycle.model_registry import ModelRegistry
from backend.ml.lifecycle.model_loader import ModelLoader
from backend.ml.lifecycle.training_scheduler import TrainingScheduler


@pytest.fixture
def temp_dirs():
    """Provides unique temporary directories for model artifacts and experiments registry ledger files."""
    models_dir = "./temp_test_models"
    experiments_file = "./temp_test_experiments/registry.json"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(experiments_file), exist_ok=True)
    
    yield models_dir, experiments_file
    
    # Clean up after tests
    for p in (models_dir, os.path.dirname(experiments_file)):
        if os.path.exists(p):
            shutil.rmtree(p)


@pytest.fixture
def db_session():
    """Clean SQLite in-memory session with lifecycle and registry schemas."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_dataset():
    """Clean training dataset containing processing-aware features, group columns, and log-targets."""
    data = {
        "alloy_family": ["steel"] * 12,
        "cooling_method": ["wrought", "cast", "wrought", "cast"] * 3,
        "paper_doi": ["10.1000/a"] * 4 + ["10.1000/b"] * 4 + ["10.1000/c"] * 4,
        "yield_strength_mpa": [400, 420, 440, 410, 500, 520, 530, 510, 300, 310, 320, 330],
        "tensile_strength_mpa": [500, 520, 540, 510, 600, 620, 630, 610, 400, 410, 420, 430],
        "hardness_hv": [120, 130, 140, 125, 150, 160, 165, 155, 100, 105, 110, 108],
        "elongation_pct": [20, 19, 18, 19.5, 15, 14, 13, 14.5, 25, 24, 23, 23.5]
    }
    return pd.DataFrame(data)


def test_ensemble_manager_fitting(mock_dataset):
    """Verify EnsembleManager fit_and_calibrate trains multi-output stacking regressors."""
    manager = EnsembleManager(coverage="rich", confidence=0.90)
    
    features = ["cooling_method"]
    # Hot-encode categorical columns
    X = pd.get_dummies(mock_dataset[features], drop_first=True)
    y = mock_dataset[["yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]]
    
    # Train stacker and calibrate MAPIE bounds
    manager.fit_and_calibrate(X_train=X, y_train=y, X_cal=X, y_cal=y)
    
    assert manager.pipeline is not None
    assert manager.conformal is not None
    
    # Run test prediction
    res = manager.conformal.predict(X.iloc[:1].values)
    assert "yield_strength_mpa" in res
    assert res["yield_strength_mpa"]["mean"] > 0
    assert res["yield_strength_mpa"]["lower"] <= res["yield_strength_mpa"]["upper"]


def test_group_kfold_leakage_prevention(temp_dirs, mock_dataset):
    """Ensure ModelTrainer partitions groups cleanly with GroupKFold cross-validation."""
    models_dir, exp_file = temp_dirs
    trainer = ModelTrainer(registry_path=exp_file, confidence=0.90)
    
    features = ["cooling_method"]
    X_df = pd.get_dummies(mock_dataset[features], drop_first=True)
    df_fit = pd.concat([X_df, mock_dataset[["paper_doi", "yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]]], axis=1)
    
    feature_cols = list(X_df.columns)
    target_cols = ["yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]
    
    # Execute GroupKFold
    manager, metrics = trainer.run_group_kfold_training(
        df=df_fit,
        features=feature_cols,
        target_columns=target_cols,
        group_column="paper_doi",
        coverage="rich",
        n_splits=3
    )
    
    assert manager.pipeline is not None
    assert "fold_checkpoints" in metrics
    assert len(metrics["fold_checkpoints"]) == 3
    
    # Check that each fold checkpoint recorded evaluation metrics
    for chk in metrics["fold_checkpoints"]:
        assert "fold" in chk
        assert "yield_strength_mpa_r2" in chk["metrics"]


def test_feature_hash_mismatch_prevention(temp_dirs, mock_dataset):
    """Verify ModelTrainer and ModelLoader reject incompatible feature signature hashes."""
    models_dir, exp_file = temp_dirs
    trainer = ModelTrainer(registry_path=exp_file)
    loader = ModelLoader(registry_dir=models_dir)
    
    features = ["cooling_method"]
    X_df = pd.get_dummies(mock_dataset[features], drop_first=True)
    df_fit = pd.concat([X_df, mock_dataset[["paper_doi", "yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]]], axis=1)
    
    feature_cols = list(X_df.columns)
    
    # Mismatch feature hash assertion during fit verification
    with pytest.raises(ValueError, match="Feature hash mismatch"):
        trainer.verify_features(df_fit, feature_cols, expected_feature_hash="invalid_hash_signature")


def test_training_resumable_scheduler(temp_dirs, mock_dataset, db_session):
    """Confirm TrainingScheduler can cache fold checkpoints and resume/skip finished training folds."""
    models_dir, exp_file = temp_dirs
    scheduler = TrainingScheduler(jobs_dir=models_dir)
    
    features = ["cooling_method"]
    X_df = pd.get_dummies(mock_dataset[features], drop_first=True)
    df_fit = pd.concat([X_df, mock_dataset[["alloy_family", "paper_doi", "yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]]], axis=1)
    
    feature_cols = list(X_df.columns)
    target_cols = ["yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]
    
    # Register job record
    job_id = "scheduler_test_job"
    job_rec = ModelTrainingJob(id=job_id, alloy_family="steel", status="pending")
    db_session.add(job_rec)
    db_session.commit()
    
    # Cache a mock checkpoint for fold 0 and fold 1 to simulate a previous aborted run
    mock_chk_metrics = {"metrics": {"yield_strength_mpa_r2": 0.88}}
    scheduler.save_fold_checkpoint(job_id, 0, mock_chk_metrics)
    scheduler.save_fold_checkpoint(job_id, 1, mock_chk_metrics)
    
    # Execute (should load fold 0 & 1 from cache, fit fold 2 and succeed)
    metrics = scheduler.execute_training_job(
        job_id=job_id,
        df=df_fit,
        features=feature_cols,
        target_columns=target_cols,
        group_column="paper_doi",
        coverage="rich",
        n_splits=3,
        db_session=db_session
    )
    
    assert metrics is not None
    assert len(metrics["fold_checkpoints"]) == 3
    # Checkpoints 0 and 1 must hold our mock metrics values
    assert metrics["fold_checkpoints"][0]["metrics"]["yield_strength_mpa_r2"] == 0.88
    
    # Verify SQL updates
    updated_job = db_session.query(ModelTrainingJob).filter_by(id=job_id).first()
    assert updated_job.status == "complete"
    assert updated_job.progress == 1.0


def test_registry_persistence_and_rollback_loading(temp_dirs, mock_dataset, db_session):
    """Ensure ModelRegistry serializes logs persistently, and ModelLoader falls back recursively during unpickling failure."""
    models_dir, exp_file = temp_dirs
    registry = ModelRegistry(registry_dir=models_dir)
    loader = ModelLoader(registry_dir=models_dir)
    
    features = ["cooling_method"]
    X_df = pd.get_dummies(mock_dataset[features], drop_first=True)
    feature_cols = list(X_df.columns)
    
    # 1. Compile dummy serialized model bytes
    dummy_manager = EnsembleManager()
    dummy_manager.build_pipeline()
    # calibrate dummy conformal
    y_dummy = mock_dataset[["yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]]
    dummy_manager.fit_and_calibrate(X_df, y_dummy, X_df, y_dummy)
    
    pipe_io = io.BytesIO()
    joblib.dump(dummy_manager.pipeline, pipe_io)
    pipe_bytes = pipe_io.getvalue()
    
    conf_io = io.BytesIO()
    dummy_manager.conformal.save(conf_io)
    conf_bytes = conf_io.getvalue()
    
    feature_hash = loader.compute_feature_hash(feature_cols)
    dataset_hash = "mock_dataset_sha256"
    
    # Register Model 1 (Corrupt model to trigger unpickling failure rollback check)
    registry.register_model(
        model_id="corrupt_model",
        pipeline_bytes=b"completely_corrupt_pickled_bytes",
        conformal_bytes=b"completely_corrupt_conformal_bytes",
        feature_hash=feature_hash,
        dataset_hash=dataset_hash,
        metrics={"r2": 0.85},
        alloy_family="steel",
        status="validated",
        db_session=db_session
    )
    
    # Register Model 2 (Valid Model)
    registry.register_model(
        model_id="valid_model",
        pipeline_bytes=pipe_bytes,
        conformal_bytes=conf_bytes,
        feature_hash=feature_hash,
        dataset_hash=dataset_hash,
        metrics={"r2": 0.94},
        alloy_family="steel",
        status="validated",
        db_session=db_session
    )
    
    # Retrieve registry entries
    entries = db_session.query(ModelRegistryEntry).all()
    assert len(entries) == 2
    
    # 2. Attempt model loading: Latest model is validated (Model 2, valid). Let's corrupt Model 2 locally by overwriting its stack files!
    # If the valid model's pickles become corrupted, it must rollback to Model 1. But wait, Model 1 is also corrupt, so it falls back again.
    # Let's arrange so the newest model is corrupt, and the previous model is valid, to verify unpickling failures trigger loading rollback fallback!
    # Overwrite registry order to make the corrupt model the newest one
    from datetime import datetime, timedelta
    corrupt_db = db_session.query(ModelRegistryEntry).filter_by(model_id="corrupt_model").first()
    valid_db = db_session.query(ModelRegistryEntry).filter_by(model_id="valid_model").first()
    # Change timestamp of corrupt_model to be newer
    corrupt_db.created_at = datetime.utcnow()
    valid_db.created_at = datetime.utcnow() - timedelta(days=1)
    db_session.commit()
    
    # Now, corrupt_model is checked first. Deserialization of corrupt_model will fail.
    # It must trigger the fallback loop and load valid_model successfully!
    loaded_manager = loader.load_active_model(
        alloy_family="steel",
        current_features=feature_cols,
        db_session=db_session
    )
    
    assert loaded_manager is not None
    assert loaded_manager.pipeline is not None
