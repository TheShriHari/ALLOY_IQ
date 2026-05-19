import pytest
import os
import shutil
import pandas as pd

from backend.ml.publication.dataset_exporter import DatasetExporter
from backend.ml.publication.metadata_manifest import MetadataManifestTracker
from backend.ml.publication.reproducibility_bundle import ReproducibilityBundler
from backend.ml.publication.publication_report import PublicationReporter
from backend.ml.publication.DOI_packager import DOIPackager

@pytest.fixture
def temp_output_dir():
    path = "backend/tests/temp_publication_outputs"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        if os.path.exists(path):
            shutil.rmtree(path)

def test_dataset_exporter_correctness(temp_output_dir):
    """Ensure DatasetExporter writes valid files for datasets, parameters, and splits."""
    exporter = DatasetExporter(temp_output_dir)
    
    # 1. Clean dataset CSV
    df = pd.DataFrame({"Fe": [50.0, 60.0], "Ni": [50.0, 40.0], "user_id": ["u1", "u2"]})
    csv_path = exporter.export_cleaned_dataset(df)
    assert os.path.exists(csv_path)
    
    loaded_df = pd.read_csv(csv_path)
    assert "user_id" not in loaded_df.columns
    assert loaded_df.shape == (2, 2)
    
    # 2. Processing metadata
    proc = [{"specimen_id": "s1", "temp": 1100.0}]
    proc_path = exporter.export_processing_metadata(proc)
    assert os.path.exists(proc_path)
    
    # 3. Features definitions
    feats = {"Fe": "Iron weight percentage"}
    feat_path = exporter.export_feature_definitions(feats)
    assert os.path.exists(feat_path)
    
    # 4. Splits records
    split_path = exporter.export_split_records([0], [1])
    assert os.path.exists(split_path)


def test_metadata_manifest_integrity_and_checksum(temp_output_dir):
    """Assert MetadataManifestTracker generates structured manifests and audits checksum matches."""
    tracker = MetadataManifestTracker(temp_output_dir)
    
    manifest = tracker.generate_manifest(
        dataset_hash="sha_dataset_123",
        feature_hash="sha_feature_456",
        model_hash="sha_model_789",
        git_commit="git_commit_abc",
        experiment_ids=["exp1", "exp2"]
    )
    
    assert manifest["dataset_hash"] == "sha_dataset_123"
    assert manifest["git_commit"] == "git_commit_abc"
    
    manifest_path = tracker.save_manifest_to_file(manifest)
    assert os.path.exists(manifest_path)
    
    # Assert checksum verification
    assert tracker.verify_manifest_checksum(manifest_path, "sha_dataset_123") is True
    assert tracker.verify_manifest_checksum(manifest_path, "sha_dataset_tampered") is False


def test_reproducibility_bundle_generation(temp_output_dir):
    """Assert ReproducibilityBundler records system, configuration, and environment properties."""
    bundler = ReproducibilityBundler(temp_output_dir)
    
    configs = {"learning_rate": 0.01, "max_depth": 5}
    registry = [{"model_id": "model_1", "status": "active"}]
    benchmarks = {"yield_strength": {"r2": 0.85, "mae": 15.2}}
    lock = ["numpy==1.24.0", "scikit-learn==1.2.0"]
    
    bundle = bundler.assemble_reproducibility_bundle(
        configs=configs,
        registry_snapshot=registry,
        benchmarks=benchmarks,
        lock_file_lines=lock
    )
    
    assert "environment_details" in bundle
    assert bundle["training_configurations"]["learning_rate"] == 0.01
    assert bundle["dependency_lock_file"] == lock
    
    assert os.path.exists(os.path.join(temp_output_dir, "reproducibility_bundle.json"))


def test_scholarly_publication_report(temp_output_dir):
    """Verify PublicationReporter produces scholarly Markdown outlines without autonomous discovery claims."""
    reporter = PublicationReporter(temp_output_dir)
    
    benchmarks = {"yield_strength": {"r2": 0.82, "mae": 22.1, "conformal_coverage": 96.0}}
    blind = [{
        "experiment_id": "exp_novel_1",
        "composition": {"Ti": 50.0, "Al": 50.0},
        "comparison": {
            "yield_strength": {
                "predicted": 420.0,
                "measured": 435.0,
                "prediction_interval": [400.0, 450.0],
                "coverage_success": True
            }
        }
    }]
    leakage = {"composition_leakage": "Passed (Zero overlaps)"}
    failures = [{
        "record_id": "rec_fail_1",
        "uncertainty_width": 250.0,
        "refusal_reason": "High OOD distance",
        "flags": ["OOD", "ExtrapolationRisk"]
    }]
    family = {"hea": {"sample_count": 45, "mae": 18.5}}
    
    report_path = reporter.generate_publication_report(benchmarks, blind, leakage, failures, family)
    assert os.path.exists(report_path)
    
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Assert decision support assertions and lack of autonomous language
    assert "Materials Informatics Reproducibility & Validation Report" in content
    assert "interactive decision-support tool" in content
    assert "does not operate autonomously" in content
    assert "spillover" in content
    assert "TAMPERED" not in content


def test_doi_packager_staging_and_compression(temp_output_dir):
    """Verify DOIPackager stages nested folders and archives release outputs to ZIP."""
    # Staging workspace simulation
    workspace = os.path.join(temp_output_dir, "workspace")
    os.makedirs(workspace, exist_ok=True)
    
    packager = DOIPackager(workspace_root=workspace)
    
    # 1. Create structure
    package_dir = "publication_release"
    package_path = packager.create_release_package(package_dir)
    assert os.path.exists(package_path)
    
    # Subdirectories presence
    for sd in ["dataset", "models", "configs", "reports", "manifests"]:
        assert os.path.exists(os.path.join(package_path, sd))
        
    # 2. Stage a mock file
    mock_file = os.path.join(workspace, "mock_manifest.json")
    with open(mock_file, "w") as f:
        f.write("{}")
        
    staged_path = packager.stage_file_to_package(mock_file, package_dir, "manifests")
    assert os.path.exists(staged_path)
    assert os.path.basename(staged_path) == "mock_manifest.json"
    
    # 3. Zip package
    archive_zip = packager.archive_release_package(package_dir)
    assert os.path.exists(archive_zip)
    assert archive_zip.endswith(".zip")
