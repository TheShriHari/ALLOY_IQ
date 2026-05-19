import pytest
import pandas as pd
import numpy as np
from backend.data.processing import (
    ProcessingFeatureEngineer,
    AlloyDeduplicator,
    MaterialsValidationSplitter,
    ProcessingAwareFeaturePipeline
)

def test_processing_feature_imputations():
    """Verify categorical standardisation and temperature/cooling defaults."""
    engineer = ProcessingFeatureEngineer()
    
    # 1. Test Steels Annealing Defaults
    steel_rec = {
        "heat_treatment_category": "anneal",
        "cooling_method": "furnace",
        "manufacturing_route": "roll"
    }
    imputed = engineer.impute_missing_processing(steel_rec, family="steel")
    assert imputed["heat_treatment_category"] == "annealing"
    assert imputed["cooling_method"] == "furnace_cool"
    assert imputed["manufacturing_route"] == "wrought"
    # Verify defaults
    assert imputed["annealing_temperature"] == 850.0
    assert imputed["thermal_budget_category"] == "MEDIUM"

    # 2. Test HEA Cast Quench defaults
    hea_rec = {
        "heat_treatment_category": "quench",
        "annealing_temperature": 1150.0
    }
    imputed_hea = engineer.impute_missing_processing(hea_rec, family="hea")
    assert imputed_hea["heat_treatment_category"] == "quenching_tempering"
    assert imputed_hea["cooling_method"] == "water_quench"
    assert imputed_hea["manufacturing_route"] == "casting"
    assert imputed_hea["thermal_budget_category"] == "HIGH"


def test_alloy_deduplication():
    """Ensure composition-only duplicates are preserved, but exact composite key duplicates are pruned."""
    deduplicator = AlloyDeduplicator()
    
    records = [
        # Two identical compositions with completely different processes (Must be preserved!)
        {
            "composition": {"Fe": 0.98, "C": 0.02},
            "heat_treatment_category": "annealing",
            "annealing_temperature": 850.0,
            "cooling_method": "furnace_cool",
            "manufacturing_route": "wrought",
            "property_target": "yield_strength",
            "prediction": 350.0,
            "paper_doi": "10.1000/xyz"
        },
        {
            "composition": {"Fe": 0.98, "C": 0.02},
            "heat_treatment_category": "quenching_tempering",
            "annealing_temperature": 920.0,
            "cooling_method": "water_quench",
            "manufacturing_route": "wrought",
            "property_target": "yield_strength",
            "prediction": 800.0,
            "paper_doi": "10.1000/xyz"
        },
        # Absolute exact duplicate of the second record (Must be pruned!)
        {
            "composition": {"Fe": 0.98, "C": 0.02},
            "heat_treatment_category": "quenching_tempering",
            "annealing_temperature": 920.0,
            "cooling_method": "water_quench",
            "manufacturing_route": "wrought",
            "property_target": "yield_strength",
            "prediction": 800.0,
            "paper_doi": "10.1000/xyz"
        }
    ]
    
    deduped = deduplicator.deduplicate(records)
    
    # Assert composition duplicates are preserved, but exact composite duplicate is pruned
    assert len(deduped) == 2
    assert deduped[0]["heat_treatment_category"] == "annealing"
    assert deduped[1]["heat_treatment_category"] == "quenching_tempering"


def test_leakage_preventing_group_kfold():
    """Verify GroupKFold segments records strictly by group with zero paper leakage."""
    splitter = MaterialsValidationSplitter()
    
    df = pd.DataFrame([
        {"research_group_id": "group_a", "paper_doi": "10.1000/1", "composition": {"Fe": 0.99}},
        {"research_group_id": "group_a", "paper_doi": "10.1000/1", "composition": {"Fe": 0.95}},
        {"research_group_id": "group_b", "paper_doi": "10.1000/2", "composition": {"Fe": 0.90}},
        {"research_group_id": "group_b", "paper_doi": "10.1000/2", "composition": {"Fe": 0.85}},
        {"research_group_id": "group_c", "paper_doi": "10.1000/3", "composition": {"Fe": 0.80}},
        {"research_group_id": "group_d", "paper_doi": "10.1000/4", "composition": {"Fe": 0.75}}
    ])
    
    splits = splitter.split(df, n_splits=3)
    
    # Assert zero overlapping papers/groups between train and val
    for fold, (train_df, val_df) in enumerate(splits):
        train_groups = set(splitter.generate_groups(train_df))
        val_groups = set(splitter.generate_groups(val_df))
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Overlapping leakage in fold {fold}: {overlap}"


def test_risk_flags_engine():
    """Verify materials informatics risk flags are correctly generated."""
    splitter = MaterialsValidationSplitter()
    
    # Normal training set references
    train_df = pd.DataFrame([
        {"alloy_family": "steel", "annealing_temperature": 850.0, "composition": {"Fe": 0.98, "C": 0.02}},
        {"alloy_family": "steel", "annealing_temperature": 900.0, "composition": {"Fe": 0.97, "C": 0.03}},
        {"alloy_family": "steel", "annealing_temperature": 800.0, "composition": {"Fe": 0.98, "Mn": 0.02}}
    ])
    
    # Case 1: Out of Distribution Temperature + Sparse family
    test_1 = {
        "alloy_family": "hea", # SPARSE (Not in training set)
        "annealing_temperature": 1250.0, # OOD (> 900 + 100)
        "heat_treatment_category": "annealing",
        "composition": {"Fe": 0.98, "C": 0.02}
    }
    flags_1 = splitter.evaluate_risk_flags(train_df, test_1)
    assert "OOD" in flags_1
    assert "SPARSE_FAMILY" in flags_1

    # Case 2: Missing processing
    test_2 = {
        "alloy_family": "steel",
        "heat_treatment_category": "annealing",
        "annealing_temperature": 0.0, # Missing annealing temp!
        "composition": {"Fe": 0.98, "C": 0.02}
    }
    flags_2 = splitter.evaluate_risk_flags(train_df, test_2)
    assert "MISSING_PROCESSING" in flags_2

    # Case 3: Low confidence (composition space distance > 15%)
    test_3 = {
        "alloy_family": "steel",
        "heat_treatment_category": "none",
        "annealing_temperature": 0.0,
        # Closest is Fe-C. Here we introduce massive Ni (20%) + Cr (20%) -> high deviation!
        "composition": {"Fe": 0.58, "Ni": 0.20, "Cr": 0.20, "C": 0.02}
    }
    flags_3 = splitter.evaluate_risk_flags(train_df, test_3)
    assert "LOW_CONFIDENCE" in flags_3


def test_feature_pipeline_versioning_and_compat():
    """Verify v1 (composition legacy) and v2 (processing expansion) execution flows."""
    pipeline_v1 = ProcessingAwareFeaturePipeline(alloy_family="steel", version="v1")
    pipeline_v2 = ProcessingAwareFeaturePipeline(alloy_family="steel", version="v2")
    
    record = {
        "composition": {"Fe": 0.70, "Cr": 0.18, "Ni": 0.08, "Mn": 0.02, "C": 0.02},
        "heat_treatment_category": "anneal",
        "annealing_temperature": 850.0,
        "cooling_method": "furnace",
        "manufacturing_route": "roll"
    }
    
    # Verify Legacy v1 flow has Magpie/Steel features but no processing one-hot indicators
    res_v1 = pipeline_v1.transform_record(record)
    assert "carbon_equivalent" in res_v1
    assert "mean_atomic_radius" in res_v1
    assert "feat_proc_annealing_temp" not in res_v1
    assert res_v1["feature_pipeline_version"] == "2.0.0"

    # Verify v2 flow includes Magpie/Steel AND all the one-hot categories
    res_v2 = pipeline_v2.transform_record(record)
    assert "carbon_equivalent" in res_v2
    assert "mean_atomic_radius" in res_v2
    assert "feat_proc_annealing_temp" in res_v2
    assert res_v2["feat_proc_annealing_temp"] == 850.0
    assert res_v2["feat_proc_ht_annealing"] == 1.0
    assert res_v2["feat_proc_cool_furnace_cool"] == 1.0
    assert res_v2["feat_proc_route_wrought"] == 1.0
    assert res_v2["feat_proc_budget_MEDIUM"] == 1.0
