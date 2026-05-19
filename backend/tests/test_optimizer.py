import pytest
import numpy as np
from backend.inverse.optimizer import AlloyOptimizer, ObjectiveTarget
from backend.inverse.pareto import classify_candidate, rank_pareto_candidates, filter_feasible
from backend.ml.fatigue_model import estimate_fatigue_limit, estimate_fracture_toughness, add_fatigue_fracture
from backend.models.predictor import predict_composition

def test_predict_composition():
    comp = {"Fe": 0.70, "Cr": 0.18, "Ni": 0.08, "C": 0.02, "Mn": 0.02}
    res = predict_composition(comp, "steel")
    assert "yield_strength_mpa" in res
    assert "tensile_strength_mpa" in res
    assert "hardness_hv" in res
    assert "elongation_pct" in res
    assert "corrosion_pren" in res
    assert isinstance(res["yield_strength_mpa"], float)

def test_fatigue_and_fracture_toughness():
    # Test fatigue
    fat = estimate_fatigue_limit(1000.0, "steel")
    assert fat["fatigue_limit_mpa"] == 475.0
    assert "Wöhler proxy" in fat["method"]

    fat_high = estimate_fatigue_limit(1600.0, "steel")
    assert fat_high["fatigue_limit_mpa"] == 640.0

    # Test fracture
    frac = estimate_fracture_toughness(800.0, 300.0, "steel")
    assert frac["fracture_toughness_kic_mpa_sqrtm"] > 0
    assert "Barsom-Rolfe" in frac["method"]
    assert "detectable" in frac["ndt_guidance"]

    # Test integrated prediction wrapper
    pred = {
        "predictions": {
            "yield_strength_mpa": {"mean": 600.0},
            "tensile_strength_mpa": {"mean": 800.0},
            "hardness_hv": {"mean": 250.0}
        }
    }
    updated = add_fatigue_fracture(pred, {"Fe": 1.0}, "steel")
    assert "fatigue" in updated
    assert "fracture_toughness" in updated

def test_pareto_functions():
    candidates = [
        {
            "predictions": {"yield_strength_mpa": 800, "corrosion_pren": 30},
            "composition": {"Fe": 0.75, "Cr": 0.15, "Ni": 0.10}
        },
        {
            "predictions": {"yield_strength_mpa": 1100, "corrosion_pren": 20},
            "composition": {"Fe": 0.70, "Cr": 0.10, "Ni": 0.20}
        }
    ]

    # Test rank_pareto_candidates
    priorities = ["yield_strength_mpa", "corrosion_pren"]
    ranked = rank_pareto_candidates(candidates, priorities)
    assert len(ranked) == 2
    assert ranked[0]["predictions"]["yield_strength_mpa"] == 1100

    # Test classify_candidate
    classified0 = classify_candidate(candidates[0])
    classified1 = classify_candidate(candidates[1])
    assert "Medium-strength" in classified0["classification"]
    assert "High-strength" in classified1["classification"]
    assert len(classified0["suggested_applications"]) > 0

    # Test filter_feasible
    constraints = {"frac_Cr": {"min": 0.12}}
    filtered = filter_feasible(candidates, constraints)
    assert len(filtered) == 1
    assert filtered[0]["composition"]["Cr"] == 0.15

def test_optimizer_ga_run():
    # Short test run
    targets = [
        ObjectiveTarget("yield_strength_mpa", "maximize", min_val=900),
        ObjectiveTarget("corrosion_pren", "maximize", min_val=20)
    ]
    constraints = {
        "frac_C": (0.0, 0.01),
        "frac_Cr": (0.0, 0.25),
        "frac_Ni": (0.0, 0.20)
    }
    
    optimizer = AlloyOptimizer(targets, constraints, alloy_family="steel")
    
    # Run only 3 generations, population 10 for speed
    results = list(optimizer.run(n_generations=3, pop_size=10))
    assert len(results) == 3
    
    last_res = results[-1]
    assert last_res.generation == 3
    assert last_res.population_size == 12
    assert len(last_res.pareto_front) >= 1
    assert last_res.constraint_violation_rate >= 0.0
