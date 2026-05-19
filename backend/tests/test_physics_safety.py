import pytest
import numpy as np

from backend.ml.safety.physics_constraints import PhysicsConstraints
from backend.ml.safety.prediction_guardrails import (
    PredictionGuardrails,
    PHYSICS_VIOLATION,
    INVALID_INPUT,
    CONFLICTING_PROPERTIES,
    UNSTABLE_FEATURES
)
from backend.ml.safety.feature_validator import FeatureValidator
from backend.ml.safety.explanation_engine import SafetyExplanationEngine


def test_physics_composition_sums():
    """Ensure composition validators catch non-100% sums and negative constituent ratios."""
    pc = PhysicsConstraints()
    
    # 1. Invalid sum
    bad_sum = {"Fe": 50.0, "Cr": 20.0, "Ni": 10.0}  # sums to 80
    v1 = pc.validate_composition(bad_sum)
    assert any("outside the physically valid" in v for v in v1)
    
    # 2. Negative fraction
    bad_frac = {"Fe": 110.0, "Cr": -10.0}
    v2 = pc.validate_composition(bad_frac)
    assert any("Negative weight fraction" in v for v in v2)
    
    # 3. Clean composition
    good = {"Fe": 70.0, "Cr": 20.0, "Ni": 10.0}
    v3 = pc.validate_composition(good)
    assert len(v3) == 0


def test_rule_of_mixtures_density_and_modulus():
    """Verify density and elastic modulus physical boundary sanity testing."""
    pc = PhysicsConstraints()
    
    # Normal steel-like composition
    steel = {"Fe": 74.0, "Cr": 18.0, "Ni": 8.0}
    density = pc.estimate_density(steel)
    modulus = pc.estimate_elastic_modulus(steel)
    
    assert 7.0 <= density <= 8.5
    assert 190.0 <= modulus <= 230.0
    
    # Run sanity bounds test
    preds = {
        "yield_strength_mpa": 400.0,
        "tensile_strength_mpa": 600.0,
        "hardness_hv": 180.0,
        "elongation_pct": 20.0
    }
    violations = pc.check_physical_sanity(steel, preds)
    assert len(violations) == 0


def test_physics_yield_exceeds_tensile_violation():
    """Assert yield strength exceeding ultimate tensile strength triggers PHYSICS_VIOLATION refusal."""
    pc = PhysicsConstraints()
    guardrails = PredictionGuardrails()
    
    comp = {"Fe": 74.0, "Cr": 18.0, "Ni": 8.0}
    
    # Yield > Tensile
    bad_preds = {
        "yield_strength_mpa": 800.0,
        "tensile_strength_mpa": 750.0,  # lower than yield!
        "hardness_hv": 240.0,
        "elongation_pct": 12.0
    }
    
    violations = pc.check_physical_sanity(comp, bad_preds)
    assert any("exceeds ultimate tensile strength" in v for v in violations)
    
    # Verify guardrails refuse this prediction
    is_refused, flags, messages = guardrails.audit_prediction(comp, bad_preds)
    assert is_refused is True
    assert PHYSICS_VIOLATION in flags


def test_guardrails_ood_extrapolation_refusal():
    """Verify guardrails flag extreme extrapolation and refuse predictions."""
    guardrails = PredictionGuardrails()
    comp = {"Fe": 74.0, "Cr": 18.0, "Ni": 8.0}
    preds = {
        "yield_strength_mpa": 400.0,
        "tensile_strength_mpa": 600.0,
        "hardness_hv": 180.0,
        "elongation_pct": 20.0
    }
    
    # 1. Mild extrapolation: flag UNSTABLE_FEATURES but don't refuse
    is_refused, flags, messages = guardrails.audit_prediction(comp, preds, ood_score=4.0)
    assert is_refused is False
    assert UNSTABLE_FEATURES in flags
    
    # 2. Extreme extrapolation: refuse prediction
    is_refused_extrap, flags_extrap, messages_extrap = guardrails.audit_prediction(comp, preds, ood_score=7.0)
    assert is_refused_extrap is True
    assert UNSTABLE_FEATURES in flags_extrap


def test_impossible_processing_combinations():
    """Ensure feature validators catch missing processing routes and extreme temperatures."""
    validator = FeatureValidator()
    
    comp = {"Fe": 74.0, "Cr": 18.0, "Ni": 8.0}
    
    # 1. Missing cooling coupling
    bad_coupling = {"annealing_temperature": 900.0}  # missing cooling method
    is_valid, msgs = validator.validate_inputs(comp, bad_coupling)
    assert is_valid is False
    assert any("Missing critical processing coupling" in m for m in msgs)
    
    # 2. Impossible temperature (above melting point)
    bad_temp = {"annealing_temperature": 1800.0, "cooling_method": "water_quenched"}
    is_valid_t, msgs_t = validator.validate_inputs(comp, bad_temp)
    assert is_valid_t is False
    assert any("exceeds liquidus melting point" in m.lower() for m in msgs_t)


def test_explanation_engine_compilation():
    """Verify SafetyExplanationEngine correctly packages rationales, nearest neighbors, and SHAP contributors."""
    engine = SafetyExplanationEngine()
    
    comp = {"Fe": 74.0, "Cr": 18.0, "Ni": 8.0}
    preds = {
        "yield_strength_mpa": 420.0,
        "tensile_strength_mpa": 620.0,
        "hardness_hv": 190.0,
        "elongation_pct": 22.0
    }
    shap = {
        "yield_strength_mpa": {"frac_Cr": 40.0, "frac_Ni": 12.0, "frac_Fe": -5.0}
    }
    intervals = {
        "yield_strength_mpa": {"mean": 420.0, "lower": 380.0, "upper": 460.0}
    }
    
    report = engine.generate_explanation(
        composition=comp,
        predictions=preds,
        shap_dicts=shap,
        conformal_intervals=intervals,
        is_ood=False
    )
    
    assert report["is_refused"] is False
    assert "triggered_flags" in report
    assert len(report["top_features"]) > 0
    assert "chromium" in report["top_features"][0]["element_name"]
    assert len(report["nearest_neighbors"]) > 0
    assert "human_readable_rationale" in report
    assert "Rule of Mixtures" in report["human_readable_rationale"]
