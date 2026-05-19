from typing import Dict, List, Any, Tuple, Optional
from backend.ml.safety.physics_constraints import PhysicsConstraints

# Define standardized risk flags
PHYSICS_VIOLATION = "PHYSICS_VIOLATION"
INVALID_INPUT = "INVALID_INPUT"
CONFLICTING_PROPERTIES = "CONFLICTING_PROPERTIES"
UNSTABLE_FEATURES = "UNSTABLE_FEATURES"

class PredictionGuardrails:
    """
    Validates model predictions and inputs to intercept, flag, or refuse
    unphysical or extrapolated predictions.
    """
    def __init__(self):
        self.physics = PhysicsConstraints()

    def audit_prediction(
        self,
        composition: Dict[str, float],
        predictions: Dict[str, float],
        ood_score: Optional[float] = None,
        is_ood: bool = False,
        processing: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Intercepts the prediction job and performs multi-tier guardrail auditing.
        
        Returns:
            is_refused (bool): True if prediction should be blocked/refused.
            triggered_flags (List[str]): Standardized triggered risk flags.
            messages (List[str]): Plain-English validation warning logs.
        """
        is_refused = False
        triggered_flags = []
        messages = []

        # 1. Input validations
        comp_violations = self.physics.validate_composition(composition)
        if comp_violations:
            is_refused = True
            triggered_flags.append(INVALID_INPUT)
            messages.extend(comp_violations)

        # 2. Physics consistency checks
        physics_violations = self.physics.check_physical_sanity(composition, predictions)
        if physics_violations:
            # Distinguish between absolute physics violations and property conflicts
            ys = predictions.get("yield_strength_mpa", 0.0)
            uts = predictions.get("tensile_strength_mpa", 0.0)
            hv = predictions.get("hardness_hv", 0.0)

            # Absolute violations (refusal triggers)
            has_absolute = any(
                "exceeds ultimate" in v or "must be strictly positive" in v or "Composition sum" in v
                for v in physics_violations
            )
            if has_absolute:
                is_refused = True
                triggered_flags.append(PHYSICS_VIOLATION)
            
            # Conflicting properties flags (e.g. unphysical hardness/strength ratios)
            has_conflict = any("ratio" in v or "elongation" in v for v in physics_violations)
            if has_conflict:
                triggered_flags.append(CONFLICTING_PROPERTIES)

            messages.extend(physics_violations)

        # 3. Extrapolation / OOD checking
        if is_ood or (ood_score is not None and ood_score > 3.0):  # Mahalanobis threshold
            triggered_flags.append(UNSTABLE_FEATURES)
            messages.append("Prediction displays elevated extrapolation risk (OOD). Unstable features detected.")
            # Set refusal if extrapolation is severe (e.g., ood_score > 6.0)
            if ood_score is not None and ood_score > 6.0:
                is_refused = True
                messages.append("Extreme extrapolation beyond safe ML training space. Prediction refused.")

        # 4. Processing-aware sanity checking
        if processing:
            anneal_temp = processing.get("annealing_temperature")
            cooling = processing.get("cooling_method")
            if anneal_temp is not None:
                if anneal_temp < 0.0 or anneal_temp > 1600.0:
                    is_refused = True
                    triggered_flags.append(INVALID_INPUT)
                    messages.append(f"Annealing temperature ({anneal_temp:.1f}°C) exceeds metallurgical bounds [0°C, 1600°C].")
                
                # Annealing but no cooling specified is unstable
                if not cooling:
                    triggered_flags.append(UNSTABLE_FEATURES)
                    messages.append("Heat treatment (annealing) specified without associated cooling route (possible unstable feature setup).")

        # De-duplicate flags
        triggered_flags = list(set(triggered_flags))

        return is_refused, triggered_flags, messages
