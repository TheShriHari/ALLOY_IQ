from typing import Dict, List, Any, Tuple

# Densities in g/cm3
ELEMENT_DENSITIES: Dict[str, float] = {
    "Fe": 7.87, "Cr": 7.19, "Ni": 8.90, "Mo": 10.28, "Mn": 7.21,
    "Co": 8.90, "Al": 2.70, "Ti": 4.50, "Cu": 8.96, "V": 6.11,
    "W": 19.25, "Nb": 8.57, "Si": 2.33, "Mg": 1.74, "Zn": 7.14,
    "C": 2.26, "B": 2.34, "N": 0.00125, "S": 2.07, "P": 1.82
}

# Elastic Moduli in GPa
ELEMENT_MODULI: Dict[str, float] = {
    "Fe": 211.0, "Cr": 279.0, "Ni": 200.0, "Mo": 329.0, "Mn": 191.0,
    "Co": 209.0, "Al": 70.0, "Ti": 116.0, "Cu": 128.0, "V": 128.0,
    "W": 411.0, "Nb": 105.0, "Si": 150.0, "Mg": 45.0, "Zn": 108.0,
    "C": 10.0, "B": 400.0, "N": 1.0, "S": 10.0, "P": 10.0
}

class PhysicsConstraints:
    """
    Implements fundamental metallurgy and solid mechanics validation rules.
    Acts as a deterministic validator for ML predictions and inputs.
    """

    @staticmethod
    def validate_composition(composition: Dict[str, float]) -> List[str]:
        """Validates that alloy fractions sum to 100% (+/-1%) and are non-negative."""
        violations = []
        if not composition:
            violations.append("Empty composition vector provided.")
            return violations

        total = sum(composition.values())
        if not (99.0 <= total <= 101.0):
            violations.append(f"Composition sum is {total:.2f}%, which is outside the physically valid 100+/-1% range.")

        for element, frac in composition.items():
            if frac < 0:
                violations.append(f"Negative weight fraction found for element {element}: {frac}%.")
        return violations

    @staticmethod
    def estimate_density(composition: Dict[str, float]) -> float:
        """Estimates bulk density using the Rule of Mixtures (g/cm3)."""
        total_frac = sum(composition.values()) or 100.0
        weighted_density = 0.0
        for el, pct in composition.items():
            density = ELEMENT_DENSITIES.get(el.strip(), 5.0)  # Default fallback density
            weighted_density += (pct / total_frac) * density
        return weighted_density

    @staticmethod
    def estimate_elastic_modulus(composition: Dict[str, float]) -> float:
        """Estimates bulk elastic modulus using the Rule of Mixtures (GPa)."""
        total_frac = sum(composition.values()) or 100.0
        weighted_modulus = 0.0
        for el, pct in composition.items():
            modulus = ELEMENT_MODULI.get(el.strip(), 120.0)  # Default fallback modulus
            weighted_modulus += (pct / total_frac) * modulus
        return weighted_modulus

    def check_physical_sanity(
        self,
        composition: Dict[str, float],
        predictions: Dict[str, float]
    ) -> List[str]:
        """
        Runs comprehensive physical consistency checks on alloy compositions and predicted mechanical properties.
        Target properties map to target names:
          yield_strength_mpa, tensile_strength_mpa, hardness_hv, elongation_pct
        """
        violations = []

        # 1. Composition verification
        comp_violations = self.validate_composition(composition)
        violations.extend(comp_violations)

        # 2. Estimated properties validation
        density = self.estimate_density(composition)
        if not (0.5 <= density <= 25.0):
            violations.append(f"Calculated density {density:.2f} g/cm³ is outside the physical sanity bound [0.5, 25.0] g/cm³.")

        modulus = self.estimate_elastic_modulus(composition)
        if not (1.0 <= modulus <= 700.0):
            violations.append(f"Calculated elastic modulus {modulus:.2f} GPa is outside the physical sanity bound [1.0, 700.0] GPa.")

        # 3. Target properties values consistency checks
        ys = predictions.get("yield_strength_mpa", 0.0)
        uts = predictions.get("tensile_strength_mpa", 0.0)
        hv = predictions.get("hardness_hv", 0.0)
        elong = predictions.get("elongation_pct", 0.0)

        # Non-negative properties check
        for name, val in [("yield_strength_mpa", ys), ("tensile_strength_mpa", uts), ("hardness_hv", hv), ("elongation_pct", elong)]:
            if val <= 0:
                violations.append(f"Predicted value for {name} ({val:.1f}) must be strictly positive.")

        # Ultimate Tensile Strength must exceed Yield Strength
        if ys > uts:
            violations.append(f"Yield strength ({ys:.1f} MPa) exceeds ultimate tensile strength ({uts:.1f} MPa) which violates physical plasticity limits.")

        # Strength-to-Hardness ratios validation
        if hv > 0:
            uts_to_hv = uts / hv
            if uts_to_hv > 10.0 or uts_to_hv < 0.1:
                violations.append(f"Predicted tensile-strength-to-hardness ratio ({uts_to_hv:.2f}) is physically impossible for metallic structures.")

        # Elongation bounds check
        if not (0.1 <= elong <= 100.0):
            violations.append(f"Predicted elongation ({elong:.2f}%) is outside the realistic bounds [0.1%, 100.0%].")

        return violations
