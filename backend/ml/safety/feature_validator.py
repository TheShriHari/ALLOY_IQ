from typing import Dict, Any, List, Tuple

class FeatureValidator:
    """
    Enforces strict pre-inference input sanity validations.
    Identifies missing critical descriptors or impossible manufacturing setups before model execution.
    """

    @staticmethod
    def validate_inputs(
        composition: Dict[str, float],
        processing: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validates the raw alloy composition and heat treatment inputs.
        
        Returns:
            is_valid (bool): True if inputs are clean and safe for inference.
            error_msgs (List[str]): List of validation errors discovered.
        """
        is_valid = True
        error_msgs = []

        # 1. Composition validation
        if not composition:
            is_valid = False
            error_msgs.append("Empty composition provided. Must specify chemical constituents.")
            return is_valid, error_msgs

        total_pct = sum(composition.values())
        if not (99.0 <= total_pct <= 101.0):
            is_valid = False
            error_msgs.append(f"Invalid composition: sum of elements is {total_pct:.2f}%. Must be 100+/-1%.")

        for el, frac in composition.items():
            if frac < 0.0:
                is_valid = False
                error_msgs.append(f"Invalid element weight fraction: {el} is {frac}%. Constituents cannot be negative.")

        # 2. Processing-aware validations
        if processing:
            anneal_temp = processing.get("annealing_temperature")
            cooling = processing.get("cooling_method")
            manufacturing_route = processing.get("manufacturing_route")
            thermal_budget = processing.get("thermal_budget_category")

            # Check impossible temperature ranges
            if anneal_temp is not None:
                try:
                    t_val = float(anneal_temp)
                    if t_val < 0.0:
                        is_valid = False
                        error_msgs.append(f"Impossible annealing temperature: {t_val}°C. Values must be non-negative.")
                    elif t_val > 1600.0:
                        is_valid = False
                        error_msgs.append(f"Impossible annealing temperature: {t_val}°C. Exceeds liquidus melting point of steel/nickel.")
                except (ValueError, TypeError):
                    is_valid = False
                    error_msgs.append("Annealing temperature must be a numeric value.")

            # Check missing critical processing couplings
            # If thermal treatment is specified, a cooling method should be defined
            if anneal_temp is not None and float(anneal_temp) > 100.0 and not cooling:
                # Flag as invalid to force complete processing configurations
                is_valid = False
                error_msgs.append("Missing critical processing coupling: High-temperature annealing specified without a cooling method.")

            # If manufacturing route is specified, verify it has a valid label
            if manufacturing_route is not None:
                r_val = str(manufacturing_route).strip().lower()
                valid_routes = {"wrought", "cast", "powder", "additive", "rolled", "forged", "unknown", ""}
                if r_val and not any(route in r_val for route in valid_routes):
                    is_valid = False
                    error_msgs.append(f"Invalid manufacturing route specified: '{manufacturing_route}'.")

        return is_valid, error_msgs
