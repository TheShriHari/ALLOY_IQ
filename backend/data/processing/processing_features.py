import numpy as np
import pandas as pd
from typing import Dict, Any

class ProcessingFeatureEngineer:
    """
    Standardizes raw, heterogeneous processing parameters,
    normalizes categoricals, handles missing data via family-based defaults,
    and derives thermodynamic descriptors like thermal budget.
    """
    
    # Standard mapping dictionaries for normalization
    COOLING_MAP = {
        "water": "water_quench",
        "wq": "water_quench",
        "quenched": "water_quench",
        "air": "air_cool",
        "ac": "air_cool",
        "normalizing": "air_cool",
        "furnace": "furnace_cool",
        "fc": "furnace_cool",
        "slow": "furnace_cool",
        "oil": "oil_quench",
        "oq": "oil_quench",
        "none": "none",
        "as_cast": "none"
    }
    
    TREATMENT_MAP = {
        "anneal": "annealing",
        "homogeniz": "homogenization",
        "solution": "solution_treatment",
        "quench": "quenching_tempering",
        "temper": "quenching_tempering",
        "aging": "solution_aging",
        "cast": "as_cast",
        "none": "none"
    }

    MANUFACTURING_MAP = {
        "cast": "casting",
        "wrought": "wrought",
        "roll": "wrought",
        "forge": "wrought",
        "powder": "powder_metallurgy",
        "pm": "powder_metallurgy",
        "additive": "additive_manufacturing",
        "am": "additive_manufacturing",
        "print": "additive_manufacturing",
        "slm": "additive_manufacturing"
    }

    # Family-based default annealing temperatures (C)
    FAMILY_ANNEALING_DEFAULTS = {
        "steel": 850.0,
        "hea": 1100.0,
        "aluminum": 400.0,
        "unknown": 600.0
    }

    def normalize_categorical(self, val: str, mapping: Dict[str, str], default: str = "unknown") -> str:
        """Helper to match and normalize noisy raw strings to standardized keys."""
        if not val or not isinstance(val, str):
            return default
        clean_val = val.strip().lower()
        # Sort keys by length descending to match more specific/longer terms first,
        # preventing collision (e.g. "ac" matching inside "furnace")
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        for k in sorted_keys:
            if k in clean_val:
                return mapping[k]
        return default

    def impute_missing_processing(self, record: Dict[str, Any], family: str = "steel") -> Dict[str, Any]:
        """
        Imputes missing processing parameters based on standard
        metallurgical heuristics tailored to each alloy family.
        """
        imputed = record.copy()
        
        # 1. Normalize Categories
        imputed["heat_treatment_category"] = self.normalize_categorical(
            record.get("heat_treatment_category"), self.TREATMENT_MAP, "unknown"
        )
        imputed["cooling_method"] = self.normalize_categorical(
            record.get("cooling_method"), self.COOLING_MAP, "unknown"
        )
        imputed["manufacturing_route"] = self.normalize_categorical(
            record.get("manufacturing_route"), self.MANUFACTURING_MAP, "unknown"
        )

        # 2. Impute Annealing Temperature
        temp = record.get("annealing_temperature")
        if temp is None or pd.isna(temp) or temp <= 0:
            if imputed["heat_treatment_category"] in ("annealing", "homogenization", "solution_treatment"):
                imputed["annealing_temperature"] = self.FAMILY_ANNEALING_DEFAULTS.get(family.lower(), 600.0)
            elif imputed["heat_treatment_category"] == "none":
                imputed["annealing_temperature"] = 0.0
            else:
                # Bounded standard fallback if state is unknown
                imputed["annealing_temperature"] = 0.0
        else:
            imputed["annealing_temperature"] = float(temp)

        # 3. Smart Category-Based Imputation Flow
        if imputed["cooling_method"] == "unknown":
            if imputed["heat_treatment_category"] == "annealing":
                imputed["cooling_method"] = "furnace_cool"
            elif imputed["heat_treatment_category"] in ("quenching_tempering", "solution_treatment"):
                imputed["cooling_method"] = "water_quench"
            elif imputed["heat_treatment_category"] == "none":
                imputed["cooling_method"] = "none"
            else:
                imputed["cooling_method"] = "air_cool"

        if imputed["manufacturing_route"] == "unknown":
            if family.lower() == "hea":
                imputed["manufacturing_route"] = "casting"
            elif family.lower() == "steel":
                imputed["manufacturing_route"] = "wrought"
            else:
                imputed["manufacturing_route"] = "casting"

        # 4. Compute Thermal Budget Category
        imputed["thermal_budget_category"] = self.compute_thermal_budget(imputed["annealing_temperature"])

        return imputed

    def compute_thermal_budget(self, temp: float) -> str:
        """
        Derives thermodynamic thermal budget category.
        - NONE: No thermal exposure
        - LOW: < 500 C (e.g. low-temp aging or tempering)
        - MEDIUM: 500 - 900 C (e.g. standard steel annealing)
        - HIGH: >= 900 C (e.g. homogenization / solution annealing of superalloys/HEAs)
        """
        if temp is None or pd.isna(temp) or temp <= 0:
            return "NONE"
        elif temp < 500.0:
            return "LOW"
        elif temp < 900.0:
            return "MEDIUM"
        else:
            return "HIGH"
