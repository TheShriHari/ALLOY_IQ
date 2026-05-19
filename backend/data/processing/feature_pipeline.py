import pandas as pd
import numpy as np
from typing import Dict, Any, List
from backend.data.features import FeatureEngineer, ELEMENT_PROPS
from backend.data.processing.processing_features import ProcessingFeatureEngineer

class ProcessingAwareFeaturePipeline:
    """
    Unified, versioned Feature Pipeline.
    Combines compositional Matminer physics descriptors with standardized processing features.
    Maintains full backward compatibility for legacy inputs by generating defaults.
    """
    
    FEATURE_VERSION = "2.0.0"

    def __init__(self, alloy_family: str = "steel", version: str = "v2"):
        self.alloy_family = alloy_family
        self.version = version  # "v1" (legacy composition-only), "v2" (processing-aware)
        self.comp_engineer = FeatureEngineer(alloy_family)
        self.proc_engineer = ProcessingFeatureEngineer()

        # Predefined categories for model alignment consistency
        self.ht_categories = ["annealing", "homogenization", "solution_treatment", "quenching_tempering", "solution_aging", "as_cast", "none", "unknown"]
        self.cooling_methods = ["water_quench", "air_cool", "furnace_cool", "oil_quench", "none", "unknown"]
        self.manufacturing_routes = ["casting", "wrought", "powder_metallurgy", "additive_manufacturing", "unknown"]
        self.thermal_budgets = ["NONE", "LOW", "MEDIUM", "HIGH"]

    def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms a single raw prediction input record (composition and optional processing metrics)
        into a flat model-ready dictionary.
        """
        # 1. Impute/Normalize processing features
        proc_data = self.proc_engineer.impute_missing_processing(record, family=self.alloy_family)
        
        # 2. Extract and run Matminer composition engineer
        comp_dict = record.get("composition") or {}
        comp_df = pd.DataFrame([comp_dict])
        
        # Fill missing elements to align with Matminer expectation
        for el in ELEMENT_PROPS.keys():
            if el not in comp_df.columns:
                comp_df[el] = 0.0
                
        comp_features_df = self.comp_engineer.transform(comp_df)
        comp_features = comp_features_df.iloc[0].to_dict()

        # 3. Form base flat record dictionary
        result = comp_features.copy()
        result["feature_pipeline_version"] = self.FEATURE_VERSION
        
        # If legacy version, we discard processing metrics
        if self.version == "v1":
            return result
            
        # 4. Integrate processing metrics for v2 (Processing-Aware)
        result["feat_proc_annealing_temp"] = proc_data["annealing_temperature"]
        
        # One-hot categorical indicators for robust ML model training
        for cat in self.ht_categories:
            result[f"feat_proc_ht_{cat}"] = 1.0 if proc_data["heat_treatment_category"] == cat else 0.0
            
        for cool in self.cooling_methods:
            result[f"feat_proc_cool_{cool}"] = 1.0 if proc_data["cooling_method"] == cool else 0.0
            
        for route in self.manufacturing_routes:
            result[f"feat_proc_route_{route}"] = 1.0 if proc_data["manufacturing_route"] == route else 0.0
            
        for budget in self.thermal_budgets:
            result[f"feat_proc_budget_{budget}"] = 1.0 if proc_data["thermal_budget_category"] == budget else 0.0
            
        return result

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch transformation pipeline for large training datasets.
        Creates one-hot and continuous processing descriptors from records.
        """
        records = []
        for _, row in df.iterrows():
            # Build normalized record mapping
            rec = {
                "composition": row.get("composition") or {},
                "heat_treatment_category": row.get("heat_treatment_category"),
                "annealing_temperature": row.get("annealing_temperature"),
                "cooling_method": row.get("cooling_method"),
                "manufacturing_route": row.get("manufacturing_route"),
                "paper_doi": row.get("paper_doi"),
                "research_group_id": row.get("research_group_id")
            }
            # Handle direct element column format if input is raw matrix
            for el in ELEMENT_PROPS.keys():
                if el in df.columns:
                    rec["composition"][el] = float(row[el])
                    
            transformed = self.transform_record(rec)
            records.append(transformed)
            
        return pd.DataFrame(records, index=df.index)
