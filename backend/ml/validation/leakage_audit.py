import pandas as pd
from typing import Dict, Any, Set
from loguru import logger

class DataLeakageAuditor:
    """
    Materials informatics Data Leakage auditing suite.
    Identifies composition duplicates, paper DOI leaks, research group overlaps,
    and family spillover occurrences between train and validation datasets.
    """
    
    @staticmethod
    def _extract_composition_set(df: pd.DataFrame) -> Set[str]:
        """Converts composition dict values into a sorted, normalized key representation."""
        comp_keys = set()
        comp_col = next((col for col in df.columns if col.lower() == "composition"), None)
        if not comp_col:
            return comp_keys
            
        for val in df[comp_col]:
            if isinstance(val, dict):
                # Sort elements to get an unambiguous canonical string representation
                canonical = ",".join(f"{k}:{float(v):.4f}" for k, v in sorted(val.items()) if float(v) > 0.0)
                if canonical:
                    comp_keys.add(canonical)
        return comp_keys

    @staticmethod
    def _extract_composition_dicts(df: pd.DataFrame) -> list:
        comp_col = next((col for col in df.columns if col.lower() == "composition"), None)
        if not comp_col:
            return []
        dicts = []
        for val in df[comp_col]:
            if isinstance(val, dict):
                dicts.append({k: float(v) for k, v in val.items() if float(v) > 0.0})
        return dicts

    def audit_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs complete cryptographic and bibliographic checks between train and test data splits.
        Returns leakage status flags, count tallies, and descriptive alerts.
        """
        logger.info("Executing split-level data leakage audit over train/test segments.")
        
        # 1. Composition Leakage Audit
        train_comps = self._extract_composition_set(train_df)
        test_comps = self._extract_composition_set(test_df)
        leaked_comps = train_comps.intersection(test_comps)
        has_comp_leak = len(leaked_comps) > 0

        # 2. DOI Leakage Audit
        doi_col = next((col for col in train_df.columns if col.lower() in ("paper_doi", "doi")), None)
        has_doi_leak = False
        leaked_dois = set()
        if doi_col and doi_col in test_df.columns:
            train_dois = {str(d).strip().lower() for d in train_df[doi_col].dropna()}
            test_dois = {str(d).strip().lower() for d in test_df[doi_col].dropna()}
            leaked_dois = train_dois.intersection(test_dois)
            has_doi_leak = len(leaked_dois) > 0

        # 3. Research Group Leakage Audit
        group_col = next((col for col in train_df.columns if col.lower() in ("research_group_id", "group")), None)
        has_group_leak = False
        leaked_groups = set()
        if group_col and group_col in test_df.columns:
            train_groups = {str(g).strip().lower() for g in train_df[group_col].dropna()}
            test_groups = {str(g).strip().lower() for g in test_df[group_col].dropna()}
            leaked_groups = train_groups.intersection(test_groups)
            has_group_leak = len(leaked_groups) > 0

        # 4. Family Spillover Audit
        family_col = next((col for col in train_df.columns if col.lower() == "alloy_family"), None)
        spillover_families = set()
        if family_col and family_col in test_df.columns:
            train_families = set(train_df[family_col].astype(str).str.strip().str.lower().dropna())
            test_families = set(test_df[family_col].astype(str).str.strip().str.lower().dropna())
            # Spillover tracks families present in both splits (normal if family is large, 
            # but logged here for transparency)
            spillover_families = train_families.intersection(test_families)

        # 5. Overlap Proximity Risk Audit (Identifies compositions differing by less than 2wt% L1)
        train_dicts = self._extract_composition_dicts(train_df)
        test_dicts = self._extract_composition_dicts(test_df)
        overlap_risk_count = 0
        for test_c in test_dicts:
            min_dist = 1.0
            for train_c in train_dicts:
                elements = set(test_c.keys()).union(set(train_c.keys()))
                l1_dist = sum(abs(test_c.get(el, 0.0) - train_c.get(el, 0.0)) for el in elements)
                if l1_dist < min_dist:
                    min_dist = l1_dist
            if min_dist < 0.02:
                overlap_risk_count += 1
        has_overlap_risk = overlap_risk_count > 0

        audit_report = {
            "has_leakage": bool(has_comp_leak or has_doi_leak or has_group_leak),
            "composition_leakage": {
                "detected": has_comp_leak,
                "overlap_count": len(leaked_comps),
                "leaked_examples": list(leaked_comps)[:5]
            },
            "doi_leakage": {
                "detected": has_doi_leak,
                "overlap_count": len(leaked_dois),
                "leaked_dois": list(leaked_dois)
            },
            "research_group_leakage": {
                "detected": has_group_leak,
                "overlap_count": len(leaked_groups),
                "leaked_groups": list(leaked_groups)
            },
            "family_spillover": {
                "overlapping_families": list(spillover_families),
                "overlap_count": len(spillover_families)
            },
            "train_test_overlap_risk": {
                "detected": has_overlap_risk,
                "high_similarity_count": overlap_risk_count
            }
        }
        
        if audit_report["has_leakage"] or has_overlap_risk:
            logger.warning("Data leakage or overlap risk detected! Composition overlaps: {}, Overlap risk count: {}", 
                           len(leaked_comps), overlap_risk_count)
        else:
            logger.info("Leakage audit complete. Splits are mutually independent.")
            
        return audit_report
