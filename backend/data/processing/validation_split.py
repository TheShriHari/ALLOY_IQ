import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from loguru import logger

try:
    from sklearn.model_selection import GroupKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class MaterialsValidationSplitter:
    """
    Manages leakage-preventing validation splits and out-of-distribution (OOD) risk flagging.
    Uses GroupKFold based on Paper DOI or Research Group to prevent identical experiment/paper spillover.
    """

    def generate_groups(self, df: pd.DataFrame) -> pd.Series:
        """
        Creates group identifiers based on research_group_id or paper_doi.
        Falls back to unique index strings for independent single-data entries.
        """
        group_col = []
        for idx, row in df.iterrows():
            group_id = row.get("research_group_id") or row.get("paper_doi")
            if not group_id or pd.isna(group_id):
                group_id = f"isolated_record_{idx}"
            group_col.append(str(group_id).strip().lower())
        return pd.Series(group_col, index=df.index)

    def split(self, df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits the dataset into K folds ensuring that records from the same DOI/Research Group
        never overlap between train and validation folds.
        """
        logger.info("Executing GroupKFold validation split (K={}) with leakage protection.", n_splits)
        
        # 1. Generate split groups
        groups = self.generate_groups(df)
        group_array = groups.values
        
        splits = []
        
        if HAS_SKLEARN:
            gkf = GroupKFold(n_splits=n_splits)
            # Scikit-learn expects dummy target arrays to extract indices
            dummy_y = np.zeros(len(df))
            for train_idx, val_idx in gkf.split(df, dummy_y, groups=group_array):
                splits.append((df.iloc[train_idx], df.iloc[val_idx]))
        else:
            # Fallback custom GroupKFold algorithm if sklearn is missing
            logger.warning("scikit-learn not found. Executing manual deterministic group split.")
            unique_groups = list(set(group_array))
            np.random.seed(42)
            np.random.shuffle(unique_groups)
            
            fold_groups = np.array_split(unique_groups, n_splits)
            for i in range(n_splits):
                val_g = set(fold_groups[i])
                val_mask = df.index.map(lambda idx: group_array[idx] in val_g)
                
                val_df = df[val_mask]
                train_df = df[~val_mask]
                splits.append((train_df, val_df))

        # 2. Check and log leakage sanity check
        for fold, (train, val) in enumerate(splits):
            train_groups = set(self.generate_groups(train))
            val_groups = set(self.generate_groups(val))
            overlapping = train_groups.intersection(val_groups)
            if overlapping:
                logger.error("Leakage Alert in Fold {}! Overlapping papers: {}", fold, overlapping)
                raise ValueError("Data leakage detected between training and validation splits!")
                
        logger.info("Successfully completed K-fold splits without group leakage.")
        return splits

    def evaluate_risk_flags(self, train_df: pd.DataFrame, test_record: Dict[str, Any]) -> List[str]:
        """
        Fuzzy risk classification engine. Computes OOD distance metrics of a single
        incoming prediction request compared to the historical training dataset.
        Flags: LOW_CONFIDENCE, OOD, SPARSE_FAMILY, MISSING_PROCESSING
        """
        flags = []

        # 1. MISSING_PROCESSING Check
        annealing_temp = test_record.get("annealing_temperature")
        heat_treat_cat = str(test_record.get("heat_treatment_category") or "").lower()
        
        if annealing_temp is None or pd.isna(annealing_temp) or annealing_temp <= 0:
            if heat_treat_cat not in ("none", "as_cast"):
                flags.append("MISSING_PROCESSING")

        # 2. SPARSE_FAMILY Check
        family = str(test_record.get("alloy_family") or "unknown").strip().lower()
        if "alloy_family" in train_df.columns:
            family_count = sum(train_df["alloy_family"].str.strip().str.lower() == family)
            if family_count < 5:
                flags.append("SPARSE_FAMILY")
        else:
            # If no historical reference
            flags.append("SPARSE_FAMILY")

        # 3. OOD (Out of Distribution Processing) Check
        if "annealing_temperature" in train_df.columns:
            train_temps = train_df["annealing_temperature"].dropna()
            if not train_temps.empty and annealing_temp is not None:
                min_t, max_t = train_temps.min(), train_temps.max()
                # Flag OOD if 100 degrees Celsius outside of range bounds
                if annealing_temp > (max_t + 100.0) or annealing_temp < (min_t - 5.0):
                    flags.append("OOD")

        # 4. LOW_CONFIDENCE (Composition-space distance) Check
        test_comp = test_record.get("composition") or {}
        if test_comp and "composition" in train_df.columns:
            # Calculate average L1 element-distance to find local density
            min_l1_dist = float("inf")
            for _, row in train_df.iterrows():
                row_comp = row.get("composition") or {}
                # Calculate L1 distance between element sets
                all_elements = set(test_comp.keys()).union(set(row_comp.keys()))
                l1_dist = 0.0
                for el in all_elements:
                    test_frac = float(test_comp.get(el, 0.0))
                    row_frac = float(row_comp.get(el, 0.0))
                    l1_dist += abs(test_frac - row_frac)
                if l1_dist < min_l1_dist:
                    min_l1_dist = l1_dist
            
            # If closest element mapping deviates by > 15wt%, mark as Low Confidence
            if min_l1_dist > 0.15:
                flags.append("LOW_CONFIDENCE")
                
        return flags
