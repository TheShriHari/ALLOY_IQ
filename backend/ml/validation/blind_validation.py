import pandas as pd
from typing import Tuple, List
from loguru import logger

class BlindValidator:
    """
    Materials Informatics Blind Validation suite.
    Enforces strict holdout partitions on unseen domains (alloy families, DOI cohorts,
    or manufacturing processing routes) to test extrapolative generalization.
    """
    
    @staticmethod
    def holdout_by_family(df: pd.DataFrame, family: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset by holding out a specific alloy family (e.g. 'hea' or 'steel')
        exclusively for blind testing, training the models on all remaining families.
        """
        logger.info("Generating blind validation partition by holding out family: {}", family)
        family_clean = str(family).strip().lower()
        
        # Identify family column in dataset
        family_col = next((col for col in df.columns if col.lower() == "alloy_family"), None)
        if not family_col:
            raise ValueError("Dataset does not contain 'alloy_family' column!")
            
        test_mask = df[family_col].astype(str).str.strip().str.lower() == family_clean
        test_df = df[test_mask]
        train_df = df[~test_mask]
        
        logger.info("Partition complete. Train samples: {}, Held-out Test samples: {}", len(train_df), len(test_df))
        return train_df, test_df

    @staticmethod
    def holdout_by_doi_groups(df: pd.DataFrame, test_dois: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset by holding out a specific list of paper DOIs for testing,
        ensuring bibliographic independence.
        """
        logger.info("Generating blind validation partition by holding out {} DOIs", len(test_dois))
        dois_clean = {str(d).strip().lower() for d in test_dois}
        
        doi_col = next((col for col in df.columns if col.lower() in ("paper_doi", "doi")), None)
        if not doi_col:
            raise ValueError("Dataset does not contain paper DOI columns!")
            
        test_mask = df[doi_col].astype(str).str.strip().str.lower().isin(dois_clean)
        test_df = df[test_mask]
        train_df = df[~test_mask]
        
        logger.info("Partition complete. Train samples: {}, Held-out Test samples: {}", len(train_df), len(test_df))
        return train_df, test_df

    @staticmethod
    def holdout_by_processing_route(df: pd.DataFrame, route: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset by holding out an entire manufacturing route (e.g. 'wrought' or 'cast')
        for the blind validation cohort.
        """
        logger.info("Generating blind validation partition by holding out processing route: {}", route)
        route_clean = str(route).strip().lower()
        
        route_col = next((col for col in df.columns if col.lower() in ("manufacturing_route", "route")), None)
        if not route_col:
            raise ValueError("Dataset does not contain 'manufacturing_route' column!")
            
        test_mask = df[route_col].astype(str).str.strip().str.lower() == route_clean
        test_df = df[test_mask]
        train_df = df[~test_mask]
        
        logger.info("Partition complete. Train samples: {}, Held-out Test samples: {}", len(train_df), len(test_df))
        return train_df, test_df
