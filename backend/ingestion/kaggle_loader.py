"""
ALLOY IQ — Kaggle Direct Loader
=============================================
Tier 2 Pipeline: Direct data loaders that ingest pre-curated tabular datasets
directly into pandas DataFrames.
"""

import os
from pathlib import Path
import pandas as pd
from backend.ingestion.logger import get_logger
from backend.ingestion.schema import make_empty_frame, standardize_columns

log = get_logger(__name__)

KAGGLE_DIR = Path(os.getenv("KAGGLE_DIR", "backend/data/kaggle"))

class KaggleLoader:
    def __init__(self):
        KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        
    def load_datasets(self) -> pd.DataFrame:
        """
        Loads all CSVs from the Kaggle directory. In production, this would use
        the kaggle API to download the datasets first.
        """
        log.info(f"KaggleLoader: checking for datasets in {KAGGLE_DIR}")
        
        frames = []
        for file_path in KAGGLE_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                df["src_name"] = "kaggle"
                df["src_id"] = file_path.name
                df["source_tier"] = "tier2"
                df = standardize_columns(df)
                frames.append(df)
                log.info(f"Loaded Kaggle dataset: {file_path.name} ({len(df)} rows)")
            except Exception as e:
                log.error(f"Failed to load Kaggle dataset {file_path.name}: {e}")
                
        if not frames:
            return make_empty_frame()
            
        return pd.concat(frames, ignore_index=True)
