"""
ALLOY IQ — Niche HEA Pipeline (GitHub & NOMAD)
=============================================
Tier 3 Pipeline: Scripts to clone/download raw CSVs and Excel files. 
Handle extreme data sparsity (manage NaN values and missing columns).
"""

import httpx
import pandas as pd
import asyncio
from typing import List, Dict, Any

from backend.ingestion.logger import get_logger
from backend.ingestion.schema import make_empty_frame, standardize_columns

log = get_logger(__name__)

class HEARepositoryLoader:
    MPEA_DB_URL = "https://raw.githubusercontent.com/materialsintelligence/MPEA-dataset/main/data/MPEA_dataset.csv"
    
    async def fetch_mpea_database(self) -> pd.DataFrame:
        log.info("Fetching MPEA Database from GitHub...")
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(self.MPEA_DB_URL)
                resp.raise_for_status()
                
                # We would normally save this to a file or parse it with io.StringIO
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                
                # Extreme data sparsity handling for HEA
                df = df.dropna(how="all")
                
                df["src_name"] = "mpea_github"
                df["alloy_family"] = "hea"
                df["source_tier"] = "tier3"
                
                df = standardize_columns(df)
                log.info(f"MPEA Database loaded: {len(df)} rows")
                return df
            except Exception as e:
                log.error(f"Failed to fetch MPEA Database: {e}")
                return make_empty_frame()
                
    async def fetch_nomad_hea(self) -> pd.DataFrame:
        # Placeholder for NOMAD integration
        log.info("NOMAD fetch for HEA is mocked.")
        return make_empty_frame()

    async def fetch_all(self) -> pd.DataFrame:
        mpea_df = await self.fetch_mpea_database()
        nomad_df = await self.fetch_nomad_hea()
        frames = [f for f in [mpea_df, nomad_df] if not f.empty]
        if frames:
            return pd.concat(frames, ignore_index=True)
        return make_empty_frame()
