"""
ALLOY IQ — OQMD Asynchronous REST API Client
=============================================
Queries the OQMD database for thermodynamic descriptors:
  - Formation energy
  - Band gaps
  - Atomic volume

Uses httpx and asyncio.Semaphore to respect server limits.
"""

import asyncio
import random
from typing import Any, Dict, List, Optional
import httpx
import pandas as pd

from backend.ingestion.logger import get_logger
from backend.ingestion.schema import make_empty_frame, standardize_columns

log = get_logger(__name__)

OQMD_BASE_URL = "http://oqmd.org/oqmdapi"
MAX_CONCURRENT = 5
MAX_RETRIES = 3

class OqmdClient:
    async def fetch_async(self, limit: int = 1000) -> pd.DataFrame:
        log.info(f"OQMD: fetching up to {limit} entries …")
        
        async def fetch_page(client: httpx.AsyncClient, offset: int) -> List[Dict[str, Any]]:
            url = f"{OQMD_BASE_URL}/formationenergy"
            params = {"limit": 100, "offset": offset}
            
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.get(url, params=params, timeout=15.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("data", [])
                    elif resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                    else:
                        break
                except httpx.RequestError:
                    await asyncio.sleep(2 ** attempt)
            return []

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async def bounded_page(client: httpx.AsyncClient, offset: int) -> List[Dict[str, Any]]:
            async with semaphore:
                return await fetch_page(client, offset)

        async with httpx.AsyncClient() as client:
            tasks = [bounded_page(client, i) for i in range(0, limit, 100)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        records = []
        for res in results:
            if isinstance(res, list):
                for item in res:
                    records.append({
                        "src_name": "oqmd",
                        "src_id": str(item.get("entry_id")),
                        "formation_enthalpy_eV_atom": item.get("delta_e"),
                        "volume_A3_atom": item.get("volume"),
                        "band_gap": item.get("band_gap"),
                        "alloy_family": "unknown",
                        "source_tier": "tier1"
                    })
                    
        df = pd.DataFrame(records)
        if df.empty:
            return make_empty_frame()
            
        df = standardize_columns(df)
        log.info(f"OQMD: DataFrame shape = {df.shape}")
        return df
