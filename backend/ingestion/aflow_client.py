"""
ALLOY IQ — AFLOW Asynchronous REST API Client
==============================================
Queries the AFLOW database for thermodynamic properties:
  - Formation enthalpy (enthalpy_formation_atom)
  - Volume per atom (volume_atom)
  - Composition (species + composition arrays)
  - Alloy prototype + space group

Fault-tolerance design:
  - Fully async using httpx.AsyncClient
  - Exponential back-off with jitter on 429 / 5xx responses
  - Per-request timeout; individual entry failures never stop the batch
  - Progress logged per page; summary logged at end

API docs: http://aflow.org/API/aflowlib.org/

Usage (sync wrapper provided for non-async callers):
    from backend.ingestion.aflow_client import AflowClient
    client = AflowClient()
    df = client.fetch_sync(keywords=["enthalpy_formation_atom", "volume_atom"],
                           nmax=2000)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

from backend.ingestion.logger import get_logger
from backend.ingestion.schema import make_empty_frame, standardize_columns

load_dotenv()
log = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
AFLOW_BASE_URL: str = os.getenv(
    "AFLOW_BASE_URL", "http://aflow.org/API/aflowlib.org"
)
MAX_CONCURRENT: int = int(os.getenv("AFLOW_MAX_CONCURRENT", "5"))
BACKOFF_BASE: float = float(os.getenv("AFLOW_BACKOFF_BASE", "1.0"))
MAX_RETRIES: int = int(os.getenv("AFLOW_MAX_RETRIES", "5"))

# AFLOW returns JSON; typical page size
PAGE_SIZE = 200


async def _fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: Dict[str, Any],
    retries: int = MAX_RETRIES,
) -> Optional[Any]:
    """
    Perform a GET request with exponential back-off + jitter.
    Returns parsed JSON or None on terminal failure.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = await client.get(url, params=params, timeout=30.0)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                # Rate limited — always back off
                wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                log.warning(
                    f"AFLOW rate-limited (attempt {attempt}/{retries}). "
                    f"Retrying in {wait:.1f}s … URL: {url}"
                )
                await asyncio.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                log.warning(
                    f"AFLOW server error {resp.status_code} (attempt {attempt}/{retries}). "
                    f"Retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)
                continue

            log.error(f"AFLOW HTTP {resp.status_code} for {url} — skipping.")
            return None

        except httpx.TimeoutException:
            wait = BACKOFF_BASE * (2 ** attempt)
            log.warning(f"AFLOW timeout (attempt {attempt}/{retries}). Retrying in {wait:.1f}s")
            await asyncio.sleep(wait)

        except httpx.RequestError as exc:
            log.error(f"AFLOW request error: {exc} — URL: {url}")
            return None

    log.error(f"AFLOW: all {retries} retries exhausted for {url}")
    return None


def _parse_aflow_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse a single AFLOW REST JSON entry into a flat dict aligned
    with the canonical schema. Returns None if entry is malformed.
    """
    try:
        record: Dict[str, Any] = {
            "src_name": "aflow",
            "src_id": entry.get("auid", ""),
            "src_url": entry.get("aurl", ""),
        }

        # ── Thermodynamic properties ──────────────────────────────────────
        record["formation_enthalpy_eV_atom"] = entry.get("enthalpy_formation_atom")
        record["volume_A3_atom"] = entry.get("volume_atom")
        record["density_g_cm3"] = entry.get("density")

        # ── Composition ───────────────────────────────────────────────────
        # AFLOW gives species as "Fe,Cr,Ni" and composition as "60,20,20"
        species_raw = entry.get("species", "")
        comp_raw = entry.get("composition", "")

        if species_raw and comp_raw:
            try:
                species = [s.strip() for s in species_raw.split(",")]
                comp_vals = [float(c) for c in str(comp_raw).split(",")]

                if len(species) == len(comp_vals) and sum(comp_vals) > 0:
                    total = sum(comp_vals)
                    for el, val in zip(species, comp_vals):
                        record[f"{el}_wt"] = (val / total) * 100.0  # wt% approx
            except ValueError as e:
                log.debug(f"Composition parse error for {record['src_id']}: {e}")

        # ── Prototype / alloy classification ──────────────────────────────
        prototype = entry.get("prototype", "")
        record["notes"] = f"prototype={prototype}"

        # Best-effort alloy family classification from elements present
        if species_raw:
            els = set(species_raw.split(","))
            if "Fe" in els:
                record["alloy_family"] = "steel"
            elif "Al" in els and len(els) <= 5:
                record["alloy_family"] = "aluminum"
            else:
                record["alloy_family"] = "hea"
        else:
            record["alloy_family"] = "hea"

        return record

    except Exception as exc:
        log.warning(f"Failed to parse AFLOW entry: {exc}")
        return None


async def _fetch_page(
    client: httpx.AsyncClient,
    keywords: List[str],
    pstart: int,
    psize: int = PAGE_SIZE,
) -> List[Dict[str, Any]]:
    """Fetch a single page from the AFLOW REST API."""
    params = {
        "format": "json",
        "pstart": pstart,
        "psize": psize,
        "keywords": ",".join(keywords),
    }
    url = f"{AFLOW_BASE_URL}/?aflowlib_entries"
    data = await _fetch_with_retry(client, url, params)
    if data is None or not isinstance(data, list):
        return []
    return data


async def _fetch_all(
    keywords: List[str],
    nmax: int,
) -> List[Dict[str, Any]]:
    """
    Paginate through AFLOW results using bounded concurrency.
    Respects MAX_CONCURRENT simultaneous requests.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_entries: List[Dict[str, Any]] = []

    n_pages = math.ceil(nmax / PAGE_SIZE)
    log.info(f"AFLOW: fetching up to {nmax} entries across {n_pages} pages …")

    async def bounded_page(pstart: int) -> List[Dict[str, Any]]:
        async with semaphore:
            return await _fetch_page(client_ref[0], keywords, pstart)

    limits = httpx.Limits(max_connections=MAX_CONCURRENT + 2)
    async with httpx.AsyncClient(limits=limits) as client:
        client_ref = [client]
        tasks = [
            bounded_page(pstart=i * PAGE_SIZE)
            for i in range(n_pages)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            log.error(f"AFLOW page {i} raised exception: {result}")
        elif isinstance(result, list):
            all_entries.extend(result)
            log.debug(f"  → Page {i}: {len(result)} entries")

    log.info(f"AFLOW: raw entries collected = {len(all_entries)}")
    return all_entries[:nmax]


class AflowClient:
    """
    High-level client for the AFLOW REST API.
    Provides both async and synchronous (blocking) interfaces.
    """

    DEFAULT_KEYWORDS: List[str] = [
        "auid",
        "aurl",
        "species",
        "composition",
        "prototype",
        "enthalpy_formation_atom",
        "volume_atom",
        "density",
    ]

    def fetch_sync(
        self,
        keywords: Optional[List[str]] = None,
        nmax: int = 2000,
    ) -> pd.DataFrame:
        """
        Synchronous wrapper around the async fetch.
        Blocks until all pages are fetched.

        Parameters
        ----------
        keywords : list[str], optional
            AFLOW property keywords to request. Defaults to thermodynamic set.
        nmax : int
            Maximum number of entries to retrieve.

        Returns
        -------
        pd.DataFrame with canonical schema or empty frame on failure.
        """
        kw = keywords or self.DEFAULT_KEYWORDS
        try:
            entries = asyncio.run(_fetch_all(kw, nmax))
        except Exception as exc:
            log.error(f"AFLOW async fetch failed: {exc}", exc_info=True)
            return make_empty_frame()

        if not entries:
            log.warning("AFLOW returned 0 entries.")
            return make_empty_frame()

        records: List[Dict[str, Any]] = []
        skipped = 0
        for entry in entries:
            parsed = _parse_aflow_entry(entry)
            if parsed:
                records.append(parsed)
            else:
                skipped += 1

        if skipped:
            log.warning(f"AFLOW: skipped {skipped} malformed entries")

        df = pd.DataFrame(records)
        df = standardize_columns(df)
        log.info(f"AFLOW: final DataFrame shape = {df.shape}")
        return df

    async def fetch_async(
        self,
        keywords: Optional[List[str]] = None,
        nmax: int = 2000,
    ) -> pd.DataFrame:
        """
        Fully async version for use in async contexts.
        """
        kw = keywords or self.DEFAULT_KEYWORDS
        try:
            entries = await _fetch_all(kw, nmax)
        except Exception as exc:
            log.error(f"AFLOW async fetch failed: {exc}", exc_info=True)
            return make_empty_frame()

        records = [r for e in entries if (r := _parse_aflow_entry(e)) is not None]
        return pd.DataFrame(records)
