"""
ALLOY IQ — Centralized Logging Configuration
=============================================
All ingestion modules import `get_logger()` from this module.
Guarantees that every failed API call, dropped PDF table, and flagged
outlier is captured in logs/ingestion_errors.log with structured context.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ── Master ingestion errors file ─────────────────────────────────────────────
_ERROR_HANDLER = logging.handlers.RotatingFileHandler(
    filename=LOG_DIR / "global_ingestion_errors.log",
    maxBytes=10 * 1024 * 1024,   # 10 MB per file
    backupCount=5,
    encoding="utf-8",
)
_ERROR_HANDLER.setLevel(logging.WARNING)
_ERROR_HANDLER.setFormatter(_FORMATTER)

# ── Full pipeline log (DEBUG+) ───────────────────────────────────────────────
_FULL_HANDLER = logging.handlers.RotatingFileHandler(
    filename=LOG_DIR / "ingestion_full.log",
    maxBytes=20 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_FULL_HANDLER.setLevel(logging.DEBUG)
_FULL_HANDLER.setFormatter(_FORMATTER)

# ── Console handler ──────────────────────────────────────────────────────────
_CONSOLE_HANDLER = logging.StreamHandler()
_CONSOLE_HANDLER.setLevel(LOG_LEVEL)
_CONSOLE_HANDLER.setFormatter(_FORMATTER)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger scoped to `name` with:
      - Console output at configured LOG_LEVEL
      - Full log captured to logs/ingestion_full.log
      - Warnings/errors also captured to logs/ingestion_errors.log

    Usage:
        from backend.ingestion.logger import get_logger
        log = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_CONSOLE_HANDLER)
        logger.addHandler(_FULL_HANDLER)
        logger.addHandler(_ERROR_HANDLER)
        logger.propagate = False
    return logger
