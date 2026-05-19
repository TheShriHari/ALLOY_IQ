import redis
import json
import hashlib
import os
from loguru import logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = 60 * 60 * 24 * 7   # 7 days

try:
    r = redis.from_url(REDIS_URL, socket_timeout=1.0, socket_connect_timeout=1.0)
    # Ping to check if connection is active
    r.ping()
    logger.info("Connected to Redis successfully for caching.")
    redis_available = True
except Exception as e:
    logger.warning("Redis is not available. Caching will be bypassed. Error: {}", e)
    redis_available = False

def _cache_key(composition: dict) -> str:
    """Generate a stable cache key from composition dict."""
    sorted_comp = sorted((k, round(v, 6)) for k, v in composition.items() if v > 1e-6)
    return "magpie:" + hashlib.sha256(json.dumps(sorted_comp).encode()).hexdigest()[:16]

def get_cached_features(composition: dict) -> dict | None:
    if not redis_available:
        return None
    try:
        key = _cache_key(composition)
        cached = r.get(key)
        if cached:
            logger.debug("Redis cache HIT for composition: {}", sorted(composition.keys()))
            return json.loads(cached)
    except Exception as e:
        logger.error("Redis get failed: {}", e)
    return None

def set_cached_features(composition: dict, features: dict) -> None:
    if not redis_available:
        return
    try:
        key = _cache_key(composition)
        r.setex(key, CACHE_TTL, json.dumps(features))
        logger.debug("Redis cache SET for key: {}", key)
    except Exception as e:
        logger.error("Redis set failed: {}", e)
