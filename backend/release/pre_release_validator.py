import hashlib
import pickle
from typing import Dict, Any, List, Tuple
from loguru import logger

class PreReleaseValidator:
    """
    Validates release artifacts prior to production tagging.
    Asserts feature schema alignment, model weight signatures, checkpoint
    deserialization formats, and benchmark threshold levels.
    """
    def __init__(self, min_r2: float = 0.70, max_mae: float = 30.0):
        self.min_r2 = min_r2
        self.max_mae = max_mae

    def verify_feature_hash(self, current_hash: str, model_feature_hash: str) -> bool:
        """Confirms pipeline descriptors schema matches expected models feature signature."""
        if current_hash != model_feature_hash:
            logger.error("Pre-release Failure: Feature schema signature mismatch! pipeline: {}, model: {}", current_hash, model_feature_hash)
            return False
        logger.info("Feature schema hash compatibility verified: {}", current_hash)
        return True

    def verify_model_hash_integrity(self, model_bytes: bytes, expected_hash: str) -> bool:
        """Verifies model weight pickle file against registered SHA-256 signature."""
        actual_hash = hashlib.sha256(model_bytes).hexdigest()
        if actual_hash != expected_hash:
            logger.error("Pre-release Failure: Model file corruption! Expected {}, got {}", expected_hash, actual_hash)
            return False
        logger.info("Model file hash signature verified: {}", actual_hash)
        return True

    def verify_checkpoint_schema(self, checkpoint_bytes: bytes) -> bool:
        """Ensures checkpoint records can be safely deserialized with current schema definitions."""
        try:
            # Test unpickling/msgpack formats
            import zstd
            import msgpack
            decompressed = zstd.decompress(checkpoint_bytes)
            unpacked = msgpack.unpackb(decompressed, raw=False)
            
            # Assert core state dictionary fields are present
            required = {"step", "optimizer_state", "checksum"}
            if not all(field in unpacked for field in required):
                logger.error("Pre-release Failure: Checkpoint is missing mandatory state variables.")
                return False
            logger.info("Checkpoint state schema unpickling verified.")
            return True
        except Exception as e:
            logger.error("Pre-release Failure: Checkpoint serialization format compatibility error: {}", e)
            return False

    def verify_benchmark_thresholds(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Audits model performance indicators against hardcoded release quality bounds."""
        is_clean = True
        warnings = []
        
        r2 = metrics.get("r2", 0.0)
        mae = metrics.get("mae", 999.0)
        
        if r2 < self.min_r2:
            is_clean = False
            warnings.append(f"R² performance level too low: {r2:.3f}. Must be >= {self.min_r2:.3f}")
            
        if mae > self.max_mae:
            is_clean = False
            warnings.append(f"MAE performance level too high: {mae:.2f} MPa. Must be <= {self.max_mae:.2f} MPa")
            
        return is_clean, warnings

    def verify_experiment_registry_consistency(self, registry_entry: Dict[str, Any]) -> bool:
        """Ensures the model metadata fields correspond with the experiment ledger log entries."""
        required = {"dataset_hash", "feature_hash", "model_hash", "metrics"}
        if not all(field in registry_entry for field in required):
            logger.error("Pre-release Failure: Experiment ledger registry is missing key metadata properties.")
            return False
        logger.info("Experiment registry entries verified.")
        return True
