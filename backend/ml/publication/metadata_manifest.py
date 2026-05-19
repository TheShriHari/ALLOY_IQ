import os
import json
import time
from typing import Dict, Any, List
from loguru import logger

class MetadataManifestTracker:
    """
    Manages publication metadata manifests.
    Integrates system version hashes, dataset checksums, and Git commit identifiers.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_manifest(
        self,
        dataset_hash: str,
        feature_hash: str,
        model_hash: str,
        git_commit: str,
        experiment_ids: List[str],
        package_version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """Assembles the metadata manifest parameters."""
        manifest = {
            "dataset_hash": dataset_hash,
            "feature_hash": feature_hash,
            "model_hash": model_hash,
            "git_commit": git_commit,
            "experiment_ids": experiment_ids,
            "package_version": package_version,
            "timestamp": time.time(),
            "manifest_format_version": "1.0"
        }
        return manifest

    def save_manifest_to_file(self, manifest: Dict[str, Any], filename: str = "metadata_manifest.json") -> str:
        """Saves structured manifest registry metadata to file."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
            
        logger.info("Saved metadata manifest JSON to: {}", path)
        return path

    def verify_manifest_checksum(self, manifest_path: str, expected_dataset_hash: str) -> bool:
        """Confirms that the dataset signature recorded in the manifest matches actual parameters."""
        if not os.path.exists(manifest_path):
            return False
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            actual_hash = data.get("dataset_hash")
            return actual_hash == expected_dataset_hash
        except Exception as e:
            logger.error("Failed to verify manifest: {}", e)
            return False
