import os
import json
import time
import threading
from typing import Dict, List, Any, Optional

class ReleaseRegistry:
    """
    Cryptographic and metadata release registry.
    Saves and tracks model release IDs, dataset hashes, and commit signatures to a local ledger.
    """
    def __init__(self, ledger_path: str = "backend/release/release_ledger.json"):
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        
        # Load or init registry ledger
        self.releases: List[Dict[str, Any]] = self._load_ledger()

    def _load_ledger(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not os.path.exists(self.ledger_path):
                return []
            try:
                with open(self.ledger_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []

    def _save_ledger(self):
        try:
            with open(self.ledger_path, "w") as f:
                json.dump(self.releases, f, indent=2)
        except Exception as e:
            print(f"Error saving release registry ledger: {e}")

    def register_release(
        self,
        release_id: str,
        model_hash: str,
        dataset_hash: str,
        git_commit: str,
        benchmark_summary: Dict[str, float]
    ) -> Dict[str, Any]:
        """Saves a new release record to the registry ledger."""
        record = {
            "release_id": release_id,
            "model_hash": model_hash,
            "dataset_hash": dataset_hash,
            "git_commit": git_commit,
            "benchmark_summary": benchmark_summary,
            "created_at": time.time()
        }
        
        with self._lock:
            # Overwrite if exists, otherwise append
            self.releases = [r for r in self.releases if r["release_id"] != release_id]
            self.releases.append(record)
            self._save_ledger()
            
        return record

    def get_release(self, release_id: str) -> Optional[Dict[str, Any]]:
        """Queries a release record by ID."""
        with self._lock:
            for r in self.releases:
                if r["release_id"] == release_id:
                    return r
            return None

    def get_latest_release(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent successfully registered release."""
        with self._lock:
            if not self.releases:
                return None
            # Sort by created_at time
            sorted_releases = sorted(self.releases, key=lambda x: x["created_at"])
            return sorted_releases[-1]
