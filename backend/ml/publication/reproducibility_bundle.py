import os
import json
import sys
import platform
from typing import Dict, Any, List
from loguru import logger

class ReproducibilityBundler:
    """
    Assembles model registries snapshots, configurations, and environment lock details.
    Guarantees that third-party researchers can replicate system executions exactly.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_environment_report(self) -> Dict[str, Any]:
        """Gathers system parameters (python, OS, architecture) into report."""
        report = {
            "python_version": sys.version,
            "os_platform": platform.platform(),
            "os_system": platform.system(),
            "os_release": platform.release(),
            "machine_architecture": platform.machine(),
            "cpu_count": os.cpu_count() or 1,
            "environment_resolved_at": time_resolved_str()
        }
        return report

    def assemble_reproducibility_bundle(
        self,
        configs: Dict[str, Any],
        registry_snapshot: List[Dict[str, Any]],
        benchmarks: Dict[str, Any],
        lock_file_lines: List[str]
    ) -> Dict[str, Any]:
        """Packages all reproducibility parameters into a unified structured manifest dictionary."""
        env_details = self.generate_environment_report()
        bundle = {
            "environment_details": env_details,
            "training_configurations": configs,
            "model_registry_snapshot": registry_snapshot,
            "benchmarks_results": benchmarks,
            "dependency_lock_file": lock_file_lines
        }
        
        # Save bundle file
        bundle_path = os.path.join(self.output_dir, "reproducibility_bundle.json")
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)
            
        logger.info("Saved reproducibility bundle JSON to: {}", bundle_path)
        return bundle

def time_resolved_str() -> str:
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")
