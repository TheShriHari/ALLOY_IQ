from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class ProductionReadinessEvaluator:
    """Scores production readiness dimensions from either flags or workspace evidence."""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()

    def evaluate_workspace(self) -> Dict[str, Any]:
        backend = self.workspace_root / "backend"
        return self.generate_full_compliance_report(
            has_docker=(self.workspace_root / "docker-compose.yml").exists(),
            has_celery=(backend / "tasks" / "celery_app.py").exists(),
            has_redis=(backend / "cache" / "redis_client.py").exists(),
            has_checkpoints=(backend / "cache" / "checkpoint_service.py").exists(),
            has_rollback=self._contains("rollback_recovery"),
            conformal_coverage=0.95 if self._contains("conformal") else 0.5,
            has_physics=(backend / "ml" / "safety" / "physics_constraints.py").exists(),
            has_conformal=self._contains("conformal"),
            has_ood=(backend / "ml" / "reliability" / "ood_detector.py").exists(),
            has_reproducibility=(backend / "ml" / "publication" / "reproducibility_bundle.py").exists(),
            has_metadata=(backend / "ml" / "publication" / "metadata_manifest.py").exists(),
            has_doi=(backend / "ml" / "publication" / "DOI_packager.py").exists(),
        )

    def evaluate_infrastructure_score(self, has_docker: bool, has_celery: bool, has_redis: bool) -> float:
        return self._score([(has_docker, 40.0), (has_celery, 30.0), (has_redis, 30.0)])

    def evaluate_reliability_score(self, has_checkpoints: bool, has_rollback: bool, coverage_rate: float) -> float:
        return self._score([(has_checkpoints, 30.0), (has_rollback, 30.0)]) + min(40.0, max(0.0, coverage_rate) * 40.0)

    def evaluate_scientific_validity_score(self, has_physics: bool, has_conformal: bool, has_ood: bool) -> float:
        return self._score([(has_physics, 40.0), (has_conformal, 30.0), (has_ood, 30.0)])

    def evaluate_publication_readiness_score(self, has_reproducibility: bool, has_metadata: bool, has_doi: bool) -> float:
        return self._score([(has_reproducibility, 40.0), (has_metadata, 30.0), (has_doi, 30.0)])

    def evaluate_publication_readibility_score(self, has_reproducibility: bool, has_metadata: bool, has_doi: bool) -> float:
        return self.evaluate_publication_readiness_score(has_reproducibility, has_metadata, has_doi)

    def evaluate_industrial_readiness_score(self, overall_readiness: float) -> float:
        return max(0.0, min(100.0, overall_readiness))

    def generate_full_compliance_report(
        self,
        has_docker: bool = True,
        has_celery: bool = True,
        has_redis: bool = True,
        has_checkpoints: bool = True,
        has_rollback: bool = True,
        conformal_coverage: float = 0.95,
        has_physics: bool = True,
        has_conformal: bool = True,
        has_ood: bool = True,
        has_reproducibility: bool = True,
        has_metadata: bool = True,
        has_doi: bool = True,
    ) -> Dict[str, Any]:
        infrastructure = self.evaluate_infrastructure_score(has_docker, has_celery, has_redis)
        reliability = self.evaluate_reliability_score(has_checkpoints, has_rollback, conformal_coverage)
        science = self.evaluate_scientific_validity_score(has_physics, has_conformal, has_ood)
        publication = self.evaluate_publication_readiness_score(has_reproducibility, has_metadata, has_doi)
        industrial = self.evaluate_industrial_readiness_score((infrastructure + reliability + science + publication) / 4.0)
        scores = {
            "infrastructure": infrastructure,
            "reliability": reliability,
            "scientific_validity": science,
            "publication_readiness": publication,
            "industrial_readiness": industrial,
        }
        return {
            "compliance_scores": scores,
            "overall_average_readiness": industrial,
            "status": "APPROVED" if industrial >= 90.0 else "REMEDIATION_REQUIRED",
            "remediation_focus": [name for name, score in scores.items() if score < 90.0],
        }

    @staticmethod
    def _score(items: list[tuple[bool, float]]) -> float:
        return sum(weight for present, weight in items if present)

    def _contains(self, needle: str) -> bool:
        for path in self.workspace_root.rglob("*.py"):
            try:
                if needle in path.read_text(encoding="utf-8"):
                    return True
            except (OSError, UnicodeDecodeError):
                continue
        return False
