from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


class ExperimentalGapReporter:
    """Summarizes validation coverage gaps without adding new experiment systems."""

    TARGET_SYNTHESIZED_ALLOYS = 30
    TARGET_BLIND_TRIALS = 10
    TARGET_UNCERTAINTY_EVIDENCE_RATE = 0.9

    def generate(self, trials: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        records = list(trials)
        synthesized = [item for item in records if item.get("lab_status") in {"synthesizing", "completed"}]
        blind_trials = [item for item in records if item.get("locked_hash")]
        uncertainty_backed = [item for item in records if item.get("prediction_interval") or item.get("uncertainty_evidence")]
        family_counts = Counter(item.get("family", "unknown") for item in records)
        route_counts = Counter(self._route_name(item.get("processing_route")) for item in records)

        gaps = []
        if len(synthesized) < self.TARGET_SYNTHESIZED_ALLOYS:
            gaps.append(self._gap("synthesized_alloy_count", len(synthesized), self.TARGET_SYNTHESIZED_ALLOYS))
        if len(blind_trials) < self.TARGET_BLIND_TRIALS:
            gaps.append(self._gap("blind_trial_count", len(blind_trials), self.TARGET_BLIND_TRIALS))
        evidence_rate = len(uncertainty_backed) / len(records) if records else 0.0
        if evidence_rate < self.TARGET_UNCERTAINTY_EVIDENCE_RATE:
            gaps.append(self._gap("uncertainty_evidence", evidence_rate, self.TARGET_UNCERTAINTY_EVIDENCE_RATE))

        return {
            "synthesized_alloy_count": len(synthesized),
            "blind_trial_count": len(blind_trials),
            "uncertainty_evidence_rate": evidence_rate,
            "family_coverage": dict(sorted(family_counts.items())),
            "processing_route_coverage": dict(sorted(route_counts.items())),
            "family_coverage_gaps": [family for family, count in family_counts.items() if count < 2],
            "processing_route_coverage_gaps": [route for route, count in route_counts.items() if count < 2],
            "gaps": gaps,
            "status": "complete" if not gaps else "gaps_detected",
        }

    @staticmethod
    def _route_name(route: Any) -> str:
        if isinstance(route, dict):
            return str(route.get("route") or route.get("name") or route.get("method") or "unknown")
        return str(route or "unknown")

    @staticmethod
    def _gap(metric: str, actual: float, target: float) -> Dict[str, Any]:
        return {
            "metric": metric,
            "actual": actual,
            "target": target,
            "severity": "high" if actual < target * 0.5 else "medium",
            "fix": f"Increase {metric.replace('_', ' ')} evidence before release.",
        }
