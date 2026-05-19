from __future__ import annotations

from typing import Any, Dict, Iterable, List


class TechDebtReporter:
    """Converts audit outputs into prioritized remediation items."""

    DEFAULT_REMEDIATIONS = {
        "dead_modules": "Remove the module or wire it into an explicit entrypoint if it is still required.",
        "duplicate_logic": "Extract a shared helper only if both call sites are active and behavior must remain aligned.",
        "circular_imports": "Move shared constants/types to a lower-level module or invert the import boundary.",
        "redundant_abstractions": "Delete the abstraction or merge it into the concrete implementation until a second implementation exists.",
        "unused_services": "Remove the service or add a documented owner and integration point.",
        "duplicate_packages": "Keep a single dependency declaration in the canonical requirements file.",
        "version_conflicts": "Converge version specifiers to one compatible pinned version.",
        "oversized_dependencies": "Move heavyweight dependencies out of hot paths or make them optional extras.",
        "vulnerable_dependencies": "Upgrade to the minimum safe version and run the affected test suite.",
        "hardcoded_secrets": "Move secrets to environment variables or a secret manager with no production fallback.",
        "unsafe_deserialization": "Require signatures, checksums, or safe loaders before reading serialized artifacts.",
        "unrestricted_file_access": "Constrain paths to an allow-listed base directory and reject traversal.",
        "weak_validation": "Add bounds and schema validators at the API or data ingestion boundary.",
        "insecure_defaults": "Replace permissive development defaults with explicit environment-scoped settings.",
        "performance": "Profile the slowest operation with representative payloads and set a regression threshold.",
        "readiness": "Address readiness dimensions below 90 before release approval.",
        "missing_surfaces": "Expose the missing production workflow in existing navigation and route structure.",
        "experimental_gaps": "Close validation evidence gaps before industrial or publication claims.",
    }

    SEVERITY = {
        "vulnerable_dependencies": "critical",
        "hardcoded_secrets": "critical",
        "unsafe_deserialization": "critical",
        "unrestricted_file_access": "high",
        "insecure_defaults": "high",
        "circular_imports": "high",
        "version_conflicts": "medium",
        "weak_validation": "medium",
        "dead_modules": "medium",
        "unused_services": "medium",
        "duplicate_logic": "low",
        "redundant_abstractions": "low",
        "duplicate_packages": "low",
        "oversized_dependencies": "low",
        "performance": "medium",
        "readiness": "medium",
        "missing_surfaces": "medium",
        "experimental_gaps": "high",
    }

    PRIORITY = {"critical": 1, "high": 2, "medium": 3, "low": 4}

    def generate(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for category, payload in audit_results.items():
            if category == "performance":
                items.extend(self._performance_items(payload))
                continue
            if category == "readiness":
                items.extend(self._readiness_items(payload))
                continue
            if category not in self.SEVERITY and isinstance(payload, dict):
                for nested_category, nested_payload in payload.items():
                    if nested_category == "performance":
                        items.extend(self._performance_items(nested_payload))
                    elif nested_category == "readiness":
                        items.extend(self._readiness_items(nested_payload))
                    else:
                        for finding in self._iter_findings(nested_payload):
                            items.append(self._debt_item(nested_category, finding))
                continue
            for finding in self._iter_findings(payload):
                items.append(self._debt_item(category, finding))

        items.sort(key=lambda item: (item["fix_priority"], item["category"]))
        return {
            "total_items": len(items),
            "severity_counts": self._severity_counts(items),
            "items": items,
        }

    def _debt_item(self, category: str, finding: Any) -> Dict[str, Any]:
        severity = self.SEVERITY.get(category, "medium")
        evidence = finding if isinstance(finding, dict) else {"evidence": finding}
        remediation = evidence.get("suggested_remediation") or self.DEFAULT_REMEDIATIONS.get(
            category, "Review and remediate before production release."
        )
        return {
            "category": category,
            "severity": severity,
            "impact": self._impact_for(category),
            "fix_priority": self.PRIORITY[severity],
            "finding": evidence,
            "suggested_remediation": remediation,
        }

    def _performance_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = []
        for name, report in payload.items():
            if not isinstance(report, dict):
                continue
            duration = float(report.get("duration_ms") or report.get("average_latency_ms") or 0.0)
            memory = float(report.get("memory_growth_kb") or 0.0)
            if duration > 500.0 or memory > 1024.0:
                severity = "high" if duration > 2000.0 or memory > 10_240.0 else "medium"
                items.append(
                    {
                        "category": "performance",
                        "severity": severity,
                        "impact": "Latency or memory growth can erode API, websocket, checkpoint, or training reliability.",
                        "fix_priority": self.PRIORITY[severity],
                        "finding": {"operation": name, **report},
                        "suggested_remediation": self.DEFAULT_REMEDIATIONS["performance"],
                    }
                )
        return items

    def _readiness_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        scores = payload.get("compliance_scores", {}) if isinstance(payload, dict) else {}
        items = []
        for dimension, score in scores.items():
            if score < 90.0:
                severity = "high" if score < 60.0 else "medium"
                items.append(
                    {
                        "category": "readiness",
                        "severity": severity,
                        "impact": f"{dimension} readiness is below release threshold.",
                        "fix_priority": self.PRIORITY[severity],
                        "finding": {"dimension": dimension, "score": score},
                        "suggested_remediation": self.DEFAULT_REMEDIATIONS["readiness"],
                    }
                )
        return items

    @staticmethod
    def _iter_findings(payload: Any) -> Iterable[Any]:
        if isinstance(payload, dict):
            for value in payload.values():
                yield from TechDebtReporter._iter_findings(value)
        elif isinstance(payload, list):
            for item in payload:
                yield item
        elif isinstance(payload, set):
            for item in sorted(payload):
                yield item
        elif payload:
            yield payload

    @staticmethod
    def _severity_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for item in items:
            counts[item["severity"]] += 1
        return counts

    @staticmethod
    def _impact_for(category: str) -> str:
        if category in {"hardcoded_secrets", "unsafe_deserialization", "vulnerable_dependencies"}:
            return "Can create direct compromise or supply-chain exposure."
        if category in {"circular_imports", "unrestricted_file_access", "insecure_defaults"}:
            return "Can produce release instability or unsafe production behavior."
        if category in {"dead_modules", "unused_services", "version_conflicts", "weak_validation"}:
            return "Raises maintenance cost and increases regression risk."
        return "Low-grade maintainability drag unless it expands or blocks ownership."
