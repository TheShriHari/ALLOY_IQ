from __future__ import annotations

import re
import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from packaging.version import Version
except Exception:  # pragma: no cover
    Version = None


class DependencyAuditor:
    """Audits dependency declarations without requiring network access."""

    VULNERABILITY_RULES = {
        "urllib3": ("1.26.17", "CVE-2023-43804: Proxy-Authorization leak risk"),
        "requests": ("2.31.0", "CVE-2023-32681: credential leakage via redirects"),
        "pyjwt": ("2.4.0", "CVE-2022-29217: key confusion risk"),
        "jinja2": ("3.1.3", "CVE-2024-22195: template escaping risk"),
        "pydantic": ("1.10.13", "CVE-2023-45803: denial-of-service risk in v1"),
        "aiohttp": ("3.9.4", "CVE-2024-27306: request smuggling risk"),
    }
    OVERSIZED_PACKAGES = {
        "torch": (1500.0, "Large GPU/ML runtime; keep optional unless actively used."),
        "tensorflow": (800.0, "Large ML runtime; keep optional unless actively used."),
        "pymatgen": (250.0, "Heavy materials stack; isolate from latency-sensitive services."),
        "matminer": (180.0, "Heavy featurization stack; isolate from API hot paths."),
        "mlflow": (120.0, "Operational tracking stack; avoid importing in request handlers."),
        "scipy": (80.0, "Large scientific dependency; acceptable but monitor cold starts."),
    }

    def __init__(self, lock_lines: Iterable[str] | None = None, source_root: str | None = None):
        self.lock_lines = list(lock_lines or [])
        self.source_root = Path(source_root).resolve() if source_root else None

    def run_full_audit(self) -> Dict[str, Any]:
        return {
            "duplicate_packages": self.audit_duplicate_packages(),
            "version_conflicts": self.audit_version_conflicts(),
            "oversized_dependencies": self.detect_oversized_bloat(),
            "vulnerable_dependencies": self.audit_vulnerable_packages(),
            "unused_libraries": self.audit_unused_libraries(),
        }

    def audit_duplicate_packages(self) -> List[str]:
        seen = set()
        duplicates = []
        for req in self._parse_requirements():
            name = req["name"]
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        return duplicates

    def audit_version_conflicts(self) -> List[Dict[str, Any]]:
        by_name: Dict[str, set[str]] = {}
        for req in self._parse_requirements():
            by_name.setdefault(req["name"], set()).add(req["specifier"])
        return [
            {
                "package": name,
                "specifiers": sorted(specifiers),
                "issue": "Multiple version constraints declared for the same package.",
            }
            for name, specifiers in sorted(by_name.items())
            if len(specifiers) > 1
        ]

    def audit_vulnerable_packages(self) -> List[Dict[str, Any]]:
        findings = []
        for req in self._parse_requirements():
            package = req["name"]
            version = req["pinned_version"]
            if package not in self.VULNERABILITY_RULES or not version:
                continue
            minimum, bulletin = self.VULNERABILITY_RULES[package]
            if self._version_lt(version, minimum):
                findings.append(
                    {
                        "package": package,
                        "installed_version": version,
                        "minimum_safe_version": minimum,
                        "cve_bulletin": bulletin,
                        "suggested_remediation": f"Upgrade {package} to >= {minimum}.",
                    }
                )
        return findings

    def detect_oversized_bloat(self) -> List[Dict[str, Any]]:
        findings = []
        declared = {req["name"] for req in self._parse_requirements()}
        for package, (size_mb, bulletin) in self.OVERSIZED_PACKAGES.items():
            if package in declared:
                findings.append(
                    {
                        "package": package,
                        "estimated_size_mb": size_mb,
                        "impact": bulletin,
                        "suggested_remediation": "Keep dependency out of API import path or make it optional.",
                    }
                )
        return findings

    def audit_unused_libraries(self) -> List[Dict[str, Any]]:
        if not self.source_root or not self.source_root.exists():
            return []
        declared = {req["name"] for req in self._parse_requirements()}
        imported = self._top_level_imports()
        return [
            {
                "package": package,
                "issue": "Declared dependency has no detected import in source tree.",
                "suggested_remediation": "Remove the dependency or document why it is runtime-loaded.",
            }
            for package in sorted(declared - imported)
            if package not in {"pytest", "pytest-asyncio", "uvicorn", "python-multipart", "python-dotenv"}
        ]

    def _top_level_imports(self) -> set[str]:
        imports: set[str] = set()
        aliases = {
            "sklearn": "scikit-learn",
            "jose": "python-jose",
            "dotenv": "python-dotenv",
            "PIL": "pillow",
            "fitz": "pymupdf",
            "cv2": "opencv-python",
        }
        for path in self.source_root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update((alias.name.split(".", 1)[0].lower().replace("_", "-") for alias in node.names))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".", 1)[0].lower().replace("_", "-"))
        return {aliases.get(name, name) for name in imports}

    def _parse_requirements(self) -> List[Dict[str, str]]:
        requirements = []
        for raw in self.lock_lines:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith(("-r", "--")):
                continue
            match = re.match(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*([<>=!~].*)?$", line)
            if not match:
                continue
            name = match.group(1).lower().replace("_", "-")
            specifier = (match.group(2) or "").strip()
            pinned = ""
            pin_match = re.search(r"==\s*([A-Za-z0-9_.!+-]+)", specifier)
            if pin_match:
                pinned = pin_match.group(1)
            requirements.append({"name": name, "specifier": specifier or "unbounded", "pinned_version": pinned})
        return requirements

    @staticmethod
    def _version_lt(current: str, minimum: str) -> bool:
        if Version:
            try:
                return Version(current) < Version(minimum)
            except Exception:
                pass
        current_parts = [int(part) for part in re.findall(r"\d+", current)]
        minimum_parts = [int(part) for part in re.findall(r"\d+", minimum)]
        return current_parts < minimum_parts
