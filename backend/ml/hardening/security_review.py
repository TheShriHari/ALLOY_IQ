from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


class SecurityReviewer:
    """Static scanner for hardening issues that have concrete remediations."""

    SECRET_NAMES = re.compile(r"(password|secret|api[_-]?key|private[_-]?key|token|auth[_-]?key)", re.IGNORECASE)
    SAFE_PLACEHOLDERS = {"", "change-me", "changeme", "example", "dummy", "test", "placeholder"}

    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir).resolve()

    def run_full_review(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "hardcoded_secrets": self.scan_for_hardcoded_secrets(),
            "unsafe_deserialization": self.scan_for_unsafe_deserialization(),
            "unrestricted_file_access": self.scan_for_unrestricted_file_access(),
            "weak_validation": self.scan_for_weak_validations(),
            "insecure_defaults": self.scan_for_insecure_defaults(),
        }

    def scan_for_hardcoded_secrets(self) -> List[Dict[str, Any]]:
        findings = []
        for path in self._python_files():
            if path.name.startswith("test_"):
                continue
            tree = self._parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        name = self._name_of(target)
                        if name and self.SECRET_NAMES.search(name):
                            value = self._literal_string(node.value)
                            if value is not None and value.lower() not in self.SAFE_PLACEHOLDERS:
                                findings.append(self._finding(path, node.lineno, "Hardcoded secret-like value.", name))
                elif isinstance(node, ast.Call) and self._name_of(node.func) in {"os.getenv", "getenv"}:
                    if len(node.args) >= 2:
                        key = self._literal_string(node.args[0]) or ""
                        default = self._literal_string(node.args[1])
                        if self.SECRET_NAMES.search(key) and default and default.lower() not in self.SAFE_PLACEHOLDERS:
                            findings.append(self._finding(path, node.lineno, "Secret environment fallback is hardcoded.", key))
        return findings

    def scan_for_unsafe_deserialization(self) -> List[Dict[str, Any]]:
        unsafe = []
        unsafe_calls = {"pickle.load", "pickle.loads", "dill.load", "dill.loads", "yaml.load", "joblib.load"}
        for path in self._python_files():
            tree = self._parse(path)
            if tree is None:
                continue
            imports = self._import_aliases(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = self._name_of(node.func)
                    resolved = imports.get(name or "", name)
                    if resolved in unsafe_calls:
                        unsafe.append(
                            self._finding(
                                path,
                                node.lineno,
                                "Unsafe deserialization entrypoint requires signature, checksum, or safe loader.",
                                resolved,
                            )
                        )
        return unsafe

    def scan_for_unrestricted_file_access(self) -> List[Dict[str, Any]]:
        findings = []
        file_calls = {"open", "Path.open", "os.remove", "os.unlink", "shutil.rmtree"}
        for path in self._python_files():
            tree = self._parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and self._name_of(node.func) in file_calls and node.args:
                    first = node.args[0]
                    literal = self._literal_string(first)
                    if literal and (".." in literal or os.path.isabs(literal)):
                        findings.append(self._finding(path, node.lineno, "Path traversal or absolute path access.", literal))
                    elif isinstance(first, (ast.Name, ast.JoinedStr, ast.BinOp)):
                        findings.append(
                            self._finding(
                                path,
                                node.lineno,
                                "Dynamic file access should be constrained to an approved base directory.",
                                self._name_of(node.func) or "file_call",
                            )
                        )
        return findings

    def scan_for_weak_validations(self) -> List[Dict[str, Any]]:
        findings = []
        risky_fields = {"composition", "processing", "confidence", "temperature", "yield_strength", "elongation"}
        for path in self._python_files():
            if path.name.startswith("test_"):
                continue
            tree = self._parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign):
                    name = self._name_of(node.target)
                    if name and name in risky_fields and not self._has_field_bounds(node.value):
                        findings.append(
                            self._finding(
                                path,
                                node.lineno,
                                "Scientific input lacks local bounds or validator enforcement.",
                                name,
                            )
                        )
        return findings

    def scan_for_insecure_defaults(self) -> List[Dict[str, Any]]:
        findings = []
        for path in self._python_files():
            tree = self._parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.keyword) and node.arg == "allow_origins":
                    if isinstance(node.value, ast.List) and any(self._literal_string(item) == "*" for item in node.value.elts):
                        findings.append(self._finding(path, node.lineno, "Wildcard CORS origin is enabled.", "allow_origins=*"))
                if isinstance(node, ast.keyword) and node.arg == "debug" and isinstance(node.value, ast.Constant) and node.value.value is True:
                    findings.append(self._finding(path, node.lineno, "Debug mode defaults to true.", "debug=True"))
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if self._name_of(target) == "SECRET_KEY":
                            value = self._literal_string(node.value)
                            if value and "change-me" in value.lower():
                                findings.append(self._finding(path, node.lineno, "Production secret fallback is insecure.", "SECRET_KEY"))
                if isinstance(node, ast.Call) and self._name_of(node.func) in {"os.getenv", "getenv"}:
                    if len(node.args) >= 2:
                        key = self._literal_string(node.args[0]) or ""
                        default = self._literal_string(node.args[1]) or ""
                        if key == "SECRET_KEY" and "change-me" in default.lower():
                            findings.append(self._finding(path, node.lineno, "Production secret fallback is insecure.", key))
        return findings

    def _python_files(self) -> Iterable[Path]:
        for path in self.target_dir.rglob("*.py"):
            yield path.resolve()

    @staticmethod
    def _parse(path: Path) -> ast.AST | None:
        try:
            return ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return None

    @staticmethod
    def _import_aliases(tree: ast.AST) -> Dict[str, str]:
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        return aliases

    @staticmethod
    def _has_field_bounds(value: ast.AST | None) -> bool:
        if not isinstance(value, ast.Call):
            return False
        name = SecurityReviewer._name_of(value.func)
        if name not in {"Field", "pydantic.Field"}:
            return False
        return any(keyword.arg in {"gt", "ge", "lt", "le", "min_length", "max_length"} for keyword in value.keywords)

    @staticmethod
    def _literal_string(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    @staticmethod
    def _name_of(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = SecurityReviewer._name_of(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return None

    def _finding(self, path: Path, line: int, issue: str, evidence: str) -> Dict[str, Any]:
        return {
            "file": str(path.relative_to(self.target_dir)),
            "line": line,
            "issue": issue,
            "evidence": evidence,
            "suggested_remediation": self._remediation_for(issue),
        }

    @staticmethod
    def _remediation_for(issue: str) -> str:
        if "secret" in issue.lower():
            return "Load secret from environment or secret manager and fail closed when absent."
        if "deserialization" in issue.lower():
            return "Use signed artifacts, checksum validation, or safe loaders before loading data."
        if "file access" in issue.lower() or "path" in issue.lower():
            return "Resolve paths against an allow-listed base directory and reject traversal."
        if "validation" in issue.lower():
            return "Add Pydantic bounds or field validators close to the API/schema boundary."
        return "Replace insecure default with explicit production configuration."
