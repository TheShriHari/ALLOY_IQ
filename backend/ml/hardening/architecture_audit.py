from __future__ import annotations

import ast
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


class ArchitectureAuditor:
    """Static architecture checks for remediation-focused hardening reports."""

    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir).resolve()
        self.package_root = self.target_dir.parent

    def run_full_audit(self, entrypoint_files: Iterable[str] | None = None) -> Dict[str, Any]:
        entrypoints = list(entrypoint_files or ["main.py"])
        return {
            "dead_modules": sorted(self.detect_unused_modules(entrypoints)),
            "duplicate_logic": self.detect_duplicate_logic(),
            "circular_imports": self.scan_for_circular_imports(),
            "redundant_abstractions": self.identify_redundant_abstractions(),
            "unused_services": self.detect_unused_services(entrypoints),
            "oversized_modules": self.detect_oversized_modules(),
        }

    def scan_for_circular_imports(self) -> List[List[str]]:
        graph = self._build_import_graph()
        cycles: List[List[str]] = []
        visiting: List[str] = []
        visited: Set[str] = set()
        seen_cycles: Set[tuple[str, ...]] = set()

        def dfs(module: str) -> None:
            if module in visiting:
                cycle = visiting[visiting.index(module) :] + [module]
                key = tuple(cycle)
                if key not in seen_cycles:
                    cycles.append(cycle)
                    seen_cycles.add(key)
                return
            if module in visited:
                return
            visiting.append(module)
            for imported in graph.get(module, set()):
                if imported in graph:
                    dfs(imported)
            visiting.pop()
            visited.add(module)

        for module in sorted(graph):
            dfs(module)
        return cycles

    def detect_unused_modules(self, entrypoint_files: Iterable[str]) -> Set[str]:
        graph = self._build_import_graph()
        entry_modules = {
            self._module_from_path((self.target_dir / entrypoint).resolve())
            for entrypoint in entrypoint_files
            if (self.target_dir / entrypoint).exists()
        }
        reachable: Set[str] = set()
        stack = list(entry_modules)
        while stack:
            module = stack.pop()
            if module in reachable:
                continue
            reachable.add(module)
            stack.extend(graph.get(module, set()) - reachable)
        return set(graph) - reachable

    def detect_duplicate_logic(self, min_body_nodes: int = 3) -> List[Dict[str, Any]]:
        fingerprints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for path in self._python_files():
            tree = self._parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(node.body) >= min_body_nodes:
                    normalized = ast.dump(node, annotate_fields=False, include_attributes=False)
                    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    fingerprints[digest].append(
                        {"module": self._module_from_path(path), "name": node.name, "line": node.lineno}
                    )
        return [
            {"fingerprint": digest, "locations": locations}
            for digest, locations in fingerprints.items()
            if len(locations) > 1
        ]

    def identify_redundant_abstractions(self) -> List[Dict[str, Any]]:
        class_defs: Dict[str, Dict[str, Any]] = {}
        base_refs: Set[str] = set()
        for path in self._python_files(include_tests=False):
            tree = self._parse(path)
            if tree is None:
                continue
            module = self._module_from_path(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [item for item in node.body if isinstance(item, ast.FunctionDef)]
                    non_doc_body = [item for item in node.body if not isinstance(item, ast.Expr)]
                    class_defs[node.name] = {
                        "module": module,
                        "class": node.name,
                        "line": node.lineno,
                        "method_count": len(methods),
                        "body_count": len(non_doc_body),
                    }
                    for base in node.bases:
                        base_name = self._name_of(base)
                        if base_name:
                            base_refs.add(base_name.split(".")[-1])
        findings = []
        for class_name, meta in class_defs.items():
            is_marker = meta["body_count"] == 1 and meta["method_count"] == 0
            is_unimplemented_interface = class_name.endswith(("Base", "Interface", "Protocol")) and class_name not in base_refs
            if is_marker or is_unimplemented_interface:
                findings.append({**meta, "issue": "Redundant abstraction with no detected implementation pressure."})
        return findings

    def detect_unused_services(self, entrypoint_files: Iterable[str]) -> List[Dict[str, Any]]:
        reachable = self._reachable_modules(entrypoint_files)
        findings = []
        for path in self._python_files(include_tests=False):
            module = self._module_from_path(path)
            if module in reachable:
                continue
            tree = self._parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.endswith(("Service", "Manager", "Client")):
                    findings.append(
                        {
                            "module": module,
                            "class": node.name,
                            "line": node.lineno,
                            "issue": "Service-style class is not reachable from configured entrypoints.",
                        }
                    )
        return findings

    def detect_oversized_modules(self, max_lines: int = 500, max_classes: int = 8, max_functions: int = 30) -> List[Dict[str, Any]]:
        findings = []
        for path in self._python_files(include_tests=False):
            tree = self._parse(path)
            if tree is None:
                continue
            try:
                line_count = len(path.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeDecodeError):
                line_count = 0
            class_count = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
            function_count = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
            if line_count > max_lines or class_count > max_classes or function_count > max_functions:
                findings.append(
                    {
                        "module": self._module_from_path(path),
                        "lines": line_count,
                        "classes": class_count,
                        "functions": function_count,
                        "issue": "Module exceeds maintainability threshold.",
                    }
                )
        return findings

    def _reachable_modules(self, entrypoint_files: Iterable[str]) -> Set[str]:
        graph = self._build_import_graph()
        dead = self.detect_unused_modules(entrypoint_files)
        return set(graph) - dead

    def _build_import_graph(self) -> Dict[str, Set[str]]:
        modules = {self._module_from_path(path): path for path in self._python_files(include_tests=False)}
        graph: Dict[str, Set[str]] = {module: set() for module in modules}
        for module, path in modules.items():
            for imported in self._extract_file_imports(path):
                match = self._resolve_internal_import(imported, modules)
                if match:
                    graph[module].add(match)
        return graph

    def _extract_file_imports(self, filepath: str | Path) -> List[str]:
        path = Path(filepath)
        tree = self._parse(path)
        if tree is None:
            return []
        imports: List[str] = []
        current_module = self._module_from_path(path)
        current_package = current_module.rsplit(".", 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base_parts = current_package.split(".")
                    prefix = ".".join(base_parts[: max(0, len(base_parts) - node.level + 1)])
                    if node.module:
                        imports.append(f"{prefix}.{node.module}".strip("."))
                    else:
                        imports.extend(f"{prefix}.{alias.name}".strip(".") for alias in node.names)
                elif node.module:
                    imports.append(node.module)
                    imports.extend(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")
        return imports

    def _resolve_internal_import(self, imported: str, modules: Dict[str, Path]) -> str | None:
        if imported in modules:
            return imported
        matches = [module for module in modules if module == imported or module.startswith(f"{imported}.")]
        return sorted(matches, key=len)[0] if matches else None

    def _python_files(self, include_tests: bool = True) -> Iterable[Path]:
        for path in self.target_dir.rglob("*.py"):
            if path.name == "__init__.py":
                continue
            if not include_tests and (path.name.startswith("test_") or "\\tests\\" in str(path)):
                continue
            yield path.resolve()

    def _module_from_path(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.package_root)
        return ".".join(rel.with_suffix("").parts)

    @staticmethod
    def _parse(path: Path) -> ast.AST | None:
        try:
            return ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return None

    @staticmethod
    def _name_of(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = ArchitectureAuditor._name_of(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return None
