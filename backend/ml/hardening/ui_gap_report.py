from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


class UIGapReporter:
    """Reports expected production UI surfaces missing from the Next.js app."""

    REQUIRED_SURFACES = {
        "experiment_dashboard": ("experiment", "trial", "synthesis"),
        "audit_viewer": ("audit", "security", "readiness"),
        "ood_explanations": ("ood", "out-of-distribution", "nearest"),
        "publication_pages": ("publication", "doi", "reproducibility"),
        "admin_monitoring": ("admin", "monitor", "health"),
        "validation_reports": ("validation", "blind", "coverage"),
    }

    def __init__(self, frontend_root: str):
        self.frontend_root = Path(frontend_root).resolve()

    def generate(self) -> Dict[str, Any]:
        present_routes = self._routes()
        corpus = self._page_text()
        gaps = []
        for surface, terms in self.REQUIRED_SURFACES.items():
            hits = [term for term in terms if term in corpus]
            if not hits:
                gaps.append(
                    {
                        "surface": surface,
                        "severity": "high",
                        "impact": "Required production workflow is not discoverable in the UI.",
                        "fix": f"Add or expose {surface.replace('_', ' ')} in existing frontend routing.",
                    }
                )
        return {
            "route_count": len(present_routes),
            "routes": present_routes,
            "missing_surfaces": gaps,
            "status": "complete" if not gaps else "gaps_detected",
        }

    def _routes(self) -> List[str]:
        app_dir = self.frontend_root / "src" / "app"
        routes = []
        for path in app_dir.rglob("page.tsx"):
            rel = path.parent.relative_to(app_dir).as_posix()
            routes.append("/" if rel == "." else f"/{rel}")
        return sorted(routes)

    def _page_text(self) -> str:
        chunks: List[str] = []
        for path in self._page_files():
            try:
                chunks.append(path.read_text(encoding="utf-8").lower())
            except (OSError, UnicodeDecodeError):
                continue
        return "\n".join(chunks)

    def _page_files(self) -> Iterable[Path]:
        app_dir = self.frontend_root / "src" / "app"
        if not app_dir.exists():
            return []
        return app_dir.rglob("page.tsx")
