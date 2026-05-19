import pickle
import textwrap

from backend.ml.hardening.architecture_audit import ArchitectureAuditor
from backend.ml.hardening.dependency_audit import DependencyAuditor
from backend.ml.hardening.experimental_gap_report import ExperimentalGapReporter
from backend.ml.hardening.performance_profiler import PerformanceProfiler
from backend.ml.hardening.production_readiness import ProductionReadinessEvaluator
from backend.ml.hardening.security_review import SecurityReviewer
from backend.ml.hardening.tech_debt_report import TechDebtReporter
from backend.ml.hardening.ui_gap_report import UIGapReporter


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_architecture_audit_detects_cycles_dead_modules_and_duplicate_logic(tmp_path):
    package = tmp_path / "backend"
    write_file(package / "main.py", "from backend import a\n")
    write_file(
        package / "a.py",
        """
        from backend import b

        def repeated(value):
            total = value + 1
            total = total * 2
            return total
        """,
    )
    write_file(
        package / "b.py",
        """
        from backend import a

        def repeated(value):
            total = value + 1
            total = total * 2
            return total
        """,
    )
    write_file(
        package / "unused_service.py",
        """
        class UnusedService:
            def run(self):
                return "unused"
        """,
    )
    write_file(package / "empty_layer.py", "class EmptyInterface:\n    pass\n")

    audit = ArchitectureAuditor(str(package)).run_full_audit(["main.py"])

    assert any("backend.a" in cycle and "backend.b" in cycle for cycle in audit["circular_imports"])
    assert "backend.unused_service" in audit["dead_modules"]
    assert audit["duplicate_logic"]
    assert any(item["class"] == "EmptyInterface" for item in audit["redundant_abstractions"])
    assert any(item["class"] == "UnusedService" for item in audit["unused_services"])
    assert audit["oversized_modules"] == []


def test_dependency_audit_detects_duplicates_conflicts_bloat_vulnerabilities_and_unused(tmp_path):
    write_file(tmp_path / "app.py", "import urllib3\n")
    lines = [
        "requests==2.30.0",
        "requests>=2.31.0",
        "urllib3==1.26.16",
        "pymatgen==2024.5.1",
    ]

    audit = DependencyAuditor(lines, source_root=str(tmp_path)).run_full_audit()

    assert "requests" in audit["duplicate_packages"]
    assert any(item["package"] == "requests" for item in audit["version_conflicts"])
    assert any(item["package"] == "urllib3" for item in audit["vulnerable_dependencies"])
    assert any(item["package"] == "pymatgen" for item in audit["oversized_dependencies"])
    assert any(item["package"] == "pymatgen" for item in audit["unused_libraries"])


def test_security_review_detects_vulnerability_patterns(tmp_path):
    write_file(
        tmp_path / "service.py",
        """
        import os
        import pickle
        from pydantic import BaseModel

        SECRET_KEY = "real-production-secret"

        class Request(BaseModel):
            confidence: float

        def load_blob(path):
            with open(path, "rb") as handle:
                return pickle.load(handle)

        def remove_parent():
            os.remove("../outside.txt")
        """,
    )
    write_file(
        tmp_path / "api.py",
        """
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI()
        app.add_middleware(CORSMiddleware, allow_origins=["*"])
        """,
    )

    review = SecurityReviewer(str(tmp_path)).run_full_review()

    assert review["hardcoded_secrets"]
    assert review["unsafe_deserialization"]
    assert review["unrestricted_file_access"]
    assert review["weak_validation"]
    assert review["insecure_defaults"]


def test_performance_profiler_executes_latency_checkpoint_and_memory_paths():
    profiler = PerformanceProfiler()

    api = profiler.measure_api_latency(lambda: {"ok": True}, iterations=3)
    websocket = profiler.measure_websocket_throughput(lambda message: len(message), "ping", iterations=5)
    checkpoint = profiler.measure_checkpoint_overhead(pickle.dumps, pickle.loads, {"generation": 2})
    memory = profiler.measure_memory_growth(lambda: [0] * 10, iterations=3)
    queue = profiler.measure_queue_delay(enqueued_at=100.0, started_at=101.25)
    model_load = profiler.measure_model_load_time(lambda: {"loaded": True})

    assert api["iterations"] == 3
    assert websocket["total_iterations"] == 5
    assert checkpoint["payload_footprint_bytes"] > 0
    assert memory["iterations"] == 3
    assert queue["delay_ms"] == 1250.0
    assert model_load["operation"] == "model_load_time"


def test_production_readiness_scoring_and_remediation_focus(tmp_path):
    evaluator = ProductionReadinessEvaluator(str(tmp_path))

    report = evaluator.generate_full_compliance_report(
        has_docker=True,
        has_celery=False,
        has_redis=True,
        has_checkpoints=True,
        has_rollback=False,
        conformal_coverage=0.5,
        has_physics=True,
        has_conformal=True,
        has_ood=False,
        has_reproducibility=True,
        has_metadata=False,
        has_doi=False,
    )

    scores = report["compliance_scores"]
    assert scores["infrastructure"] == 70.0
    assert scores["reliability"] == 50.0
    assert scores["scientific_validity"] == 70.0
    assert scores["publication_readiness"] == 40.0
    assert report["status"] == "REMEDIATION_REQUIRED"


def test_tech_debt_report_prioritizes_remediation_items():
    report = TechDebtReporter().generate(
        {
            "hardcoded_secrets": [{"file": "main.py", "line": 4}],
            "duplicate_logic": [{"locations": ["a.py", "b.py"]}],
            "readiness": {"compliance_scores": {"infrastructure": 70.0}},
        }
    )

    assert report["total_items"] == 3
    assert report["items"][0]["severity"] == "critical"
    assert report["items"][0]["fix_priority"] == 1
    assert report["severity_counts"]["critical"] == 1


def test_ui_gap_report_detects_missing_required_surfaces(tmp_path):
    write_file(tmp_path / "src" / "app" / "page.tsx", "export default function Page() { return <main>Predict</main> }\n")

    report = UIGapReporter(str(tmp_path)).generate()

    assert report["route_count"] == 1
    assert report["status"] == "gaps_detected"
    assert any(item["surface"] == "audit_viewer" for item in report["missing_surfaces"])


def test_experimental_gap_report_counts_trials_and_coverage_gaps():
    trials = [
        {
            "lab_status": "completed",
            "locked_hash": "abc",
            "prediction_interval": {"yield_strength": [800, 900]},
            "family": "refractory",
            "processing_route": {"route": "cast"},
        },
        {"lab_status": "locked", "locked_hash": "def", "family": "lightweight", "processing_route": {"route": "powder"}},
    ]

    report = ExperimentalGapReporter().generate(trials)

    assert report["synthesized_alloy_count"] == 1
    assert report["blind_trial_count"] == 2
    assert report["uncertainty_evidence_rate"] == 0.5
    assert "lightweight" in report["family_coverage_gaps"]
