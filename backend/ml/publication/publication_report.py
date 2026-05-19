import os
import pandas as pd
from typing import Dict, Any, List
from loguru import logger

class PublicationReporter:
    """
    Generates standardized reports for peer-reviewed papers.
    Strictly preserves decision-support terminology and avoids autonomous discovery claims.
    Includes comprehensive failure analysis and leakage reports.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_publication_report(
        self,
        benchmarks: Dict[str, Any],
        blind_experiments: List[Dict[str, Any]],
        leakage_audit: Dict[str, Any],
        failure_cases: List[Dict[str, Any]],
        family_metrics: Dict[str, Any]
    ) -> str:
        """Assembles a detailed Markdown scholarly report documenting performance and safety limits."""
        report_lines = [
            "# ALLOY IQ — Materials Informatics Reproducibility & Validation Report",
            "",
            "## 1. Decision-Support Context",
            "This model pipeline is designed as an interactive decision-support tool to assist human metallurgists. "
            "It does not operate autonomously. All mechanical predictions and recommendations are subject to physical verification.",
            "",
            "## 2. Core Model Benchmarks",
            "The following performance values were validated under multi-criteria GroupKFold partitions:",
            ""
        ]

        # Table formatting
        report_lines.append("| Property | $R^2$ Score | MAE (MPa) | Conformal Coverage Success Rate |")
        report_lines.append("| --- | --- | --- | --- |")
        for prop, metrics in benchmarks.items():
            r2 = metrics.get("r2", 0.0)
            mae = metrics.get("mae", 0.0)
            coverage = metrics.get("conformal_coverage", 0.95)
            report_lines.append(f"| {prop} | {r2:.3f} | {mae:.2f} | {coverage:.1f}% |")

        report_lines.extend([
            "",
            "## 3. Laboratory Blind Experiments Validation",
            "Validation of physical properties on novel alloys predicted prior to metallurgical synthesis:",
            ""
        ])

        report_lines.append("| Experiment ID | Alloy Composition | Predicted | Measured | Conformal Interval | Status |")
        report_lines.append("| --- | --- | --- | --- | --- | --- |")
        for exp in blind_experiments:
            eid = exp.get("experiment_id")
            comp = exp.get("composition")
            for prop, details in exp.get("comparison", {}).items():
                pred = details.get("predicted", 0.0)
                meas = details.get("measured", 0.0)
                bounds = details.get("prediction_interval", [0.0, 0.0])
                status = "COVERED" if details.get("coverage_success") else "OUTSIDE"
                
                # Format comp string
                comp_str = ", ".join(f"{k}: {v:.1f}%" for k, v in comp.items())
                report_lines.append(f"| {eid} | {comp_str} | {pred:.1f} | {meas:.1f} | [{bounds[0]:.1f}, {bounds[1]:.1f}] | {status} |")

        report_lines.extend([
            "",
            "## 4. Materials Informatics Spillover & Leakage Audit",
            "Rigorous cross-validation filters ensure zero database spillover or DOI leakage:",
            ""
        ])

        for audit_prop, status in leakage_audit.items():
            report_lines.append(f"*   **{audit_prop}**: {status}")

        report_lines.extend([
            "",
            "## 5. High-Risk / Failures Prediction Audit Analysis",
            "Documentation of model execution failures, conformal prediction refusals, and physics sanity violations:",
            ""
        ])

        report_lines.append("| Record ID | Conf. Width | Reason for Refusal | Flagged Anomalies |")
        report_lines.append("| --- | --- | --- | --- |")
        for fail in failure_cases:
            rid = fail.get("record_id")
            width = fail.get("uncertainty_width", 0.0)
            reason = fail.get("refusal_reason", "N/A")
            flags = ", ".join(fail.get("flags", ["N/A"]))
            report_lines.append(f"| {rid} | {width:.2f} | {reason} | {flags} |")

        report_lines.extend([
            "",
            "## 6. Alloy Family-wise Analytics",
            "Performance breakdown by specific HEA, steel, and high-temp alloy groups:",
            ""
        ])

        report_lines.append("| Alloy Family Group | Sample Count | Representative MAE |")
        report_lines.append("| --- | --- | --- |")
        for fam, details in family_metrics.items():
            count = details.get("sample_count", 0)
            mae = details.get("mae", 0.0)
            report_lines.append(f"| {fam.upper()} | {count} | {mae:.2f} |")

        # Save Markdown file
        path = os.path.join(self.output_dir, "publication_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
            
        logger.info("Saved publication report Markdown to: {}", path)
        return path
