import numpy as np
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.db.models import BlindValidationTrial
from loguru import logger

class ValidationReportGenerator:
    """
    Evaluates predictions against actual physical measurements.
    Computes statistical MAE, R², conformal bounds coverage rates, and generates structured comparisons.
    """
    def __init__(self, db: Session):
        self.db = db

    def generate_trial_comparison(self, trial: BlindValidationTrial) -> Dict[str, Any]:
        """Compares predicted vs measured values for a single validation trial."""
        if not trial.measured_properties:
            return {"experiment_id": trial.experiment_id, "status": "no_measurements"}
            
        comparison = {}
        predictions = trial.predicted_properties
        measured = trial.measured_properties
        intervals = trial.prediction_interval
        
        for prop, actual in measured.items():
            pred_val = predictions.get(prop)
            bounds = intervals.get(prop) # [lower, upper]
            
            if pred_val is None:
                continue
                
            abs_err = abs(pred_val - actual)
            rel_err = (abs_err / actual) * 100 if actual > 0 else 0.0
            
            # Conformal coverage check
            covered = False
            if bounds and len(bounds) == 2:
                covered = float(bounds[0]) <= actual <= float(bounds[1])
                
            comparison[prop] = {
                "predicted": float(pred_val),
                "measured": float(actual),
                "prediction_interval": [float(b) for b in bounds] if bounds else None,
                "absolute_error": float(abs_err),
                "relative_error_pct": float(rel_err),
                "coverage_success": covered
            }
            
        return {
            "experiment_id": trial.experiment_id,
            "composition": trial.alloy_composition,
            "processing_route": trial.processing_route,
            "comparison": comparison,
            "operator": trial.operator,
            "specimen_id": trial.specimen_id,
            "synthesis_date": trial.synthesis_date.isoformat() if trial.synthesis_date else None,
            "created_at": trial.created_at.isoformat()
        }

    def generate_overall_validation_report(self) -> Dict[str, Any]:
        """Aggregates all completed trials to compute metrics and calibration success."""
        completed_trials = self.db.query(BlindValidationTrial).filter(
            BlindValidationTrial.lab_status == "completed"
        ).all()
        
        if not completed_trials:
            return {
                "status": "empty",
                "completed_count": 0,
                "message": "No completed experimental validation records found."
            }

        summaries = []
        errors_by_property: Dict[str, List[float]] = {}
        coverage_by_property: Dict[str, List[bool]] = {}
        
        for trial in completed_trials:
            comp = self.generate_trial_comparison(trial)
            summaries.append(comp)
            
            for prop, metrics in comp["comparison"].items():
                errors_by_property.setdefault(prop, []).append(metrics["absolute_error"])
                coverage_by_property.setdefault(prop, []).append(metrics["coverage_success"])
                
        # Aggregate stats
        property_analytics = {}
        for prop in errors_by_property.keys():
            errors = errors_by_property[prop]
            coverages = coverage_by_property[prop]
            
            mae = float(np.mean(errors))
            max_err = float(np.max(errors))
            coverage_rate = float(np.mean(coverages)) # percentage of points within interval
            
            property_analytics[prop] = {
                "sample_count": len(errors),
                "mean_absolute_error": mae,
                "maximum_error": max_err,
                "conformal_coverage_rate": coverage_rate,
                "calibration_score": float(abs(coverage_rate - 0.95)) # Distance to typical 95% Target
            }
            
        overall_coverage = float(np.mean([
            c for coverages in coverage_by_property.values() for c in coverages
        ])) if coverage_by_property else 0.0

        return {
            "status": "ready",
            "completed_count": len(completed_trials),
            "property_analytics": property_analytics,
            "overall_conformal_coverage": overall_coverage,
            "trials_summary": summaries
        }
