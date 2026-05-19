from datetime import datetime
from typing import Dict, Any
from sqlalchemy.orm import Session
from backend.db.models import BlindValidationTrial
from loguru import logger

class LabResultIngestor:
    """
    Ingests and validates actual measured physical properties from experimental tests.
    Parses and registers mechanical measurements prior to analytics rendering.
    """
    def __init__(self, db: Session):
        self.db = db

    def validate_lab_results(self, measurements: Dict[str, float]) -> Dict[str, Any]:
        """Performs rigorous boundary and safety range validations on ingested values."""
        validated = {}
        errors = []

        # 1. Yield Strength (MPa) - metallurgical bounds: 10 to 5000 MPa
        ys = measurements.get("yield_strength")
        if ys is not None:
            if 10.0 <= ys <= 5000.0:
                validated["yield_strength"] = float(ys)
            else:
                errors.append(f"Invalid Yield Strength value: {ys} MPa (Expected: 10 - 5000)")

        # 2. Tensile Strength (MPa) - metallurgical bounds: 10 to 5000 MPa
        uts = measurements.get("tensile_strength")
        if uts is not None:
            if 10.0 <= uts <= 5000.0:
                validated["tensile_strength"] = float(uts)
            else:
                errors.append(f"Invalid Tensile Strength value: {uts} MPa (Expected: 10 - 5000)")

        # 3. Elongation (%) - metallurgical bounds: 0.1% to 100%
        elon = measurements.get("elongation")
        if elon is not None:
            if 0.1 <= elon <= 100.0:
                validated["elongation"] = float(elon)
            else:
                errors.append(f"Invalid Elongation value: {elon}% (Expected: 0.1 - 100)")

        # 4. Hardness (HV) - metallurgical bounds: 10 to 1500 HV
        hard = measurements.get("hardness")
        if hard is not None:
            if 10.0 <= hard <= 1500.0:
                validated["hardness"] = float(hard)
            else:
                errors.append(f"Invalid Hardness value: {hard} HV (Expected: 10 - 1500)")

        # 5. Density (g/cm³) - physical bounds: 0.5 to 25 g/cm³
        dens = measurements.get("density")
        if dens is not None:
            if 0.5 <= dens <= 25.0:
                validated["density"] = float(dens)
            else:
                errors.append(f"Invalid Density value: {dens} g/cm³ (Expected: 0.5 - 25)")

        # Safety cross-check: yield strength must be <= tensile strength
        if "yield_strength" in validated and "tensile_strength" in validated:
            if validated["yield_strength"] > validated["tensile_strength"]:
                errors.append(f"Physical Conflict: Ingested Yield Strength ({validated['yield_strength']}) exceeds Tensile Strength ({validated['tensile_strength']}).")

        return {"validated": validated, "errors": errors}

    def ingest_experimental_results(self, experiment_id: str, raw_measurements: Dict[str, float]) -> bool:
        """Saves physical measurements into the matching cryptographically locked trial record."""
        trial = self.db.query(BlindValidationTrial).filter(BlindValidationTrial.experiment_id == experiment_id).first()
        if not trial:
            logger.error("Ingestion failed: Trial {} not found.", experiment_id)
            return False
            
        if trial.lab_status == "completed":
            logger.warning("Measurements already ingested for trial: {}", experiment_id)
            return False

        # Run validations
        audit = self.validate_lab_results(raw_measurements)
        if audit["errors"]:
            logger.error("Ingestion rejected due to metallurgical failures: {}", audit["errors"])
            raise ValueError(f"ValidationError: {'; '.join(audit['errors'])}")

        trial.measured_properties = audit["validated"]
        trial.lab_status = "completed"
        trial.ingested_at = datetime.utcnow()
        
        self.db.commit()
        logger.info("Successfully ingested lab measurements for experiment: {}", experiment_id)
        return True
