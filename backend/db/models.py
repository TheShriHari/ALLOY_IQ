"""
ALLOY IQ — Database Models (SQLAlchemy)
========================================
Defines all persistent entities: User, PredictionJob, InverseDesignJob.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, Enum, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


class UserTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    COMPLETE = "complete"
    ERROR = "error"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id            = Column(String, primary_key=True, default=_uuid)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    hashed_pw     = Column(String(255), nullable=False)
    display_name  = Column(String(100), nullable=True)
    tier          = Column(Enum(UserTier, name="user_tier_enum", create_type=True), default=UserTier.FREE, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    predictions   = relationship("PredictionJob",    back_populates="user", cascade="all, delete-orphan")
    inverse_jobs  = relationship("InverseDesignJob", back_populates="user", cascade="all, delete-orphan")


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id         = Column(String, ForeignKey("users.id"), nullable=False, index=True)

    # Input
    alloy_family    = Column(String(20), nullable=False)   # steel | hea | aluminum
    property_target = Column(String(50), nullable=False)   # yield_strength | hardness | …
    composition     = Column(JSON, nullable=False)          # {element: fraction}
    processing              = Column(JSON, nullable=True)           # {heat_treat_temp_C: …}
    heat_treatment_category = Column(String(100), nullable=True)
    annealing_temperature   = Column(Float, nullable=True)
    cooling_method          = Column(String(100), nullable=True)
    aging_treatment         = Column(String(100), nullable=True)
    manufacturing_route     = Column(String(100), nullable=True)
    thermal_budget_category = Column(String(50), nullable=True)
    paper_doi               = Column(String(250), nullable=True)
    research_group_id       = Column(String(100), nullable=True)

    # Output
    prediction      = Column(Float, nullable=True)
    lower_ci        = Column(Float, nullable=True)
    upper_ci        = Column(Float, nullable=True)
    confidence      = Column(Float, nullable=True)
    data_confidence = Column(String(10), nullable=True)    # high | moderate | low
    shap_data       = Column(JSON, nullable=True)          # waterfall + narrative
    render_url      = Column(String(500), nullable=True)   # Blender render path

    status          = Column(Enum(JobStatus, name="job_status_enum", create_type=True), default=JobStatus.PENDING, nullable=False)
    error_msg       = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    completed_at    = Column(DateTime, nullable=True)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    version_id      = Column(Integer, nullable=False, default=1)

    user            = relationship("User", back_populates="predictions")

    __mapper_args__ = {
        "version_id_col": version_id
    }


class InverseDesignJob(Base):
    __tablename__ = "inverse_design_jobs"

    id             = Column(String, primary_key=True, default=_uuid)
    user_id        = Column(String, ForeignKey("users.id"), nullable=False, index=True)

    # Input
    alloy_family   = Column(String(20), nullable=False)
    targets        = Column(JSON, nullable=False)       # {prop: [op, value]}
    constraints    = Column(JSON, nullable=True)        # {element: [lo, hi]}
    n_generations  = Column(Integer, default=150)
    pop_size       = Column(Integer, default=300)

    # Output
    pareto_front   = Column(JSON, nullable=True)        # list of candidate dicts
    n_candidates   = Column(Integer, nullable=True)
    current_generation  = Column(Integer, default=0, nullable=True)
    latest_pareto_front = Column(JSON, nullable=True)

    status         = Column(String(20), default="pending")
    error_msg      = Column(Text, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    completed_at   = Column(DateTime, nullable=True)

    user           = relationship("User", back_populates="inverse_jobs")


class RenderJob(Base):
    __tablename__ = "render_jobs"

    id          = Column(String, primary_key=True, default=_uuid)
    user_id     = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    composition = Column(JSON, nullable=False)
    predictions = Column(JSON, nullable=True)
    status      = Column(String(20), default="queued")  # queued | running | complete | failed
    image_url   = Column(String(500), nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)


class InverseDesignCheckpoint(Base):
    __tablename__ = "inverse_design_checkpoints"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    job_id            = Column(String, ForeignKey("inverse_design_jobs.id"), nullable=False, index=True)
    generation        = Column(Integer, nullable=False)
    file_path         = Column(String(500), nullable=False)
    checksum          = Column(String(64), nullable=False)
    previous_checksum = Column(String(64), nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow)


class PredictionAudit(Base):
    __tablename__ = "prediction_audits"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    job_id            = Column(String, ForeignKey("prediction_jobs.id"), nullable=False, index=True)
    risk_tier         = Column(String(20), nullable=False)  # LOW | MEDIUM | HIGH | REFUSE
    uncertainty_width = Column(Float, nullable=False)
    ood_score         = Column(Float, nullable=False)
    refusal_reason    = Column(String(500), nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    dataset_hash = Column(String(64), nullable=False)
    feature_hash = Column(String(64), nullable=False)
    model_hash   = Column(String(64), nullable=False)
    metrics_path = Column(String(500), nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    model_id     = Column(String(50), primary_key=True)
    model_hash   = Column(String(64), nullable=False)
    feature_hash = Column(String(64), nullable=False)
    dataset_hash = Column(String(64), nullable=False)
    metrics      = Column(JSON, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)
    status       = Column(String(50), nullable=False)  # "candidate" | "validated" | "active"


class ModelTrainingJob(Base):
    __tablename__ = "model_training_jobs"

    id           = Column(String, primary_key=True, default=_uuid)
    alloy_family = Column(String(20), nullable=False)
    status       = Column(String(20), default="pending")  # pending | running | complete | failed
    progress     = Column(Float, default=0.0)             # 0.0 to 1.0
    heartbeat    = Column(DateTime, default=datetime.utcnow)
    error_msg    = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BlindValidationTrial(Base):
    __tablename__ = "blind_validation_trials"

    experiment_id        = Column(String, primary_key=True, default=_uuid)
    alloy_composition    = Column(JSON, nullable=False)
    processing_route     = Column(JSON, nullable=False)
    predicted_properties = Column(JSON, nullable=False)
    prediction_interval  = Column(JSON, nullable=False)  # conformal confidence limits
    created_at           = Column(DateTime, default=datetime.utcnow, nullable=False)
    locked_hash          = Column(String(64), nullable=False)  # SHA-256 validation lock

    # Synthesis tracking
    lab_status           = Column(String(50), default="locked", nullable=False)
    synthesis_date       = Column(DateTime, nullable=True)
    operator             = Column(String(100), nullable=True)
    specimen_id          = Column(String(100), nullable=True)
    process_notes        = Column(Text, nullable=True)

    # Ingestion results
    measured_properties  = Column(JSON, nullable=True)
    ingested_at          = Column(DateTime, nullable=True)






