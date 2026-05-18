"""
ALLOY IQ — Database Models (SQLAlchemy)
========================================
Defines all persistent entities: User, PredictionJob, InverseDesignJob.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id            = Column(String, primary_key=True, default=_uuid)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    hashed_pw     = Column(String(255), nullable=False)
    display_name  = Column(String(100), nullable=True)
    tier          = Column(String(20), default="free")   # free | pro | enterprise
    created_at    = Column(DateTime, default=datetime.utcnow)

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
    processing      = Column(JSON, nullable=True)           # {heat_treat_temp_C: …}

    # Output
    prediction      = Column(Float, nullable=True)
    lower_ci        = Column(Float, nullable=True)
    upper_ci        = Column(Float, nullable=True)
    confidence      = Column(Float, nullable=True)
    data_confidence = Column(String(10), nullable=True)    # high | moderate | low
    shap_data       = Column(JSON, nullable=True)          # waterfall + narrative
    render_url      = Column(String(500), nullable=True)   # Blender render path

    status          = Column(String(20), default="pending")  # pending | done | error
    error_msg       = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    completed_at    = Column(DateTime, nullable=True)

    user            = relationship("User", back_populates="predictions")


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

    status         = Column(String(20), default="pending")
    error_msg      = Column(Text, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    completed_at   = Column(DateTime, nullable=True)

    user           = relationship("User", back_populates="inverse_jobs")
