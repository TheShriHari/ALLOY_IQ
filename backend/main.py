"""
ALLOY IQ — FastAPI Application
================================
Exposes REST endpoints for forward prediction, inverse design,
microstructure visualization, job history, and auth.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models import Base, InverseDesignJob, PredictionJob, User
from backend.ml.model_engine import AlloyModelEngine
from backend.data.features import FeatureEngineer
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL   = os.getenv("DATABASE_URL", "sqlite:///./alloy_iq.db")
SECRET_KEY     = os.getenv("SECRET_KEY", "change-me-in-production-please")
ALGORITHM      = "HS256"
ACCESS_TTL_MIN = 60 * 24  # 24 hours

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

pwd_ctx      = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ---------------------------------------------------------------------------
# Lifespan: DB init + model engine warmup
# ---------------------------------------------------------------------------
model_engine = AlloyModelEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")
    logger.info("Model engine ready. Available cells: {}", model_engine.available_cells())
    yield
    logger.info("Shutting down ALLOY IQ.")


app = FastAPI(
    title="ALLOY IQ",
    description="ML-powered materials property prediction platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Auth utilities
# ---------------------------------------------------------------------------
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TTL_MIN)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)), db: Session = Depends(get_db)) -> User:
    # --- DEV BYPASS ---
    # Create or return a dummy user for local development if no token is provided
    if not token:
        dummy = db.query(User).filter(User.email == "dev@alloyiq.com").first()
        if not dummy:
            dummy = User(email="dev@alloyiq.com", hashed_pw="dummy", display_name="Dev User")
            db.add(dummy)
            db.commit()
            db.refresh(dummy)
        return dummy
    # ------------------

    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise cred_exc
    except JWTError:
        raise cred_exc
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise cred_exc
    return user


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PredictRequest(BaseModel):
    alloy_family: str        # steel | hea | aluminum
    property: str            # yield_strength | hardness | …
    composition: Dict[str, float]   # {element: weight_fraction}
    processing: Optional[Dict[str, float]] = None
    confidence: float = 0.90

    @field_validator("composition")
    @classmethod
    def composition_sums_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Composition fractions must sum to ~1.0 (got {total:.3f})")
        return {k: val / total for k, val in v.items()}

class MechanicalResponse(BaseModel):
    job_id: str
    prediction: float
    lower: float
    upper: float
    confidence: float
    data_confidence: str
    unit: str

class ExplainResponse(BaseModel):
    shap: Dict[str, Any]

class InverseRequest(BaseModel):
    alloy_family: str
    targets: Dict[str, List]      # {"yield_strength": [">", 900]}
    constraints: Optional[Dict[str, List[float]]] = None
    n_generations: int = 100
    pop_size: int = 200

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict] = None

class HistoryItem(BaseModel):
    job_id: str
    alloy_family: str
    property: str
    prediction: Optional[float]
    created_at: str
    status: str


# ---------------------------------------------------------------------------
# Property → unit mapping
# ---------------------------------------------------------------------------
PROPERTY_UNITS = {
    "yield_strength": "MPa",
    "hardness": "HV",
    "fatigue_limit": "MPa",
    "corrosion_pren": "PREN",
    "fracture_toughness": "MPa√m",
}


# ---------------------------------------------------------------------------
# Helper: build feature DataFrame from request
# ---------------------------------------------------------------------------
def _build_feature_df(req: PredictRequest) -> pd.DataFrame:
    row = dict(req.composition)
    if req.processing:
        row.update(req.processing)
    df = pd.DataFrame([row])
    fe = FeatureEngineer(req.alloy_family)
    return fe.transform(df)


# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------
@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    user = User(
        email=body.email,
        hashed_pw=hash_password(body.password),
        display_name=body.display_name,
    )
    db.add(user)
    db.commit()
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)


# ---------------------------------------------------------------------------
# Routes — Prediction
# ---------------------------------------------------------------------------
@app.post("/predict/mechanical", response_model=MechanicalResponse, tags=["Prediction"])
def predict_mechanical(
    req: PredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        df_feat = _build_feature_df(req)
        result  = model_engine.predict(req.alloy_family, req.property, df_feat)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction error: {e}")

    job = PredictionJob(
        user_id         = current_user.id,
        alloy_family    = req.alloy_family,
        property_target = req.property,
        composition     = req.composition,
        processing      = req.processing,
        prediction      = result["prediction"],
        lower_ci        = result["lower"],
        upper_ci        = result["upper"],
        confidence      = result["confidence"],
        data_confidence = result["data_confidence"],
        shap_data       = result["shap"],
        status          = "done",
        completed_at    = datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return MechanicalResponse(
        job_id          = job.id,
        prediction      = result["prediction"],
        lower           = result["lower"],
        upper           = result["upper"],
        confidence      = result["confidence"],
        data_confidence = result["data_confidence"],
        unit            = PROPERTY_UNITS.get(req.property, ""),
    )

@app.post("/predict/explain", response_model=ExplainResponse, tags=["Prediction"])
def predict_explain(
    req: PredictRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        df_feat = _build_feature_df(req)
        result  = model_engine.predict(req.alloy_family, req.property, df_feat)
        return ExplainResponse(shap=result["shap"])
    except Exception as e:
        logger.exception("Explain prediction failed")
        raise HTTPException(500, f"Explain error: {e}")


@app.get("/predict/{job_id}", tags=["Prediction"])
def get_prediction(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(PredictionJob).filter(
        PredictionJob.id == job_id,
        PredictionJob.user_id == current_user.id,
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ---------------------------------------------------------------------------
# Routes — Inverse Design
# ---------------------------------------------------------------------------
def _run_inverse_design(job_id: str, req: InverseRequest, db_url: str):
    """Background task: run GA, write result to DB."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.engines.inverse_design import InverseDesignEngine

    eng = create_engine(db_url, connect_args={"check_same_thread": False})
    Sess = sessionmaker(bind=eng)
    db = Sess()

    job = db.query(InverseDesignJob).filter(InverseDesignJob.id == job_id).first()
    if not job:
        return

    try:
        targets = {k: tuple(v) for k, v in req.targets.items()}
        constraints = {k: tuple(v) for k, v in (req.constraints or {}).items()}
        ide = InverseDesignEngine(model_engine, req.alloy_family)
        result = ide.optimize(
            targets=targets,
            constraints=constraints,
            n_generations=req.n_generations,
            pop_size=req.pop_size,
        )
        job.pareto_front  = result["pareto_front"]
        job.n_candidates  = result["n_candidates"]
        job.status        = "done"
        job.completed_at  = datetime.utcnow()
    except Exception as e:
        logger.exception("Inverse design failed for job {}", job_id)
        job.status    = "error"
        job.error_msg = str(e)

    db.commit()
    db.close()


@app.post("/inverse", response_model=JobStatusResponse, tags=["Inverse Design"])
def inverse_design(
    req: InverseRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = InverseDesignJob(
        user_id      = current_user.id,
        alloy_family = req.alloy_family,
        targets      = req.targets,
        constraints  = req.constraints,
        n_generations= req.n_generations,
        pop_size     = req.pop_size,
        status       = "pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_inverse_design, job.id, req, DATABASE_URL)
    return JobStatusResponse(job_id=job.id, status="pending")


@app.get("/inverse/{job_id}", response_model=JobStatusResponse, tags=["Inverse Design"])
def get_inverse_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(InverseDesignJob).filter(
        InverseDesignJob.id == job_id,
        InverseDesignJob.user_id == current_user.id,
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        result={
            "pareto_front": job.pareto_front,
            "n_candidates": job.n_candidates,
            "objective_axes": list(job.targets.keys()) if job.targets else [],
        } if job.status == "done" else None,
    )


# ---------------------------------------------------------------------------
# Routes — History
# ---------------------------------------------------------------------------
@app.get("/history", response_model=List[HistoryItem], tags=["History"])
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 50,
):
    jobs = (
        db.query(PredictionJob)
        .filter(PredictionJob.user_id == current_user.id)
        .order_by(PredictionJob.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        HistoryItem(
            job_id       = j.id,
            alloy_family = j.alloy_family,
            property     = j.property_target,
            prediction   = j.prediction,
            created_at   = j.created_at.isoformat(),
            status       = j.status,
        )
        for j in jobs
    ]


# ---------------------------------------------------------------------------
# Routes — Health & Meta
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "service": "alloy-iq-api"}

@app.get("/cells", tags=["Meta"])
def list_cells():
    return {"cells": model_engine.available_cells()}
