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

from backend.db.models import Base, InverseDesignJob, PredictionJob, User, RenderJob
from backend.tasks.render_task import render_microstructure
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
    import bcrypt
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def hash_password(plain: str) -> str:
    import bcrypt
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


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
    property: str = "yield_strength_mpa"
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
    job_id: Optional[str] = None
    predictions: Dict[str, Dict[str, float]]
    confidence_level: float
    data_confidence: str
    corrosion_analysis: Optional[Dict[str, Any]] = None

class ExplainResponse(BaseModel):
    shap_values: Dict[str, float]
    narrative: str

class PDPRequest(BaseModel):
    element: str
    alloy_family: str
    composition: Dict[str, float]

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
    from backend.cache.redis_client import get_cached_features, set_cached_features
    # Check cache
    cached = get_cached_features(req.composition)
    if cached and "features" in cached:
        return pd.DataFrame(cached["features"])

    row = dict(req.composition)
    if req.processing:
        row.update(req.processing)
    df = pd.DataFrame([row])
    fe = FeatureEngineer(req.alloy_family)
    df_transformed = fe.transform(df)

    # Cache transformed features
    set_cached_features(req.composition, {"features": df_transformed.to_dict(orient="records")})

    return df_transformed



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
        result  = model_engine.predict(req.alloy_family, df_feat)
        
        # Corrosion Physics
        from backend.ml.corrosion_features import compute_corrosion_metrics
        pren_pred = result["predictions"].get("corrosion_pren", {}).get("mean", 0.0)
        corrosion = compute_corrosion_metrics(req.composition, pren_pred)

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction error: {e}")

    # Fallback/default property tracking
    target = req.property if req.property in result["predictions"] else "yield_strength_mpa"
    pred_val = result["predictions"].get(target, {}).get("mean", 0.0)
    low_val = result["predictions"].get(target, {}).get("lower", 0.0)
    high_val = result["predictions"].get(target, {}).get("upper", 0.0)

    job = PredictionJob(
        user_id         = current_user.id,
        alloy_family    = req.alloy_family,
        property_target = target,
        composition     = req.composition,
        processing      = req.processing,
        prediction      = pred_val,
        lower_ci        = low_val,
        upper_ci        = high_val,
        confidence      = result["confidence_level"],
        data_confidence = result["data_confidence"],
        shap_data       = result.get("shap_dicts", {}).get(target, {}),
        status          = "done",
        completed_at    = datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return MechanicalResponse(
        job_id          = job.id,
        predictions     = result["predictions"],
        confidence_level= result["confidence_level"],
        data_confidence = result["data_confidence"],
        corrosion_analysis = corrosion,
    )

@app.post("/predict/explain", response_model=ExplainResponse, tags=["Prediction"])
def predict_explain(
    req: PredictRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        df_feat = _build_feature_df(req)
        result  = model_engine.predict(req.alloy_family, df_feat)
        target = req.property if req.property in result["shap_dicts"] else "yield_strength_mpa"
        return ExplainResponse(
            shap_values=result["shap_dicts"].get(target, {}),
            narrative=result["narratives"].get(target, "Narrative unavailable.")
        )
    except Exception as e:
        logger.exception("Explain prediction failed")
        raise HTTPException(500, f"Explain error: {e}")

@app.post("/api/v1/explain/pdp", tags=["Explain"])
def get_pdp(req: PDPRequest, current_user: User = Depends(get_current_user)):
    try:
        from backend.ml.pdp import compute_pdp
        model_obj = model_engine.get_model(req.alloy_family)
        if not model_obj._stack:
            raise HTTPException(400, "Model not trained")
            
        scaler = model_obj._stack.named_steps["scaler"]
        result = compute_pdp(
            model=model_obj._stack,
            scaler=scaler,
            X_median=model_obj._X_median,
            X_lo=model_obj._X_lo,
            X_hi=model_obj._X_hi,
            feature_name=f"frac_{req.element}",
            feature_names=model_obj._feature_names,
        )
        return result
    except Exception as e:
        logger.exception("PDP failed")
        raise HTTPException(500, f"PDP error: {e}")

@app.get("/api/v1/model/versions", tags=["MLflow"])
def get_model_versions():
    import mlflow
    from backend.ml.mlflow_config import EXPERIMENT_NAME, setup_mlflow
    try:
        setup_mlflow()
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return []
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
        
        # convert df to list of dicts, avoiding NaNs
        res = []
        for _, row in runs.iterrows():
            res.append({
                "run_id": row.get("run_id"),
                "status": row.get("status"),
                "metrics": {k.replace("metrics.", ""): v for k, v in row.items() if k.startswith("metrics.") and pd.notna(v)},
                "params": {k.replace("params.", ""): v for k, v in row.items() if k.startswith("params.") and pd.notna(v)},
            })
        return res
    except Exception as e:
        logger.warning(f"MLflow fetch failed: {e}")
        return []


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


# ---------------------------------------------------------------------------
# Routes — Blender Render
# ---------------------------------------------------------------------------
class RenderRequest(BaseModel):
    composition: Dict[str, float]
    predictions: Dict[str, Any]

@app.post("/blender/render", tags=["Blender"])
def request_render(
    body: RenderRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    import uuid
    job_id = str(uuid.uuid4())
    
    # Save job to DB with status="queued"
    job = RenderJob(
        id=job_id,
        user_id=current_user.id,
        composition=body.composition,
        predictions=body.predictions,
        status="queued"
    )
    db.add(job)
    db.commit()

    # Dispatch Celery background task
    try:
        render_microstructure.delay(job_id, body.composition, body.predictions)
    except Exception as e:
        logger.exception("Failed to dispatch Celery rendering task")
        # Fallback to local background tasks or mark as failed
        job.status = "failed"
        db.commit()
        raise HTTPException(500, f"Background worker dispatch error: {e}")

    return {"job_id": job_id, "status": "queued"}


@app.get("/blender/render/{job_id}", tags=["Blender"])
def poll_render(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    job = db.query(RenderJob).filter(
        RenderJob.id == job_id,
        RenderJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(404, "Render job not found")
        
    return {
        "job_id": job.id,
        "status": job.status,
        "image_url": job.image_url
    }

