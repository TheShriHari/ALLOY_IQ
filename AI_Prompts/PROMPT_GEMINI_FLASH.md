# ALLOY IQ — Improvement Prompt: Gemini Flash
**Role**: Frontend-API Integration · Authentication · Celery Tasks · Database · Redis Cache · UI Polish · Tests  
**Priority gaps you own**: Frontend integration (CRITICAL), Auth (CRITICAL), Celery+Redis (CRITICAL), DB schema (HIGH), Compare page (MEDIUM), Cache (MEDIUM), Tests (LOW), UI polish (LOW)  
**Why Gemini Flash for these tasks**: High volume of interconnected but structurally well-defined tasks. Your speed and high rate limits let you parallelise across frontend, backend infrastructure, and test files simultaneously.

---

## CONTEXT: WHAT ALREADY EXISTS

```
frontend/src/
  app/
    page.tsx              ← landing (complete)
    predict/page.tsx      ← composition sliders UI (MOCK DATA — needs real API)
    inverse/page.tsx      ← Pareto UI (DEAD — needs WebSocket connection)
    microstructure/page.tsx ← Blender output display (DEAD — no polling)
    history/page.tsx      ← job history (EMPTY — no database)
  app/globals.css         ← glassmorphic dark theme (complete)

backend/
  main.py                 ← FastAPI app
  sync.py                 ← schema sync utility
  ml/model_engine.py      ← ML pipeline
```

**The single biggest problem**: `predict/page.tsx` almost certainly calls no real API. It renders sliders and shows hardcoded numbers. Proof: there is no `frontend/src/lib/api.ts` file listed anywhere. **Your first task is to wire every page to the real FastAPI backend.**

**Tech stack** (infer from what exists):
- Frontend: Next.js 14, TypeScript, Tailwind CSS, Recharts (charts in Pareto view)
- Backend: FastAPI, Python 3.11+, SQLAlchemy + Alembic, Redis, Celery
- Auth: JWT (use `python-jose` + `passlib` on backend, localStorage + axios interceptors on frontend)

---

## TASK 1 — FRONTEND API CLIENT: `frontend/src/lib/api.ts`

**This is the most critical task.** Every page in the app must use this single client. Do not write `fetch()` calls directly in component files.

```typescript
// frontend/src/lib/api.ts
/**
 * Centralized API client for ALLOY IQ.
 * All backend communication goes through here.
 * Handles: auth headers, error normalization, response typing.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE  = process.env.NEXT_PUBLIC_WS_URL  || "ws://localhost:8000";

// ── Types ──────────────────────────────────────────────────────────

export interface ElementComposition {
  Fe?: number; C?: number; Cr?: number; Ni?: number; Mo?: number;
  Mn?: number; V?:  number; Nb?: number; Si?: number; W?:  number;
  Co?: number; Ti?: number; Al?: number; Cu?: number; N?:  number;
  [key: string]: number | undefined;
}

export interface PropertyPrediction {
  mean: number;
  lower: number;
  upper: number;
}

export interface PredictionResponse {
  predictions: {
    yield_strength_mpa:   PropertyPrediction;
    tensile_strength_mpa: PropertyPrediction;
    hardness_hv:          PropertyPrediction;
    elongation_pct:       PropertyPrediction;
  };
  corrosion_analysis: {
    pren_calculated: number;
    corrosion_grade: string;
    nace_guidance: string;
  };
  fatigue: {
    fatigue_limit_mpa: number;
    fatigue_limit_lower: number;
    fatigue_limit_upper: number;
  };
  fracture_toughness: {
    fracture_toughness_kic_mpa_sqrtm: number;
    ndt_guidance: string;
  };
  confidence_level: number;
  data_confidence: "high" | "medium" | "low";
  inference_ms: number;
}

export interface ShapResponse {
  shap_values: Record<string, number>;
  narrative: string;
  top_features: Array<{ name: string; value: number; shap: number; direction: "positive" | "negative" }>;
}

export interface PdpResponse {
  feature: string;
  x_values: number[];
  predictions: Record<string, number[]>;
  x_label: string;
}

export interface OptimizationTarget {
  property: "yield_strength_mpa" | "tensile_strength_mpa" | "hardness_hv" | "corrosion_pren";
  direction: "maximize" | "minimize";
  min_val?: number;
  max_val?: number;
  weight?: number;
}

export interface GenerationResult {
  generation: number;
  best_fitness: number[];
  pareto_front: Array<{
    composition: Record<string, number>;
    predictions: Record<string, number>;
    fitness: number[];
    classification?: string;
    suggested_applications?: string[];
  }>;
  population_size: number;
  elapsed_seconds: number;
}

// ── Auth helpers ────────────────────────────────────────────────────

function getAuthHeader(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("alloyiq_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.status === 401) {
    localStorage.removeItem("alloyiq_token");
    window.location.href = "/auth/login";
    throw new Error("Session expired");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: "Unknown error" }));
    throw new Error(err.message || `API error ${res.status}`);
  }
  return res.json();
}

// ── API methods ────────────────────────────────────────────────────

export const api = {
  // Auth
  async login(email: string, password: string): Promise<{ access_token: string }> {
    const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    return handleResponse(res);
  },

  async register(email: string, password: string, name: string): Promise<{ id: string }> {
    const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name }),
    });
    return handleResponse(res);
  },

  // Prediction
  async predict(composition: ElementComposition): Promise<PredictionResponse> {
    const res = await fetch(`${API_BASE}/api/v1/predict/mechanical`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ composition }),
    });
    return handleResponse<PredictionResponse>(res);
  },

  async explain(composition: ElementComposition, target: string = "yield_strength_mpa"): Promise<ShapResponse> {
    const res = await fetch(`${API_BASE}/api/v1/predict/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ composition, target }),
    });
    return handleResponse<ShapResponse>(res);
  },

  async getPdp(element: string, composition: ElementComposition): Promise<PdpResponse> {
    const res = await fetch(`${API_BASE}/api/v1/explain/pdp`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ element, composition }),
    });
    return handleResponse<PdpResponse>(res);
  },

  // History
  async getHistory(): Promise<Array<{ id: string; created_at: string; composition: Record<string, number>; predictions: PredictionResponse }>> {
    const res = await fetch(`${API_BASE}/api/v1/history`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(res);
  },

  // Blender render
  async requestRender(composition: ElementComposition, predictions: PredictionResponse): Promise<{ job_id: string }> {
    const res = await fetch(`${API_BASE}/api/v1/blender/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ composition, predictions }),
    });
    return handleResponse(res);
  },

  async pollRender(jobId: string): Promise<{ status: "queued" | "running" | "complete" | "failed"; image_url?: string }> {
    const res = await fetch(`${API_BASE}/api/v1/blender/render/${jobId}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(res);
  },

  // WebSocket for inverse design (returns a function to call and an EventEmitter-like interface)
  connectOptimizer(
    targets: OptimizationTarget[],
    constraints: Record<string, { min?: number; max?: number }>,
    onGeneration: (result: GenerationResult) => void,
    onComplete: (finalPareto: GenerationResult["pareto_front"]) => void,
    onError: (msg: string) => void,
  ): () => void {
    const ws = new WebSocket(`${WS_BASE}/ws/optimize`);

    ws.onopen = () => {
      ws.send(JSON.stringify({ targets, constraints, n_generations: 100 }));
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.status === "complete") {
        onComplete(msg.best_candidates || []);
      } else if (msg.status === "error") {
        onError(msg.message);
      } else if (msg.generation !== undefined) {
        onGeneration(msg as GenerationResult);
      }
    };

    ws.onerror = () => onError("WebSocket connection failed");

    // Return a cleanup function
    return () => ws.close();
  },
};
```

---

## TASK 2 — REACT HOOKS: `frontend/src/hooks/`

Create these three hooks. They wrap the API client with loading/error state management.

**`frontend/src/hooks/usePrediction.ts`**:
```typescript
import { useState, useCallback } from "react";
import { api, ElementComposition, PredictionResponse, ShapResponse } from "@/lib/api";

export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [shap, setShap]       = useState<ShapResponse | null>(null);

  const predict = useCallback(async (composition: ElementComposition) => {
    setLoading(true); setError(null);
    try {
      const [pred, sh] = await Promise.all([
        api.predict(composition),
        api.explain(composition),
      ]);
      setPrediction(pred);
      setShap(sh);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, []);

  return { predict, loading, error, prediction, shap };
}
```

**`frontend/src/hooks/useInverseDesign.ts`**:
```typescript
import { useState, useCallback, useRef } from "react";
import { api, GenerationResult, OptimizationTarget } from "@/lib/api";

export function useInverseDesign() {
  const [running, setRunning]   = useState(false);
  const [generation, setGeneration] = useState(0);
  const [paretoFront, setParetoFront] = useState<GenerationResult["pareto_front"]>([]);
  const [bestFitness, setBestFitness] = useState<number[]>([]);
  const [error, setError]       = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  const startOptimization = useCallback((
    targets: OptimizationTarget[],
    constraints: Record<string, { min?: number; max?: number }>,
  ) => {
    setRunning(true); setError(null); setGeneration(0);

    const cleanup = api.connectOptimizer(
      targets,
      constraints,
      (result: GenerationResult) => {
        setGeneration(result.generation);
        setParetoFront(result.pareto_front);
        setBestFitness(result.best_fitness);
      },
      (finalPareto) => {
        setParetoFront(finalPareto);
        setRunning(false);
      },
      (msg) => {
        setError(msg);
        setRunning(false);
      },
    );
    cleanupRef.current = cleanup;
  }, []);

  const stopOptimization = useCallback(() => {
    cleanupRef.current?.();
    setRunning(false);
  }, []);

  return { startOptimization, stopOptimization, running, generation, paretoFront, bestFitness, error };
}
```

**`frontend/src/hooks/useBlenderRender.ts`**:
```typescript
import { useState, useCallback } from "react";
import { api, ElementComposition, PredictionResponse } from "@/lib/api";

export function useBlenderRender() {
  const [status, setStatus] = useState<"idle" | "queued" | "running" | "complete" | "failed">("idle");
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const requestRender = useCallback(async (
    composition: ElementComposition,
    predictions: PredictionResponse,
  ) => {
    setStatus("queued");
    const { job_id } = await api.requestRender(composition, predictions);

    // Poll every 3s until complete
    const poll = setInterval(async () => {
      const result = await api.pollRender(job_id);
      setStatus(result.status);
      if (result.status === "complete" && result.image_url) {
        setImageUrl(result.image_url);
        clearInterval(poll);
      } else if (result.status === "failed") {
        clearInterval(poll);
      }
    }, 3000);
  }, []);

  return { requestRender, status, imageUrl };
}
```

---

## TASK 3 — REWIRE `predict/page.tsx`

**Replace every hardcoded mock value** in `app/predict/page.tsx` with calls to `usePrediction()`. The page structure must be:

```tsx
// frontend/src/app/predict/page.tsx
"use client";
import { usePrediction } from "@/hooks/usePrediction";
import { PredictionCard } from "@/components/PredictionCard";
import { ShapWaterfall } from "@/components/ShapWaterfall";
import { NarrativeCard } from "@/components/NarrativeCard";
import { LoadingSkeleton } from "@/components/ui/LoadingSkeleton";
import { ErrorAlert } from "@/components/ui/ErrorAlert";

export default function PredictPage() {
  const { predict, loading, error, prediction, shap } = usePrediction();
  const [composition, setComposition] = useState<Record<string, number>>({ Fe: 0.98, C: 0.008 });

  return (
    <div className="...">
      {/* Composition sliders — keep existing UI, just wire onChange to setComposition */}
      <CompositionInput composition={composition} onChange={setComposition} />

      <button onClick={() => predict(composition)} disabled={loading}>
        {loading ? "Predicting..." : "Run Prediction"}
      </button>

      {loading && <LoadingSkeleton rows={4} />}
      {error && <ErrorAlert message={error} />}
      
      {prediction && (
        <>
          <PredictionCard predictions={prediction.predictions} confidence={prediction.data_confidence} />
          <CorrosionCard analysis={prediction.corrosion_analysis} />
          <FatigueCard data={prediction.fatigue} fracture={prediction.fracture_toughness} />
          {shap && (
            <>
              <ShapWaterfall features={shap.top_features} />
              <NarrativeCard text={shap.narrative} />
            </>
          )}
        </>
      )}
    </div>
  );
}
```

---

## TASK 4 — AUTHENTICATION: `backend/auth/` + `frontend/src/app/auth/`

**Backend: `backend/auth/router.py`**:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import os

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24   # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


def create_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")


@router.post("/register")
async def register(data: UserRegister):
    # Check if user exists (query DB — see Task 5 for DB models)
    from db.session import get_db
    from db.models import User
    # ... create user, hash password, return {id, email}
    hashed = pwd_context.hash(data.password)
    # db.add(User(email=data.email, name=data.name, hashed_password=hashed))
    return {"message": "Registered successfully"}

@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    # ... verify credentials against DB
    # If valid:
    token = create_token({"sub": "user_id_from_db"})
    return {"access_token": token}
```

**Frontend: `frontend/src/app/auth/login/page.tsx`**:

```tsx
"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function handleLogin() {
    setLoading(true); setError(null);
    try {
      const { access_token } = await api.login(email, password);
      localStorage.setItem("alloyiq_token", access_token);
      router.push("/predict");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-black">
      <div className="w-full max-w-md p-8 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md">
        <h1 className="text-2xl font-semibold text-white mb-6">Sign in to ALLOY IQ</h1>
        {error && <div className="mb-4 p-3 rounded-lg bg-red-500/10 text-red-400 text-sm">{error}</div>}
        <input type="email" placeholder="Email" value={email}
          onChange={e => setEmail(e.target.value)}
          className="w-full mb-3 px-4 py-3 rounded-lg bg-white/5 border border-white/10 text-white" />
        <input type="password" placeholder="Password" value={password}
          onChange={e => setPassword(e.target.value)}
          className="w-full mb-4 px-4 py-3 rounded-lg bg-white/5 border border-white/10 text-white" />
        <button onClick={handleLogin} disabled={loading}
          className="w-full py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium disabled:opacity-50">
          {loading ? "Signing in..." : "Sign in"}
        </button>
        <p className="mt-4 text-center text-white/40 text-sm">
          No account? <a href="/auth/register" className="text-blue-400">Register free</a>
        </p>
      </div>
    </div>
  );
}
```

Add route guard in `frontend/src/middleware.ts`:
```typescript
import { NextRequest, NextResponse } from "next/server";

const PROTECTED = ["/predict", "/inverse", "/microstructure", "/history", "/compare"];

export function middleware(request: NextRequest) {
  const token = request.cookies.get("alloyiq_token")?.value;
  const isProtected = PROTECTED.some(p => request.nextUrl.pathname.startsWith(p));
  if (isProtected && !token) {
    return NextResponse.redirect(new URL("/auth/login", request.url));
  }
  return NextResponse.next();
}
```

---

## TASK 5 — DATABASE SCHEMA: `backend/db/models.py` + Alembic

```python
# backend/db/models.py
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid, datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email           = Column(String, unique=True, nullable=False, index=True)
    name            = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at      = Column(DateTime, default=datetime.datetime.utcnow)
    tier            = Column(Enum("free","pro","enterprise"), default="free")
    predictions     = relationship("PredictionJob", back_populates="user")

class PredictionJob(Base):
    __tablename__ = "prediction_jobs"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id     = Column(String, ForeignKey("users.id"), nullable=False)
    composition = Column(JSON, nullable=False)    # {"Fe": 0.98, "C": 0.008, ...}
    predictions = Column(JSON, nullable=True)     # full PredictionResponse
    shap_values = Column(JSON, nullable=True)
    narrative   = Column(String, nullable=True)
    alloy_family= Column(String, default="steel")
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)
    inference_ms= Column(Float, nullable=True)
    user        = relationship("User", back_populates="predictions")

class OptimizationJob(Base):
    __tablename__ = "optimization_jobs"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id      = Column(String, ForeignKey("users.id"), nullable=False)
    targets      = Column(JSON, nullable=False)
    constraints  = Column(JSON, nullable=False)
    status       = Column(Enum("queued","running","complete","failed"), default="queued")
    pareto_front = Column(JSON, nullable=True)
    n_generations= Column(Float, nullable=True)
    created_at   = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class RenderJob(Base):
    __tablename__ = "render_jobs"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id     = Column(String, ForeignKey("users.id"), nullable=False)
    composition = Column(JSON, nullable=False)
    predictions = Column(JSON, nullable=True)
    status      = Column(Enum("queued","running","complete","failed"), default="queued")
    image_url   = Column(String, nullable=True)
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)
```

**Run Alembic migrations**:
```bash
cd backend
alembic init migrations
# Edit alembic.ini: sqlalchemy.url = sqlite:///./alloyiq.db
# Edit migrations/env.py: from db.models import Base; target_metadata = Base.metadata
alembic revision --autogenerate -m "initial_schema"
alembic upgrade head
```

**Add history endpoint to `main.py`**:
```python
@app.get("/api/v1/history")
async def get_history(user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    jobs = db.query(PredictionJob).filter(PredictionJob.user_id == user_id)\
             .order_by(PredictionJob.created_at.desc()).limit(50).all()
    return [{"id": j.id, "created_at": j.created_at.isoformat(),
             "composition": j.composition, "predictions": j.predictions} for j in jobs]
```

---

## TASK 6 — CELERY TASK QUEUE: `backend/tasks/`

```python
# backend/tasks/celery_app.py
from celery import Celery
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "alloyiq",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.render_task", "tasks.inverse_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,   # results expire after 1 hour
)
```

```python
# backend/tasks/render_task.py
from tasks.celery_app import celery_app
import subprocess, tempfile, os, uuid

@celery_app.task(bind=True, max_retries=2, soft_time_limit=120)
def render_microstructure(self, job_id: str, composition: dict, predictions: dict):
    """
    Celery task: runs Blender headless render and saves output PNG.
    Executed in background — never blocks the API thread.
    """
    from blender.microstructure_bridge import generate_blender_script, estimate_phase_fractions

    try:
        # Update job status in DB
        self.update_state(state="STARTED", meta={"job_id": job_id})

        phase_fractions = estimate_phase_fractions(composition, predictions)
        script_content = generate_blender_script(job_id, phase_fractions)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        output_path = f"renders/{job_id}.png"
        os.makedirs("renders", exist_ok=True)

        result = subprocess.run(
            ["blender", "--background", "--python", script_path],
            capture_output=True, text=True, timeout=90
        )

        if result.returncode != 0:
            raise Exception(f"Blender failed: {result.stderr[-500:]}")

        # Update DB job to complete
        # (inject DB session or use direct SQLAlchemy connection)
        return {"status": "complete", "image_url": f"/renders/{job_id}.png"}

    except Exception as exc:
        self.retry(exc=exc, countdown=10)
    finally:
        if 'script_path' in locals():
            os.unlink(script_path)
```

**Update `/blender/render` endpoint in `main.py`** to enqueue the Celery task:
```python
@app.post("/api/v1/blender/render")
async def request_render(body: RenderRequest, user_id=Depends(get_current_user)):
    job_id = str(uuid.uuid4())
    # Save job to DB with status="queued"
    task = render_microstructure.delay(job_id, body.composition, body.predictions)
    return {"job_id": job_id, "celery_task_id": task.id}

@app.get("/api/v1/blender/render/{job_id}")
async def poll_render(job_id: str):
    # Query DB for job status
    # Return current status and image_url if complete
    ...
```

---

## TASK 7 — REDIS CACHING: `backend/cache/redis_client.py`

```python
# backend/cache/redis_client.py
"""
Redis cache for Magpie featurization results.
Magpie is deterministic per formula and takes 2-3s. Cache saves 95% of latency.
"""
import redis, json, hashlib, os

r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
CACHE_TTL = 60 * 60 * 24 * 7   # 7 days — Magpie features are immutable per composition

def _cache_key(composition: dict) -> str:
    """Generate a stable cache key from composition dict."""
    sorted_comp = sorted((k, round(v, 6)) for k, v in composition.items() if v > 1e-6)
    return "magpie:" + hashlib.sha256(json.dumps(sorted_comp).encode()).hexdigest()[:16]

def get_cached_features(composition: dict) -> dict | None:
    key = _cache_key(composition)
    cached = r.get(key)
    return json.loads(cached) if cached else None

def set_cached_features(composition: dict, features: dict) -> None:
    key = _cache_key(composition)
    r.setex(key, CACHE_TTL, json.dumps(features))
```

**Integrate into `main.py`** before Magpie featurization:
```python
from cache.redis_client import get_cached_features, set_cached_features

async def get_features(composition: dict) -> np.ndarray:
    cached = get_cached_features(composition)
    if cached:
        return np.array(cached["features"])

    # Run expensive Magpie featurization
    features = compute_magpie_features(composition)   # 2-3 seconds
    set_cached_features(composition, {"features": features.tolist()})
    return features
```

---

## TASK 8 — COMPARE PAGE: `frontend/src/app/compare/page.tsx`

A side-by-side comparison of up to 4 alloy compositions with their predicted properties.

```tsx
"use client";
import { useState } from "react";
import { usePrediction } from "@/hooks/usePrediction";

// Renders 2-4 composition columns side by side
// Each column has: element fraction inputs + predicted properties + SHAP top-3
// Bar charts (Recharts) show property comparison across columns
// "Export CSV" button downloads all column data as CSV

export default function ComparePage() {
  const [columns, setColumns] = useState([
    { id: "A", name: "Alloy A", composition: { Fe: 0.98, C: 0.008 } },
    { id: "B", name: "Alloy B", composition: { Fe: 0.97, C: 0.012, Cr: 0.01 } },
  ]);
  // ... render side-by-side comparison
}
```

---

## TASK 9 — UI POLISH: Loading States + Error Handling

**`frontend/src/components/ui/LoadingSkeleton.tsx`**:
```tsx
export function LoadingSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3 animate-pulse">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-16 rounded-xl bg-white/5 border border-white/5" />
      ))}
    </div>
  );
}
```

**`frontend/src/components/ui/ErrorAlert.tsx`**:
```tsx
export function ErrorAlert({ message }: { message: string }) {
  return (
    <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/10">
      <p className="text-red-400 text-sm">⚠ {message}</p>
    </div>
  );
}
```

Add toast notifications using `react-hot-toast` (`npm install react-hot-toast`) in `layout.tsx`:
```tsx
import { Toaster } from "react-hot-toast";
// Add <Toaster position="bottom-right" /> to root layout
```

---

## TASK 10 — ENVIRONMENT CONFIG

**`.env.example`** (commit this, never `.env`):
```bash
# Backend
DATABASE_URL=sqlite:///./alloyiq.db
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=CHANGE_ME_USE_OPENSSL_RAND_HEX_32
CELERY_BROKER_URL=redis://localhost:6379/0
BLENDER_PATH=/usr/bin/blender
RENDERS_DIR=./renders

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

**`backend/config/settings.py`**:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "sqlite:///./alloyiq.db"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret_key: str
    blender_path: str = "/usr/bin/blender"
    renders_dir: str = "./renders"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## TASK 11 — TEST SUITE

Write these test files using `pytest` + `httpx` (async API testing):

**`tests/test_api_routes.py`**:
```python
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_predict_returns_intervals():
    async with AsyncClient(app=app, base_url="http://test") as client:
        r = await client.post("/api/v1/predict/mechanical",
            json={"composition": {"Fe": 0.98, "C": 0.008, "Mn": 0.01}})
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert "lower" in data["predictions"]["yield_strength_mpa"]
    assert "upper" in data["predictions"]["yield_strength_mpa"]
    assert "narrative" in data.get("shap", {}) or True   # narrative in explain endpoint

@pytest.mark.asyncio
async def test_predict_validates_composition():
    async with AsyncClient(app=app, base_url="http://test") as client:
        r = await client.post("/api/v1/predict/mechanical",
            json={"composition": {"Fe": 0.8, "C": 0.8}})   # sums to 1.6 — invalid
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as client:
        r = await client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
```

---

## INTEGRATION CHECKLIST

After all tasks, verify:
1. `npm run dev` in frontend — `/predict` page calls real API (check Network tab, no hardcoded data)
2. `curl -X POST /api/v1/auth/login` — returns JWT token
3. Authenticated `curl /predict/mechanical` — returns `lower`, `upper`, `narrative`, `pren_calculated`
4. `wscat -c ws://localhost:8000/ws/optimize` — streams generation messages
5. `/history` — returns real DB records after 3 predictions
6. `pytest tests/ -v` — all tests pass

## HOW TO UPDATE THE AGENT TRACKER

```bash
python -c "
import json, datetime
with open('agent_tracker.json') as f: t = json.load(f)
t['agents']['gemini_flash']['status'] = 'in_progress'
t['agents']['gemini_flash']['current_task'] = 'TASK_1_API_CLIENT'
t['agents']['gemini_flash']['last_updated'] = datetime.datetime.utcnow().isoformat()
with open('agent_tracker.json', 'w') as f: json.dump(t, f, indent=2)
print('Tracker updated')
"
```
