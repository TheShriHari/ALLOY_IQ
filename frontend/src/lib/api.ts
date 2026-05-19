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
    yield_strength_mpa?:   PropertyPrediction;
    tensile_strength_mpa?: PropertyPrediction;
    hardness_hv?:          PropertyPrediction;
    elongation_pct?:       PropertyPrediction;
    [key: string]: PropertyPrediction | undefined;
  };
  corrosion_analysis: {
    pren_calculated: number;
    corrosion_grade: string;
    nace_guidance: string;
  };
  fatigue?: {
    fatigue_limit_mpa: number;
    fatigue_limit_lower: number;
    fatigue_limit_upper: number;
  };
  fracture_toughness?: {
    fracture_toughness_kic_mpa_sqrtm: number;
    ndt_guidance: string;
  };
  confidence_level?: number;
  data_confidence: "high" | "medium" | "low";
  inference_ms?: number;
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
    if (typeof window !== "undefined") {
      localStorage.removeItem("alloyiq_token");
      window.location.href = "/auth/login";
    }
    throw new Error("Session expired");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || err.message || `API error ${res.status}`);
  }
  return res.json();
}

// ── API methods ────────────────────────────────────────────────────

export const api = {
  // Auth
  async login(email: string, password: string): Promise<{ access_token: string }> {
    const params = new URLSearchParams();
    params.append("username", email);
    params.append("password", password);
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: params,
    });
    return handleResponse(res);
  },

  async register(email: string, password: string, name: string): Promise<{ access_token: string }> {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, display_name: name }),
    });
    return handleResponse(res);
  },

  // Prediction
  async predict(alloyFamily: string, property: string, composition: ElementComposition): Promise<PredictionResponse> {
    const res = await fetch(`${API_BASE}/predict/mechanical`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ alloy_family: alloyFamily, property, composition }),
    });
    return handleResponse<PredictionResponse>(res);
  },

  async explain(alloyFamily: string, property: string, composition: ElementComposition): Promise<ShapResponse> {
    const res = await fetch(`${API_BASE}/predict/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ alloy_family: alloyFamily, property, composition }),
    });
    return handleResponse<ShapResponse>(res);
  },

  async getPdp(alloyFamily: string, element: string, composition: ElementComposition): Promise<PdpResponse> {
    const res = await fetch(`${API_BASE}/explain/pdp`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ alloy_family: alloyFamily, element, composition }),
    });
    return handleResponse<PdpResponse>(res);
  },

  // History
  async getHistory(): Promise<Array<{ id: string; created_at: string; composition: Record<string, number>; predictions: PredictionResponse }>> {
    const res = await fetch(`${API_BASE}/history`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(res);
  },

  // Blender render
  async requestRender(composition: ElementComposition, predictions: PredictionResponse): Promise<{ job_id: string }> {
    const res = await fetch(`${API_BASE}/blender/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeader() },
      body: JSON.stringify({ composition, predictions }),
    });
    return handleResponse(res);
  },

  async pollRender(jobId: string): Promise<{ status: "queued" | "running" | "complete" | "failed"; image_url?: string }> {
    const res = await fetch(`${API_BASE}/blender/render/${jobId}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(res);
  },

  // WebSocket for inverse design (returns a function to call and an EventEmitter-like interface)
  connectOptimizer(
    alloyFamily: string,
    targets: OptimizationTarget[],
    constraints: Record<string, { min?: number; max?: number }>,
    onGeneration: (result: GenerationResult) => void,
    onComplete: (finalPareto: GenerationResult["pareto_front"]) => void,
    onError: (msg: string) => void,
  ): () => void {
    const ws = new WebSocket(`${WS_BASE}/ws/optimize`);

    ws.onopen = () => {
      ws.send(JSON.stringify({ alloy_family: alloyFamily, targets, constraints, n_generations: 100 }));
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
