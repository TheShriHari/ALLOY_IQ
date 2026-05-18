"use client";

import { useState } from "react";
import { ChevronDown, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

import axios from "axios";

// ─── Constants ───────────────────────────────────────────────────────────────
const ALLOY_FAMILIES = ["steel", "hea", "aluminum"] as const;
type AlloySFamily = typeof ALLOY_FAMILIES[number];

const PROPERTIES: Record<AlloySFamily, string[]> = {
  steel:    ["yield_strength", "hardness", "fatigue_limit", "corrosion_pren", "fracture_toughness"],
  hea:      ["yield_strength", "hardness", "fatigue_limit", "corrosion_pren"],
  aluminum: ["yield_strength", "hardness", "fatigue_limit", "corrosion_pren"],
};

const ELEMENTS: Record<AlloySFamily, string[]> = {
  steel:    ["Fe", "Cr", "Ni", "Mo", "Mn", "C", "Si", "N", "Cu", "V", "Ti", "Nb"],
  hea:      ["Fe", "Cr", "Ni", "Co", "Al", "Ti", "V",  "Mo", "Cu", "Mn"],
  aluminum: ["Al", "Mg", "Si", "Cu", "Zn", "Mn", "Cr", "Ti"],
};

const PROP_UNITS: Record<string, string> = {
  yield_strength: "MPa", hardness: "HV", fatigue_limit: "MPa",
  corrosion_pren: "PREN", fracture_toughness: "MPa√m",
};

// ─── API call ───────────────────────────────
async function doPredict(payload: object) {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const [mechRes, expRes] = await Promise.all([
    axios.post(`${API_URL}/predict/mechanical`, payload),
    axios.post(`${API_URL}/predict/explain`, payload),
  ]);
  return { ...mechRes.data, shap: expRes.data.shap };
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function PredictPage() {
  const [family,     setFamily]     = useState<AlloySFamily>("steel");
  const [property,   setProperty]   = useState("yield_strength");
  const [comp,       setComp]       = useState<Record<string, number>>({ Fe: 0.70, Cr: 0.18, Ni: 0.05, Mo: 0.03, Mn: 0.02, N: 0.002 });
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState<any>(null);
  const [error,      setError]      = useState<string | null>(null);

  const total = Object.values(comp).reduce((a, b) => a + b, 0);
  const isValid = Math.abs(total - 1.0) < 0.02;

  function updateElement(el: string, val: number) {
    setComp(prev => ({ ...prev, [el]: isNaN(val) ? 0 : val }));
  }

  function removeElement(el: string) {
    setComp(prev => { const next = { ...prev }; delete next[el]; return next; });
  }

  function addElement(el: string) {
    if (!comp[el]) setComp(prev => ({ ...prev, [el]: 0 }));
  }

  async function handlePredict() {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await doPredict({ alloy_family: family, property, composition: comp });
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  const confidenceBadgeClass =
    result?.data_confidence === "high" ? "badge-high" :
    result?.data_confidence === "moderate" ? "badge-moderate" : "badge-low";

  return (
    <main style={{ paddingTop: 80 }}>
      <div className="container section" style={{ paddingTop: 40 }}>
        <h1 style={{ fontSize: "clamp(1.8rem, 3.5vw, 2.5rem)", fontWeight: 800, letterSpacing: "-0.03em", marginBottom: 8 }}>
          Property <span className="gradient-text">Prediction</span>
        </h1>
        <p style={{ color: "var(--color-muted)", fontSize: "0.95rem", marginBottom: 40 }}>
          Enter your alloy composition and get an instant ML-powered prediction with SHAP explainability.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, alignItems: "start" }}>
          {/* ── Input panel ─────────────────────────────────────────────── */}
          <div className="glass" style={{ padding: 32 }}>
            {/* Family select */}
            <div style={{ marginBottom: 24 }}>
              <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-muted)", letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>
                Alloy Family
              </label>
              <div style={{ display: "flex", gap: 8 }}>
                {ALLOY_FAMILIES.map(f => (
                  <button key={f} onClick={() => { setFamily(f); setProperty(PROPERTIES[f][0]); setResult(null); }}
                    style={{
                      flex: 1, padding: "9px 0", borderRadius: 8, cursor: "pointer", fontSize: "0.85rem", fontWeight: 600,
                      border: family === f ? "1px solid var(--color-primary)" : "1px solid var(--color-border)",
                      background: family === f ? "rgba(99,130,255,0.15)" : "var(--color-surface-2)",
                      color: family === f ? "var(--color-primary)" : "var(--color-muted)",
                      transition: "all 0.15s",
                    }}>
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Property select */}
            <div style={{ marginBottom: 24 }}>
              <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-muted)", letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>
                Target Property
              </label>
              <div style={{ position: "relative" }}>
                <select value={property} onChange={e => { setProperty(e.target.value); setResult(null); }}
                  className="input-field" style={{ appearance: "none", paddingRight: 36 }}>
                  {PROPERTIES[family].map(p => (
                    <option key={p} value={p}>{p.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}</option>
                  ))}
                </select>
                <ChevronDown size={16} style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: "var(--color-muted)", pointerEvents: "none" }} />
              </div>
            </div>

            {/* Composition inputs */}
            <div style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-muted)", letterSpacing: "0.05em", textTransform: "uppercase" }}>
                  Composition (weight fractions)
                </label>
                <span style={{
                  fontSize: "0.78rem", fontWeight: 700,
                  color: isValid ? "#00E577" : "var(--color-danger)",
                }}>
                  Σ = {total.toFixed(4)}
                </span>
              </div>

              {Object.entries(comp).map(([el, val]) => (
                <div key={el} style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "center" }}>
                  <span style={{
                    width: 36, textAlign: "center", fontWeight: 700, fontSize: "0.85rem",
                    color: "var(--color-primary)", flexShrink: 0,
                  }}>{el}</span>
                  <input
                    id={`el-${el}`}
                    type="number" step="0.001" min="0" max="1"
                    value={val}
                    onChange={e => updateElement(el, parseFloat(e.target.value))}
                    className="input-field" style={{ flex: 1 }}
                  />
                  <button onClick={() => removeElement(el)}
                    style={{ background: "none", border: "none", color: "var(--color-muted)", cursor: "pointer", fontSize: "1.1rem", lineHeight: 1 }}>
                    ×
                  </button>
                </div>
              ))}

              {/* Add element */}
              <div style={{ marginTop: 10 }}>
                <select onChange={e => { if (e.target.value) addElement(e.target.value); e.target.value = ""; }}
                  className="input-field" style={{ fontSize: "0.82rem" }}>
                  <option value="">+ Add element…</option>
                  {ELEMENTS[family].filter(el => !comp[el]).map(el => (
                    <option key={el} value={el}>{el}</option>
                  ))}
                </select>
              </div>
            </div>

            <button onClick={handlePredict} disabled={loading || !isValid}
              className="btn-glow"
              style={{ width: "100%", marginTop: 24, opacity: (loading || !isValid) ? 0.6 : 1 }}>
              {loading ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} /> Predicting…
              </span> : "Predict Property →"}
            </button>
            {!isValid && <p style={{ fontSize: "0.78rem", color: "var(--color-danger)", marginTop: 8, textAlign: "center" }}>
              Fractions must sum to 1.0 ± 0.02
            </p>}
          </div>

          {/* ── Results panel ───────────────────────────────────────────── */}
          <div>
            {!result && !error && !loading && (
              <div className="glass" style={{ padding: 48, textAlign: "center", color: "var(--color-muted)" }}>
                <Loader2 size={40} style={{ margin: "0 auto 16px", opacity: 0.3 }} />
                <p style={{ fontSize: "0.9rem" }}>Enter your composition and click Predict.</p>
              </div>
            )}

            {error && (
              <div className="glass" style={{ padding: 32, color: "var(--color-danger)", display: "flex", gap: 12, alignItems: "flex-start" }}>
                <AlertCircle size={20} style={{ flexShrink: 0, marginTop: 2 }} />
                <p>{error}</p>
              </div>
            )}

            {result && (
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                {/* Prediction card */}
                <div className="glass" style={{ padding: 28 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
                    <div>
                      <div style={{ fontSize: "0.78rem", color: "var(--color-muted)", fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 6 }}>
                        {property.replace(/_/g, " ")}
                      </div>
                      <div style={{ fontSize: "3rem", fontWeight: 800, letterSpacing: "-0.03em" }} className="gradient-text">
                        {result.prediction.toFixed(1)}
                        <span style={{ fontSize: "1.2rem", marginLeft: 6, color: "var(--color-muted)", fontWeight: 500 }}>
                          {result.unit}
                        </span>
                      </div>
                      <div style={{ fontSize: "0.82rem", color: "var(--color-muted)", marginTop: 4 }}>
                        90% CI: [{result.lower.toFixed(1)}, {result.upper.toFixed(1)}] {result.unit}
                      </div>
                    </div>
                    <span className={`badge ${confidenceBadgeClass}`}>
                      <CheckCircle2 size={12} /> {result.data_confidence} confidence
                    </span>
                  </div>

                  {/* Confidence bar */}
                  <div style={{ background: "var(--color-surface-2)", borderRadius: 4, height: 6, overflow: "hidden", marginBottom: 8 }}>
                    <div style={{
                      height: "100%", borderRadius: 4,
                      width: `${((result.prediction - result.lower) / (result.upper - result.lower + 0.01)) * 100}%`,
                      background: "linear-gradient(90deg, var(--color-primary), var(--color-accent))",
                    }} />
                  </div>
                </div>

                {/* SHAP waterfall */}
                <div className="glass" style={{ padding: 24 }}>
                  <h3 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: 16, color: "var(--color-text)" }}>
                    SHAP Feature Contributions
                  </h3>
                  <Plot
                    data={[{
                      type: "bar",
                      orientation: "h",
                      x: result.shap.waterfall.map((d: any) => d.shap),
                      y: result.shap.waterfall.map((d: any) => d.feature),
                      marker: {
                        color: result.shap.waterfall.map((d: any) => d.shap > 0 ? "#6382FF" : "#FF4D6A"),
                      },
                    }]}
                    layout={{
                      height: 280,
                      margin: { t: 8, b: 32, l: 40, r: 16 },
                      paper_bgcolor: "transparent",
                      plot_bgcolor: "transparent",
                      font: { color: "#6B7A9E", size: 11 },
                      xaxis: { gridcolor: "rgba(99,130,255,0.08)", zerolinecolor: "rgba(99,130,255,0.2)" },
                      yaxis: { gridcolor: "transparent" },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: "100%" }}
                  />
                </div>

                {/* Narrative */}
                <div className="glass" style={{
                  padding: 20,
                  borderLeft: "3px solid var(--color-primary)",
                  background: "rgba(99,130,255,0.04)",
                }}>
                  <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--color-primary)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    AI Narrative
                  </div>
                  <p style={{ fontSize: "0.88rem", color: "var(--color-text)", lineHeight: 1.7 }}>
                    {result.shap.narrative}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
        @media (max-width: 768px) {
          .predict-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </main>
  );
}
