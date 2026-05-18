"use client";

import { useState } from "react";
import { Loader2, Plus, Trash2 } from "lucide-react";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

import axios from "axios";

type Operator = ">" | "<" | ">=" | "<=";
interface TargetRow { property: string; operator: Operator; value: number; }

const ALL_PROPERTIES = [
  "yield_strength", "hardness", "fatigue_limit", "corrosion_pren", "fracture_toughness"
];
const PROP_LABELS: Record<string, string> = {
  yield_strength: "Yield Strength (MPa)",
  hardness: "Hardness (HV)",
  fatigue_limit: "Fatigue Limit (MPa)",
  corrosion_pren: "PREN",
  fracture_toughness: "Fracture Toughness (MPa√m)",
};
const ALLOY_FAMILIES = ["steel", "hea", "aluminum"] as const;

async function doInverseDesign(payload: object) {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  // 1. Start job
  const { data: startData } = await axios.post(`${API_URL}/inverse`, payload);
  const jobId = startData.job_id;
  
  // 2. Poll until done
  while (true) {
    await new Promise(r => setTimeout(r, 1000));
    const { data: pollData } = await axios.get(`${API_URL}/inverse/${jobId}`);
    if (pollData.status === "done") {
      return pollData.result;
    }
    if (pollData.status === "error") {
      throw new Error("Inverse design background job failed.");
    }
  }
}

export default function InversePage() {
  const [family,   setFamily]   = useState<"steel"|"hea"|"aluminum">("steel");
  const [targets,  setTargets]  = useState<TargetRow[]>([
    { property: "yield_strength", operator: ">", value: 900 },
    { property: "corrosion_pren", operator: ">", value: 35 },
  ]);
  const [loading,  setLoading]  = useState(false);
  const [result,   setResult]   = useState<any>(null);
  const [selected, setSelected] = useState<number | null>(null);

  function addTarget() {
    const avail = ALL_PROPERTIES.find(p => !targets.find(t => t.property === p));
    if (avail) setTargets(prev => [...prev, { property: avail, operator: ">", value: 0 }]);
  }
  function removeTarget(i: number) {
    setTargets(prev => prev.filter((_, idx) => idx !== i));
  }
  function updateTarget(i: number, key: keyof TargetRow, val: any) {
    setTargets(prev => prev.map((t, idx) => idx === i ? { ...t, [key]: val } : t));
  }

  async function handleOptimize() {
    setLoading(true); setResult(null); setSelected(null);
    try {
      const targetsPayload: any = {};
      targets.forEach(t => { targetsPayload[t.property] = [t.operator, t.value]; });
      
      const res = await doInverseDesign({
        alloy_family: family,
        targets: targetsPayload,
        n_generations: 50,
        pop_size: 100
      });
      setResult(res);
    } catch(e) {
      console.error(e);
      alert("Optimization failed.");
    } finally {
      setLoading(false);
    }
  }

  const axes = result?.objective_axes ?? [];

  return (
    <main style={{ paddingTop: 80 }}>
      <div className="container section" style={{ paddingTop: 40 }}>
        <h1 style={{ fontSize: "clamp(1.8rem, 3.5vw, 2.5rem)", fontWeight: 800, letterSpacing: "-0.03em", marginBottom: 8 }}>
          Inverse <span className="gradient-text">Design</span>
        </h1>
        <p style={{ color: "var(--color-muted)", fontSize: "0.95rem", marginBottom: 40 }}>
          Specify property targets and let NSGA-II discover the Pareto-optimal alloy candidates.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 32, alignItems: "start" }}>
          {/* Input */}
          <div className="glass" style={{ padding: 28 }}>
            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-muted)", letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>
                Alloy Family
              </label>
              <div style={{ display: "flex", gap: 6 }}>
                {ALLOY_FAMILIES.map(f => (
                  <button key={f} onClick={() => setFamily(f)}
                    style={{
                      flex: 1, padding: "8px 0", borderRadius: 8, cursor: "pointer", fontSize: "0.8rem", fontWeight: 600,
                      border: family === f ? "1px solid var(--color-primary)" : "1px solid var(--color-border)",
                      background: family === f ? "rgba(99,130,255,0.15)" : "var(--color-surface-2)",
                      color: family === f ? "var(--color-primary)" : "var(--color-muted)",
                    }}>
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-muted)", letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 12 }}>
              Property Targets
            </label>

            {targets.map((t, i) => (
              <div key={i} style={{ marginBottom: 12, padding: 14, background: "var(--color-surface-2)", borderRadius: 10, border: "1px solid var(--color-border)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <select value={t.property} onChange={e => updateTarget(i, "property", e.target.value)}
                    className="input-field" style={{ flex: 1, marginRight: 8, fontSize: "0.82rem" }}>
                    {ALL_PROPERTIES.map(p => <option key={p} value={p}>{PROP_LABELS[p]}</option>)}
                  </select>
                  <button onClick={() => removeTarget(i)}
                    style={{ background: "none", border: "none", cursor: "pointer", color: "var(--color-danger)" }}>
                    <Trash2 size={15} />
                  </button>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <select value={t.operator} onChange={e => updateTarget(i, "operator", e.target.value)}
                    className="input-field" style={{ width: 60, fontSize: "0.85rem" }}>
                    {([">","<",">=","<="] as Operator[]).map(op => <option key={op} value={op}>{op}</option>)}
                  </select>
                  <input type="number" value={t.value} onChange={e => updateTarget(i, "value", parseFloat(e.target.value))}
                    className="input-field" style={{ flex: 1 }} />
                </div>
              </div>
            ))}

            <button onClick={addTarget} disabled={targets.length >= ALL_PROPERTIES.length}
              className="btn-ghost" style={{ width: "100%", marginBottom: 16, display: "flex", alignItems: "center", justifyContent: "center", gap: 6, fontSize: "0.85rem" }}>
              <Plus size={15} /> Add Target
            </button>

            <button onClick={handleOptimize} disabled={loading || targets.length === 0}
              className="btn-glow" style={{ width: "100%", opacity: (loading || targets.length === 0) ? 0.6 : 1 }}>
              {loading ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} /> Running NSGA-II…
              </span> : "Optimize Composition →"}
            </button>

            {loading && (
              <p style={{ fontSize: "0.78rem", color: "var(--color-muted)", textAlign: "center", marginTop: 10 }}>
                Running {100} generations · pop {200} · this takes ~10s in production
              </p>
            )}
          </div>

          {/* Results */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {!result && !loading && (
              <div className="glass" style={{ padding: 48, textAlign: "center", color: "var(--color-muted)" }}>
                <p>Set targets and click <strong style={{ color: "var(--color-text)" }}>Optimize</strong> to see the Pareto front.</p>
              </div>
            )}

            {result && (
              <>
                {/* Pareto scatter */}
                <div className="glass" style={{ padding: 24 }}>
                  <h3 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: 4 }}>
                    Pareto Front — {result.n_candidates} candidates
                  </h3>
                  <p style={{ fontSize: "0.78rem", color: "var(--color-muted)", marginBottom: 16 }}>
                    Click a point to inspect its composition.
                  </p>
                  <Plot
                    data={[{
                      type: "scatter",
                      mode: "markers",
                      x: result.pareto_front.map((c: any) => c.properties[axes[0]] ?? 0),
                      y: result.pareto_front.map((c: any) => c.properties[axes[1]] ?? 0),
                      marker: {
                        size: 12,
                        color: result.pareto_front.map((_: any, i: number) => i === selected ? "#FFB547" : "#6382FF"),
                        line: { color: "#fff", width: 1.5 },
                      },
                      text: result.pareto_front.map((_: any, i: number) => `Candidate ${i+1}`),
                      hovertemplate: `%{text}<br>${axes[0]}: %{x:.1f}<br>${axes[1]}: %{y:.1f}<extra></extra>`,
                    }]}
                    layout={{
                      height: 320,
                      margin: { t: 8, b: 48, l: 56, r: 16 },
                      paper_bgcolor: "transparent",
                      plot_bgcolor: "transparent",
                      font: { color: "#6B7A9E", size: 11 },
                      xaxis: { title: axes[0]?.replace(/_/g, " "), gridcolor: "rgba(99,130,255,0.08)" },
                      yaxis: { title: axes[1]?.replace(/_/g, " "), gridcolor: "rgba(99,130,255,0.08)" },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: "100%" }}
                    onClick={(e: any) => setSelected(e.points[0]?.pointIndex ?? null)}
                  />
                </div>

                {/* Candidate detail */}
                {selected !== null && (
                  <div className="glass" style={{ padding: 24 }}>
                    <h3 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: 16, color: "var(--color-gold)" }}>
                      Candidate {selected + 1} — Composition
                    </h3>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                      {Object.entries(result.pareto_front[selected].composition).map(([el, frac]: any) => (
                        <div key={el} style={{
                          padding: "8px 16px", borderRadius: 8, background: "var(--color-surface-2)",
                          border: "1px solid var(--color-border)", textAlign: "center", minWidth: 70,
                        }}>
                          <div style={{ fontWeight: 700, fontSize: "1.05rem", color: "var(--color-primary)" }}>{el}</div>
                          <div style={{ fontSize: "0.82rem", color: "var(--color-muted)" }}>{(frac * 100).toFixed(2)}%</div>
                        </div>
                      ))}
                    </div>
                    <div style={{ marginTop: 16, display: "flex", gap: 12, flexWrap: "wrap" }}>
                      {Object.entries(result.pareto_front[selected].properties).map(([prop, val]: any) => (
                        <div key={prop} style={{
                          padding: "8px 16px", borderRadius: 8,
                          background: "rgba(99,130,255,0.08)", border: "1px solid rgba(99,130,255,0.2)",
                        }}>
                          <span style={{ fontSize: "0.78rem", color: "var(--color-muted)" }}>{prop.replace(/_/g, " ")}: </span>
                          <span style={{ fontWeight: 700, color: "var(--color-primary)" }}>{val.toFixed(1)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* CSV export */}
                <button className="btn-ghost" style={{ alignSelf: "flex-start" }}
                  onClick={() => {
                    const rows = [["Candidate", ...Object.keys(result.pareto_front[0].composition), ...axes]];
                    result.pareto_front.forEach((c: any, i: number) => {
                      rows.push([i+1, ...Object.values(c.composition), ...axes.map((a: string) => c.properties[a].toFixed(2))]);
                    });
                    const csv = rows.map(r => r.join(",")).join("\n");
                    const a = document.createElement("a"); a.href = "data:text/csv," + encodeURIComponent(csv); a.download = "pareto_front.csv"; a.click();
                  }}>
                  ↓ Export Pareto Front CSV
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      <style>{`@keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }`}</style>
    </main>
  );
}
