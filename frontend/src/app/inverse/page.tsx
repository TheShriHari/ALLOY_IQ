"use client";

import { useState, useEffect, useRef } from "react";
import { Loader2, Plus, Trash2, RotateCcw, AlertCircle, CheckCircle2 } from "lucide-react";
import dynamic from "next/dynamic";
import axios from "axios";

// Dynamically import Plotly.js to disable SSR warnings
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

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

export default function InversePage() {
  const [family, setFamily] = useState<"steel" | "hea" | "aluminum">("steel");
  const [targets, setTargets] = useState<TargetRow[]>([
    { property: "yield_strength", operator: ">", value: 900 },
    { property: "corrosion_pren", operator: ">", value: 35 },
  ]);
  
  const [loading, setLoading] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  
  // Real-time tracking states
  const [currentGeneration, setCurrentGeneration] = useState(0);
  const [totalGenerations, setTotalGenerations] = useState(50);
  const [liveCandidates, setLiveCandidates] = useState<any[]>([]);
  const [liveAxes, setLiveAxes] = useState<string[]>([]);
  
  // Final Result State
  const [result, setResult] = useState<any>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [recoveryStatus, setRecoveryStatus] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);

  // Parse websocket target URL dynamically from API endpoint
  const getWsUrl = (jobId: string) => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const wsProtocol = apiBase.startsWith("https") ? "wss:" : "ws:";
    const wsHost = apiBase.replace(/^https?:\/\//, "");
    return `${wsProtocol}//${wsHost}/ws/jobs/${jobId}`;
  };

  // Connect to the WebSocket for live updates & recovery
  const connectJobWebSocket = (jobId: string) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    setLoading(true);
    setActiveJobId(jobId);
    setResult(null);
    setSelected(null);
    setRecoveryStatus(null);

    const wsUrl = getWsUrl(jobId);
    console.log("Connecting WebSocket to", wsUrl);
    
    const socket = new WebSocket(wsUrl);
    wsRef.current = socket;

    socket.onopen = () => {
      console.log("WebSocket connected for job:", jobId);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle heartbeat pings
        if (data.type === "ping") {
          return;
        }

        if (data.status === "running") {
          setCurrentGeneration(data.generation || 0);
          setTotalGenerations(data.total_generations || 50);
          
          if (data.pareto_front && data.pareto_front.length > 0) {
            setLiveCandidates(data.pareto_front);
            // Grab property names dynamically from live front if they exist
            const sampleProps = Object.keys(data.pareto_front[0].predictions || data.pareto_front[0].properties || {});
            setLiveAxes(sampleProps.filter(p => p !== "Fe" && !p.startsWith("frac_")));
          }
          
          if (data.message && data.message.includes("Recovered")) {
            setRecoveryStatus("Recovered active session progress.");
          }
        } 
        else if (data.status === "complete") {
          console.log("Job completed successfully!");
          const finalResult = data.result;
          setResult(finalResult);
          setLiveCandidates(finalResult.pareto_front || []);
          setLiveAxes(finalResult.objective_axes || []);
          
          setLoading(false);
          setActiveJobId(null);
          setRecoveryStatus(null);
          localStorage.removeItem("active_inverse_job");
          socket.close();
        } 
        else if (data.status === "error") {
          alert(`Optimization error: ${data.message || "Failed"}`);
          setLoading(false);
          setActiveJobId(null);
          setRecoveryStatus(null);
          localStorage.removeItem("active_inverse_job");
          socket.close();
        }
      } catch (err) {
        console.error("Error parsing WS message:", err);
      }
    };

    socket.onerror = (err) => {
      console.error("WebSocket error for job:", jobId, err);
    };

    socket.onclose = () => {
      console.log("WebSocket connection closed for job:", jobId);
    };
  };

  // Browser refresh recovery hook
  useEffect(() => {
    const cachedJobId = localStorage.getItem("active_inverse_job");
    if (cachedJobId) {
      console.log("Found cached running inverse job:", cachedJobId);
      connectJobWebSocket(cachedJobId);
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

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
    setLoading(true);
    setResult(null);
    setSelected(null);
    setLiveCandidates([]);
    setLiveAxes([]);
    setCurrentGeneration(0);

    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    try {
      const targetsPayload: any = {};
      targets.forEach(t => {
        targetsPayload[t.property] = [t.operator, t.value];
      });

      // 1. Request task dispatch from FastAPI endpoints
      const response = await axios.post(`${API_URL}/api/v1/inverse`, {
        alloy_family: family,
        targets: targetsPayload,
        n_generations: 50,
        pop_size: 100
      });

      const jobId = response.data.job_id;
      if (!jobId) {
        throw new Error("API failed to return job ID.");
      }

      // 2. Cache job_id in localStorage to support recovery upon refresh
      localStorage.setItem("active_inverse_job", jobId);
      
      // 3. Connect to WS stream
      connectJobWebSocket(jobId);

    } catch (e: any) {
      console.error(e);
      alert(e.response?.data?.detail || "Failed to start optimization.");
      setLoading(false);
    }
  }

  // Determine active plot datasets (use complete result if done, otherwise show live candidates)
  const activeCandidates = result?.pareto_front || liveCandidates;
  const axes = result?.objective_axes || liveAxes;

  // Compute progress percentage
  const progressPercent = Math.min(100, Math.round((currentGeneration / totalGenerations) * 100));

  return (
    <main style={{ paddingTop: 80 }}>
      <div className="container section" style={{ paddingTop: 40 }}>
        <h1 style={{ fontSize: "clamp(1.8rem, 3.5vw, 2.5rem)", fontWeight: 800, letterSpacing: "-0.03em", marginBottom: 8 }}>
          Inverse <span className="gradient-text">Design</span>
        </h1>
        <p style={{ color: "var(--color-muted)", fontSize: "0.95rem", marginBottom: 40 }}>
          Specify property targets and let NSGA-II discover the Pareto-optimal alloy candidates.
        </p>

        {recoveryStatus && (
          <div className="glass" style={{
            display: "flex", alignItems: "center", gap: 12, padding: "12px 20px",
            border: "1px solid rgba(99,130,255,0.4)", background: "rgba(99,130,255,0.1)",
            borderRadius: 10, marginBottom: 24, fontSize: "0.88rem", color: "#6382FF"
          }}>
            <RotateCcw size={16} className="animate-spin" style={{ animation: "spin 2s linear infinite" }} />
            <span>{recoveryStatus}</span>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 32, alignItems: "start" }}>
          {/* Input Panel */}
          <div className="glass" style={{ padding: 28 }}>
            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-muted)", letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>
                Alloy Family
              </label>
              <div style={{ display: "flex", gap: 6 }}>
                {ALLOY_FAMILIES.map(f => (
                  <button key={f} onClick={() => setFamily(f)} disabled={loading}
                    style={{
                      flex: 1, padding: "8px 0", borderRadius: 8, cursor: "pointer", fontSize: "0.8rem", fontWeight: 600,
                      border: family === f ? "1px solid var(--color-primary)" : "1px solid var(--color-border)",
                      background: family === f ? "rgba(99,130,255,0.15)" : "var(--color-surface-2)",
                      color: family === f ? "var(--color-primary)" : "var(--color-muted)",
                      opacity: loading ? 0.6 : 1,
                      transition: "all 0.2s ease"
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
                  <select value={t.property} onChange={e => updateTarget(i, "property", e.target.value)} disabled={loading}
                    className="input-field" style={{ flex: 1, marginRight: 8, fontSize: "0.82rem" }}>
                    {ALL_PROPERTIES.map(p => <option key={p} value={p}>{PROP_LABELS[p]}</option>)}
                  </select>
                  <button onClick={() => removeTarget(i)} disabled={loading}
                    style={{ background: "none", border: "none", cursor: "pointer", color: "var(--color-danger)", opacity: loading ? 0.4 : 1 }}>
                    <Trash2 size={15} />
                  </button>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <select value={t.operator} onChange={e => updateTarget(i, "operator", e.target.value)} disabled={loading}
                    className="input-field" style={{ width: 60, fontSize: "0.85rem" }}>
                    {([">", "<", ">=", "<="] as Operator[]).map(op => <option key={op} value={op}>{op}</option>)}
                  </select>
                  <input type="number" value={t.value} onChange={e => updateTarget(i, "value", parseFloat(e.target.value))} disabled={loading}
                    className="input-field" style={{ flex: 1 }} />
                </div>
              </div>
            ))}

            <button onClick={addTarget} disabled={loading || targets.length >= ALL_PROPERTIES.length}
              className="btn-ghost" style={{ width: "100%", marginBottom: 16, display: "flex", alignItems: "center", justifyContent: "center", gap: 6, fontSize: "0.85rem" }}>
              <Plus size={15} /> Add Target
            </button>

            <button onClick={handleOptimize} disabled={loading || targets.length === 0}
              className="btn-glow" style={{ width: "100%", opacity: (loading || targets.length === 0) ? 0.6 : 1 }}>
              {loading ? (
                <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                  <Loader2 size={16} className="animate-spin" style={{ animation: "spin 1s linear infinite" }} /> 
                  Running NSGA-II…
                </span>
              ) : "Optimize Composition →"}
            </button>

            {loading && (
              <div style={{ marginTop: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", color: "var(--color-muted)", marginBottom: 6 }}>
                  <span>Generations: {currentGeneration}/{totalGenerations}</span>
                  <span>{progressPercent}%</span>
                </div>
                {/* Real-time Progress Bar */}
                <div style={{ width: "100%", height: 6, background: "var(--color-border)", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{
                    width: `${progressPercent}%`, height: "100%", background: "var(--color-primary)",
                    borderRadius: 3, transition: "width 0.4s ease"
                  }} />
                </div>
                <p style={{ fontSize: "0.72rem", color: "#6382FF", textAlign: "center", marginTop: 8, letterSpacing: "0.02em" }}>
                  Running securely in background worker thread.
                </p>
              </div>
            )}
          </div>

          {/* Results Visualizer Panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {!activeCandidates || activeCandidates.length === 0 ? (
              <div className="glass" style={{ padding: 48, textAlign: "center", color: "var(--color-muted)" }}>
                <p>Set targets and click <strong style={{ color: "var(--color-text)" }}>Optimize</strong> to see the Pareto front.</p>
              </div>
            ) : (
              <>
                {/* Pareto Scatter Chart */}
                <div className="glass" style={{ padding: 24 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <div>
                      <h3 style={{ fontSize: "0.95rem", fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
                        {loading ? (
                          <>
                            <span style={{ position: "relative", display: "flex", height: 8, width: 8 }}>
                              <span style={{ position: "absolute", display: "inline-flex", height: "100%", width: "100%", borderRadius: "50%", backgroundColor: "#6382FF", opacity: 0.75 }} className="animate-ping" />
                              <span style={{ position: "relative", display: "inline-flex", borderRadius: "50%", height: 8, width: 8, backgroundColor: "#6382FF" }} />
                            </span>
                            Live Pareto Front Explorer — {activeCandidates.length} candidates
                          </>
                        ) : (
                          <>
                            <CheckCircle2 size={16} style={{ color: "var(--color-success)" }} />
                            Pareto Front — {activeCandidates.length} candidates
                          </>
                        )}
                      </h3>
                      <p style={{ fontSize: "0.78rem", color: "var(--color-muted)" }}>
                        {loading ? "Chart updating dynamically as generations evolve." : "Click a point to inspect its composition."}
                      </p>
                    </div>
                  </div>
                  
                  {axes.length >= 2 ? (
                    <Plot
                      data={[{
                        type: "scatter",
                        mode: "markers",
                        x: activeCandidates.map((c: any) => (c.properties ? c.properties[axes[0]] : c.predictions ? c.predictions[axes[0]] : 0)),
                        y: activeCandidates.map((c: any) => (c.properties ? c.properties[axes[1]] : c.predictions ? c.predictions[axes[1]] : 0)),
                        marker: {
                          size: 13,
                          color: activeCandidates.map((_: any, i: number) => i === selected ? "#FFB547" : "#6382FF"),
                          line: { color: "#fff", width: 1.5 },
                        },
                        text: activeCandidates.map((_: any, i: number) => `Candidate ${i + 1}`),
                        hovertemplate: `%{text}<br>${axes[0]}: %{x:.1f}<br>${axes[1]}: %{y:.1f}<extra></extra>`,
                      }]}
                      layout={{
                        height: 340,
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
                  ) : (
                    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: 200, color: "var(--color-muted)" }}>
                      <Loader2 size={24} className="animate-spin" style={{ animation: "spin 1s linear infinite", marginRight: 8 }} />
                      <span>Generating live Pareto points...</span>
                    </div>
                  )}
                </div>

                {/* Candidate detail */}
                {selected !== null && activeCandidates[selected] && (
                  <div className="glass" style={{ padding: 24 }}>
                    <h3 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: 16, color: "var(--color-gold)" }}>
                      Candidate {selected + 1} — Composition
                    </h3>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                      {Object.entries(activeCandidates[selected].composition).map(([el, frac]: any) => (
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
                      {Object.entries(activeCandidates[selected].predictions || activeCandidates[selected].properties || {}).map(([prop, val]: any) => (
                        <div key={prop} style={{
                          padding: "8px 16px", borderRadius: 8,
                          background: "rgba(99,130,255,0.08)", border: "1px solid rgba(99,130,255,0.2)",
                        }}>
                          <span style={{ fontSize: "0.78rem", color: "var(--color-muted)" }}>{prop.replace(/_/g, " ")}: </span>
                          <span style={{ fontWeight: 700, color: "var(--color-primary)" }}>{typeof val === "number" ? val.toFixed(1) : val}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* CSV export */}
                {!loading && activeCandidates.length > 0 && (
                  <button className="btn-ghost" style={{ alignSelf: "flex-start" }}
                    onClick={() => {
                      const sampleCompKeys = Object.keys(activeCandidates[0].composition);
                      const rows = [["Candidate", ...sampleCompKeys, ...axes]];
                      activeCandidates.forEach((c: any, i: number) => {
                        const compValues = sampleCompKeys.map(k => c.composition[k] || 0);
                        const predValues = axes.map((a: string) => {
                          const val = c.properties ? c.properties[a] : c.predictions ? c.predictions[a] : 0;
                          return typeof val === "number" ? val.toFixed(2) : val;
                        });
                        rows.push([i + 1, ...compValues, ...predValues]);
                      });
                      const csv = rows.map(r => r.join(",")).join("\n");
                      const a = document.createElement("a");
                      a.href = "data:text/csv," + encodeURIComponent(csv);
                      a.download = `${family}_pareto_front.csv`;
                      a.click();
                    }}>
                    ↓ Export Pareto Front CSV
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
        .animate-spin { animation: spin 1s linear infinite; }
      `}</style>
    </main>
  );
}
