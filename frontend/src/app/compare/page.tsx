"use client";

import { useState } from "react";
import { Plus, Trash2, Download, AlertCircle, Loader2 } from "lucide-react";
import { api, PredictionResponse } from "@/lib/api";
import { ErrorAlert } from "@/components/ui/ErrorAlert";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

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

interface CompareColumn {
  id: string;
  name: string;
  composition: Record<string, number>;
  loading: boolean;
  prediction: PredictionResponse | null;
  error: string | null;
}

export default function ComparePage() {
  const [family, setFamily] = useState<AlloySFamily>("steel");
  const [columns, setColumns] = useState<CompareColumn[]>([
    {
      id: "A",
      name: "Alloy A",
      composition: { Fe: 0.70, Cr: 0.18, Ni: 0.08, C: 0.02, Mn: 0.02 },
      loading: false,
      prediction: null,
      error: null,
    },
    {
      id: "B",
      name: "Alloy B",
      composition: { Fe: 0.65, Cr: 0.20, Ni: 0.10, Mo: 0.03, C: 0.02 },
      loading: false,
      prediction: null,
      error: null,
    },
  ]);

  const [activeProperty, setActiveProperty] = useState("yield_strength");

  function addColumn() {
    if (columns.length >= 4) return;
    const nextLetter = String.fromCharCode(65 + columns.length);
    // Copy the first column composition as a baseline template
    const template = { ...columns[0].composition };
    setColumns(prev => [
      ...prev,
      {
        id: nextLetter,
        name: `Alloy ${nextLetter}`,
        composition: template,
        loading: false,
        prediction: null,
        error: null,
      },
    ]);
  }

  function removeColumn(id: string) {
    if (columns.length <= 2) return;
    setColumns(prev => prev.filter(col => col.id !== id));
  }

  function updateElementName(colId: string, oldEl: string, newEl: string) {
    setColumns(prev =>
      prev.map(col => {
        if (col.id !== colId) return col;
        const nextComp = { ...col.composition };
        const val = nextComp[oldEl] ?? 0;
        delete nextComp[oldEl];
        nextComp[newEl] = val;
        return { ...col, composition: nextComp, prediction: null };
      })
    );
  }

  function updateElementValue(colId: string, el: string, val: number) {
    setColumns(prev =>
      prev.map(col => {
        if (col.id !== colId) return col;
        return {
          ...col,
          composition: {
            ...col.composition,
            [el]: isNaN(val) ? 0 : val,
          },
          prediction: null,
        };
      })
    );
  }

  function addElementToColumn(colId: string, el: string) {
    setColumns(prev =>
      prev.map(col => {
        if (col.id !== colId) return col;
        if (col.composition[el] !== undefined) return col;
        return {
          ...col,
          composition: { ...col.composition, [el]: 0 },
          prediction: null,
        };
      })
    );
  }

  function removeElementFromColumn(colId: string, el: string) {
    setColumns(prev =>
      prev.map(col => {
        if (col.id !== colId) return col;
        const nextComp = { ...col.composition };
        delete nextComp[el];
        return { ...col, composition: nextComp, prediction: null };
      })
    );
  }

  async function predictColumn(colId: string) {
    setColumns(prev =>
      prev.map(col => (col.id === colId ? { ...col, loading: true, error: null } : col))
    );

    const col = columns.find(c => c.id === colId);
    if (!col) return;

    try {
      // Use the first property from family list to retrieve the full response (it contains predictions for ALL properties in the list)
      const data = await api.predict(family, activeProperty, col.composition);
      setColumns(prev =>
        prev.map(c =>
          c.id === colId ? { ...c, loading: false, prediction: data, error: null } : c
        )
      );
    } catch (e) {
      setColumns(prev =>
        prev.map(c =>
          c.id === colId
            ? {
                ...c,
                loading: false,
                error: e instanceof Error ? e.message : "Prediction failed",
              }
            : c
        )
      );
    }
  }

  async function predictAll() {
    columns.forEach(col => predictColumn(col.id));
  }

  function exportCSV() {
    const header = [
      "Property / Element",
      ...columns.map(c => c.name)
    ];

    const rows: string[][] = [];

    // Compositions
    const allElements = Array.from(new Set(columns.flatMap(c => Object.keys(c.composition))));
    allElements.forEach(el => {
      const row = [el, ...columns.map(c => ((c.composition[el] ?? 0) * 100).toFixed(2) + "%")];
      rows.push(row);
    });

    // Predicted values
    PROPERTIES[family].forEach(p => {
      const row = [
        p.toUpperCase(),
        ...columns.map(c => {
          if (!c.prediction) return "Not Predicted";
          const val = c.prediction.predictions[p];
          return val ? val.mean.toFixed(1) : "—";
        }),
      ];
      rows.push(row);
    });

    const csvContent = [header, ...rows].map(r => r.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "alloy_comparison.csv");
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // Plotly chart data comparing the active property
  const predictedDataPoints = columns.map(c => {
    if (!c.prediction) return 0;
    const val = c.prediction.predictions[activeProperty];
    return val ? val.mean : 0;
  });

  return (
    <main style={{ paddingTop: 80 }}>
      <div className="container section" style={{ paddingTop: 40 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <h1 style={{ fontSize: "clamp(1.8rem, 3.5vw, 2.5rem)", fontWeight: 800, letterSpacing: "-0.03em" }}>
            Alloy <span className="gradient-text">Comparison</span>
          </h1>
          <div style={{ display: "flex", gap: 12 }}>
            <button onClick={predictAll} className="btn-glow" style={{ fontSize: "0.85rem", padding: "10px 20px" }}>
              Predict All
            </button>
            <button onClick={exportCSV} className="btn-ghost" style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: 8 }}>
              <Download size={15} /> Export CSV
            </button>
          </div>
        </div>
        <p style={{ color: "var(--color-muted)", fontSize: "0.95rem", marginBottom: 36 }}>
          Compare up to 4 chemical compositions side-by-side with interactive property benchmarks.
        </p>

        {/* Global Controls */}
        <div className="glass" style={{ padding: 20, marginBottom: 32, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 16 }}>
          <div>
            <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginRight: 12 }}>
              Alloy Family:
            </span>
            <div style={{ display: "inline-flex", gap: 6 }}>
              {ALLOY_FAMILIES.map(f => (
                <button key={f} onClick={() => { setFamily(f); setActiveProperty(PROPERTIES[f][0]); }}
                  style={{
                    padding: "6px 14px", borderRadius: 8, cursor: "pointer", fontSize: "0.8rem", fontWeight: 600,
                    border: family === f ? "1px solid var(--color-primary)" : "1px solid var(--color-border)",
                    background: family === f ? "rgba(99,130,255,0.15)" : "var(--color-surface-2)",
                    color: family === f ? "var(--color-primary)" : "var(--color-muted)",
                  }}>
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div>
            <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginRight: 12 }}>
              Benchmark Chart Property:
            </span>
            <select value={activeProperty} onChange={e => setActiveProperty(e.target.value)}
              className="input-field" style={{ width: 180, display: "inline-block", fontSize: "0.82rem" }}>
              {PROPERTIES[family].map(p => (
                <option key={p} value={p}>{p.replace(/_/g, " ").toUpperCase()}</option>
              ))}
            </select>
          </div>
        </div>

        {/* benchmark comparison chart */}
        {columns.some(c => c.prediction) && (
          <div className="glass" style={{ padding: 24, marginBottom: 32 }}>
            <h3 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: 12, color: "var(--color-text)" }}>
               benchmark: {activeProperty.replace(/_/g, " ").toUpperCase()}
            </h3>
            <Plot
              data={[{
                type: "bar",
                x: columns.map(c => c.name),
                y: predictedDataPoints,
                marker: { color: ["#6382FF", "#00E5FF", "#FFB547", "#FF4D6A"] },
              }]}
              layout={{
                height: 250,
                margin: { t: 8, b: 32, l: 48, r: 16 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#6B7A9E", size: 11 },
                xaxis: { gridcolor: "transparent" },
                yaxis: { gridcolor: "rgba(99,130,255,0.08)" },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          </div>
        )}

        {/* Side-by-side columns */}
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${columns.length}, 1fr)`, gap: 20, alignItems: "start" }}>
          {columns.map(col => {
            const total = Object.values(col.composition).reduce((a, b) => a + b, 0);
            const isValid = Math.abs(total - 1.0) < 0.02;

            return (
              <div key={col.id} className="glass" style={{ padding: 20, position: "relative" }}>
                {columns.length > 2 && (
                  <button onClick={() => removeColumn(col.id)}
                    style={{ position: "absolute", top: 16, right: 16, color: "var(--color-danger)", border: "none", background: "none", cursor: "pointer" }}>
                    <Trash2 size={16} />
                  </button>
                )}

                <h2 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 16, color: "var(--color-primary)" }}>
                  {col.name}
                </h2>

                {/* Element list */}
                <div style={{ marginBottom: 20 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--color-muted)" }}>COMPOSITION</span>
                    <span style={{ fontSize: "0.75rem", fontWeight: 700, color: isValid ? "#00E577" : "var(--color-danger)" }}>
                      Σ = {total.toFixed(3)}
                    </span>
                  </div>

                  {Object.entries(col.composition).map(([el, val]) => (
                    <div key={el} style={{ display: "flex", gap: 8, marginBottom: 6, alignItems: "center" }}>
                      <span style={{ width: 32, fontSize: "0.8rem", fontWeight: 700, color: "var(--color-accent)" }}>{el}</span>
                      <input
                        type="number" step="0.005" min="0" max="1"
                        value={val}
                        onChange={e => updateElementValue(col.id, el, parseFloat(e.target.value))}
                        className="input-field" style={{ flex: 1, padding: "5px 8px", fontSize: "0.82rem" }}
                      />
                      <button onClick={() => removeElementFromColumn(col.id, el)}
                        style={{ border: "none", background: "none", color: "var(--color-muted)", cursor: "pointer", fontSize: "1rem" }}>
                        ×
                      </button>
                    </div>
                  ))}

                  {/* Add Element dropdown */}
                  <select onChange={e => { if (e.target.value) addElementToColumn(col.id, e.target.value); e.target.value = ""; }}
                    className="input-field" style={{ fontSize: "0.78rem", padding: "4px 8px", marginTop: 8 }}>
                    <option value="">+ Add element…</option>
                    {ELEMENTS[family].filter(el => col.composition[el] === undefined).map(el => (
                      <option key={el} value={el}>{el}</option>
                    ))}
                  </select>
                </div>

                {/* Action button */}
                <button onClick={() => predictColumn(col.id)} disabled={col.loading || !isValid}
                  className="btn-ghost" style={{ width: "100%", marginBottom: 16, opacity: (col.loading || !isValid) ? 0.6 : 1 }}>
                  {col.loading ? <Loader2 className="animate-spin" size={14} style={{ margin: "0 auto" }} /> : "Predict Column"}
                </button>

                {col.error && (
                  <div style={{ marginBottom: 16 }}>
                    <ErrorAlert message={col.error} />
                  </div>
                )}

                {/* Column Predictions list */}
                {col.prediction && (
                  <div style={{ background: "var(--color-surface-2)", borderRadius: 12, padding: 16 }}>
                    <h4 style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--color-muted)", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      Predicted Properties
                    </h4>
                    
                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                      {PROPERTIES[family].map(p => {
                        const val = col.prediction!.predictions[p];
                        return (
                          <div key={p} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--color-border)", paddingBottom: 6 }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--color-text)" }}>{p.replace(/_/g, " ")}</span>
                            <span style={{ fontWeight: 700, color: "var(--color-primary)", fontSize: "0.88rem" }}>
                              {val ? `${val.mean.toFixed(1)}` : "—"}
                            </span>
                          </div>
                        );
                      })}
                      
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: 4 }}>
                        <span style={{ fontSize: "0.8rem", color: "var(--color-text)" }}>PREN Corrosion</span>
                        <span style={{ fontWeight: 700, color: "#00E5FF", fontSize: "0.88rem" }}>
                          {col.prediction.corrosion_analysis.pren_calculated.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {columns.length < 4 && (
            <button onClick={addColumn} className="glass"
              style={{
                height: 250, border: "2px dashed var(--color-border)", background: "transparent",
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                gap: 12, cursor: "pointer", color: "var(--color-muted)", transition: "all 0.15s",
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = "var(--color-primary)"}
              onMouseLeave={e => e.currentTarget.style.borderColor = "var(--color-border)"}
            >
              <Plus size={24} />
              <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>Add Alloy Column</span>
            </button>
          )}
        </div>
      </div>
    </main>
  );
}
