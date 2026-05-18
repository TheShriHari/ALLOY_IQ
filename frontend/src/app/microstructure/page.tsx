"use client";

import { useState } from "react";
import { Loader2, Download } from "lucide-react";

const PHASE_COLORS = {
  Martensite: "#4A5568",
  Ferrite:    "#718096",
  Austenite:  "#D69E2E",
  Carbide:    "#1A202C",
};

// Simulated phase fractions until Blender API is live
const DEMO_PHASES = {
  martensite_pct: 65,
  ferrite_pct:    25,
  carbide_pct:    8,
  austenite_pct:  2,
  grain_size_um:  22,
};

export default function MicrostructurePage() {
  const [phases, setPhases]     = useState(DEMO_PHASES);
  const [loading, setLoading]   = useState(false);
  const [rendered, setRendered] = useState(false);
  const [imgUrl, setImgUrl]     = useState<string | null>(null);

  function updatePhase(key: string, val: number) {
    setPhases(prev => ({ ...prev, [key]: isNaN(val) ? 0 : val }));
  }

  const totalPhases = phases.martensite_pct + phases.ferrite_pct + phases.austenite_pct + phases.carbide_pct;
  const isValid = Math.abs(totalPhases - 100) < 2;

  async function handleRender() {
    setLoading(true); setRendered(false); setImgUrl(null);
    // In production: POST to /visualize with phases, get image URL back
    await new Promise(r => setTimeout(r, 2000));
    // Show a placeholder canvas visualization (production uses Blender PNG)
    setRendered(true);
    setLoading(false);
  }

  // Canvas-based preview (lightweight client-side stand-in)
  function drawMicrostructure(canvas: HTMLCanvasElement | null) {
    if (!canvas || !rendered) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#080B14";
    ctx.fillRect(0, 0, W, H);

    // Simple Voronoi-like grain viz
    const seeds: { x: number; y: number; phase: string }[] = [];
    const phaseList = [
      ...Array(Math.floor(phases.martensite_pct / 3)).fill("Martensite"),
      ...Array(Math.floor(phases.ferrite_pct / 3)).fill("Ferrite"),
      ...Array(Math.floor(phases.austenite_pct / 3)).fill("Austenite"),
    ];
    for (let i = 0; i < phaseList.length; i++) {
      seeds.push({ x: Math.random() * W, y: Math.random() * H, phase: phaseList[i] });
    }

    // Fill pixels by nearest seed
    const imgData = ctx.createImageData(W, H);
    const grain = phases.grain_size_um;
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        let best = Infinity, bestSeed = seeds[0];
        for (const s of seeds) {
          const d = (px - s.x) ** 2 + (py - s.y) ** 2;
          if (d < best) { best = d; bestSeed = s; }
        }
        const col = PHASE_COLORS[bestSeed.phase as keyof typeof PHASE_COLORS] ?? "#333";
        const r = parseInt(col.slice(1, 3), 16);
        const g = parseInt(col.slice(3, 5), 16);
        const b = parseInt(col.slice(5, 7), 16);
        const idx = (py * W + px) * 4;
        imgData.data[idx]   = r;
        imgData.data[idx+1] = g;
        imgData.data[idx+2] = b;
        imgData.data[idx+3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Scatter carbide precipitates
    const nCarbides = Math.floor(phases.carbide_pct * 6);
    for (let i = 0; i < nCarbides; i++) {
      ctx.beginPath();
      ctx.arc(Math.random() * W, Math.random() * H, 3 + Math.random() * 4, 0, Math.PI * 2);
      ctx.fillStyle = PHASE_COLORS.Carbide;
      ctx.fill();
    }
  }

  return (
    <main style={{ paddingTop: 80 }}>
      <div className="container section" style={{ paddingTop: 40 }}>
        <h1 style={{ fontSize: "clamp(1.8rem, 3.5vw, 2.5rem)", fontWeight: 800, letterSpacing: "-0.03em", marginBottom: 8 }}>
          Microstructure <span className="gradient-text">Visualizer</span>
        </h1>
        <p style={{ color: "var(--color-muted)", fontSize: "0.95rem", marginBottom: 40 }}>
          Input predicted phase fractions to generate a Blender-rendered grain microstructure.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: 32 }}>
          {/* Controls */}
          <div className="glass" style={{ padding: 28 }}>
            <div style={{ marginBottom: 4 }}>
              <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-muted)", letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 12 }}>
                Phase Fractions (must sum to 100%)
              </label>

              {[
                { key: "martensite_pct", label: "Martensite %", color: PHASE_COLORS.Martensite },
                { key: "ferrite_pct",    label: "Ferrite %",    color: PHASE_COLORS.Ferrite    },
                { key: "austenite_pct",  label: "Austenite %",  color: PHASE_COLORS.Austenite  },
                { key: "carbide_pct",    label: "Carbide %",    color: PHASE_COLORS.Carbide    },
              ].map(({ key, label, color }) => (
                <div key={key} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: color, border: "1px solid rgba(255,255,255,0.1)", display: "inline-block" }} />
                      <span style={{ fontSize: "0.82rem", color: "var(--color-text)" }}>{label}</span>
                    </div>
                    <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--color-primary)" }}>
                      {phases[key as keyof typeof phases]}%
                    </span>
                  </div>
                  <input type="range" min={0} max={100}
                    value={phases[key as keyof typeof phases]}
                    onChange={e => updatePhase(key, parseInt(e.target.value))}
                    style={{ width: "100%", accentColor: "var(--color-primary)" }}
                  />
                </div>
              ))}

              <div style={{ marginTop: 16, marginBottom: 6 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: "0.82rem", color: "var(--color-text)" }}>Grain Size (µm)</span>
                  <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--color-primary)" }}>{phases.grain_size_um} µm</span>
                </div>
                <input type="range" min={5} max={100}
                  value={phases.grain_size_um}
                  onChange={e => updatePhase("grain_size_um", parseInt(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--color-accent)", marginTop: 6 }}
                />
              </div>

              <div style={{
                marginTop: 12, padding: "8px 12px", borderRadius: 8,
                background: isValid ? "rgba(0,229,119,0.08)" : "rgba(255,77,106,0.08)",
                border: `1px solid ${isValid ? "rgba(0,229,119,0.25)" : "rgba(255,77,106,0.25)"}`,
                fontSize: "0.78rem", color: isValid ? "#00E577" : "var(--color-danger)", fontWeight: 600,
              }}>
                Σ phases = {totalPhases.toFixed(1)}% {isValid ? "✓" : " — must equal 100%"}
              </div>
            </div>

            <button onClick={handleRender} disabled={loading || !isValid}
              className="btn-glow" style={{ width: "100%", marginTop: 20, opacity: (loading || !isValid) ? 0.6 : 1 }}>
              {loading ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} /> Rendering…
              </span> : "Generate Microstructure →"}
            </button>

            {/* Phase legend */}
            <div style={{ marginTop: 24, paddingTop: 20, borderTop: "1px solid var(--color-border)" }}>
              <p style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 12 }}>
                Phase Legend
              </p>
              {Object.entries(PHASE_COLORS).map(([phase, color]) => (
                <div key={phase} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <span style={{ width: 16, height: 16, borderRadius: 4, background: color, border: "1px solid rgba(255,255,255,0.1)", flexShrink: 0 }} />
                  <span style={{ fontSize: "0.82rem", color: "var(--color-text)" }}>{phase}</span>
                  <span style={{ fontSize: "0.78rem", color: "var(--color-muted)", marginLeft: "auto" }}>
                    {{Martensite: phases.martensite_pct, Ferrite: phases.ferrite_pct, Austenite: phases.austenite_pct, Carbide: phases.carbide_pct}[phase]}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Canvas */}
          <div className="glass" style={{ padding: 24, display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ fontSize: "0.9rem", fontWeight: 700 }}>
                {rendered ? "Microstructure Preview" : "Awaiting Render"}
              </h3>
              {rendered && (
                <button className="btn-ghost" style={{ padding: "6px 14px", fontSize: "0.8rem", display: "flex", alignItems: "center", gap: 6 }}>
                  <Download size={13} /> Save PNG
                </button>
              )}
            </div>

            {!rendered ? (
              <div style={{
                flex: 1, minHeight: 380, display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center", color: "var(--color-muted)",
                background: "var(--color-surface-2)", borderRadius: 12,
                border: "1px dashed var(--color-border)",
              }}>
                <div style={{ fontSize: 3, marginBottom: 16, opacity: 0.3 }}>
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                    <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6v6H9z"/>
                    <path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>
                  </svg>
                </div>
                <p style={{ fontSize: "0.88rem" }}>Configure phases and click Generate</p>
                <p style={{ fontSize: "0.78rem", marginTop: 6 }}>
                  Production: Blender headless renders Cycles PNG (1920×1080)
                </p>
              </div>
            ) : (
              <canvas
                ref={drawMicrostructure}
                width={760} height={420}
                style={{ borderRadius: 10, width: "100%", height: "auto", border: "1px solid var(--color-border)" }}
              />
            )}

            {rendered && (
              <p style={{ fontSize: "0.78rem", color: "var(--color-muted)", marginTop: 12, textAlign: "center" }}>
                Preview rendered client-side · Production uses Blender Cycles with Voronoi grain boundaries and PBR phase shaders
              </p>
            )}
          </div>
        </div>
      </div>

      <style>{`@keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }`}</style>
    </main>
  );
}
