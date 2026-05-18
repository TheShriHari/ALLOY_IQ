"use client";

import { useState } from "react";
import { Atom, Brain, FlaskConical, Layers, Zap, Shield } from "lucide-react";
import Link from "next/link";

const FEATURES = [
  {
    icon: <Brain size={24} />,
    title: "12-Cell ML Matrix",
    desc: "XGBoost + RF + MLP stacking ensemble across all property × family combinations, with Bayesian HPO via Optuna.",
    color: "#6382FF",
  },
  {
    icon: <Zap size={24} />,
    title: "SHAP Narratives",
    desc: "Every prediction generates a SHAP waterfall chart and auto-written plain-English explanation a PhD trusts.",
    color: "#00E5FF",
  },
  {
    icon: <FlaskConical size={24} />,
    title: "Inverse Design",
    desc: "NSGA-II genetic algorithm finds Pareto-optimal alloy compositions for multi-objective target specs.",
    color: "#FFB547",
  },
  {
    icon: <Shield size={24} />,
    title: "Conformal Uncertainty",
    desc: "Calibrated confidence intervals at 80/90/95% guaranteed coverage — not just a number with no context.",
    color: "#00E577",
  },
  {
    icon: <Layers size={24} />,
    title: "3D Microstructure",
    desc: "Voronoi grain boundaries, carbide precipitates, and PBR phase shaders rendered in Blender headless.",
    color: "#FF4D6A",
  },
  {
    icon: <Atom size={24} />,
    title: "Three Alloy Families",
    desc: "Steels, High-Entropy Alloys, and Aluminum Alloys — each with tailored physics-informed descriptors.",
    color: "#B47AFF",
  },
];

const STATS = [
  { value: "12",    label: "Property Cells"     },
  { value: "0.93+", label: "R² on Rich Cells"   },
  { value: "3",     label: "Alloy Families"      },
  { value: "90%",   label: "Conformal Coverage"  },
];

function FeatureCard({ icon, title, desc, color }: { icon: React.ReactNode; title: string; desc: string; color: string }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      className="glass"
      style={{
        padding: 28,
        transition: "transform 0.2s, box-shadow 0.2s",
        transform: hovered ? "translateY(-4px)" : "translateY(0)",
        boxShadow: hovered ? `0 8px 32px ${color}22` : "none",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div style={{
        width: 48, height: 48, borderRadius: 12, marginBottom: 18,
        background: `${color}18`, border: `1px solid ${color}30`,
        display: "flex", alignItems: "center", justifyContent: "center", color,
      }}>
        {icon}
      </div>
      <h3 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: 10, letterSpacing: "-0.01em" }}>
        {title}
      </h3>
      <p style={{ fontSize: "0.9rem", color: "var(--color-muted)", lineHeight: 1.65 }}>
        {desc}
      </p>
    </div>
  );
}

export default function HomePage() {
  return (
    <main style={{ paddingTop: 64 }}>
      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section style={{
        minHeight: "92vh", display: "flex", alignItems: "center",
        position: "relative", overflow: "hidden",
      }}>
        <div style={{
          position: "absolute", top: "10%", left: "5%",
          width: 500, height: 500, borderRadius: "50%",
          background: "radial-gradient(circle, rgba(99,130,255,0.12) 0%, transparent 70%)",
          pointerEvents: "none",
        }} />
        <div style={{
          position: "absolute", bottom: "10%", right: "5%",
          width: 400, height: 400, borderRadius: "50%",
          background: "radial-gradient(circle, rgba(0,229,255,0.10) 0%, transparent 70%)",
          pointerEvents: "none",
        }} />

        <div className="container" style={{ position: "relative", zIndex: 1 }}>
          <div style={{ maxWidth: 740 }}>
            <div style={{
              display: "inline-flex", alignItems: "center", gap: 8, marginBottom: 28,
              background: "rgba(99,130,255,0.1)", border: "1px solid rgba(99,130,255,0.25)",
              borderRadius: 999, padding: "6px 16px", fontSize: "0.8rem", color: "var(--color-primary)",
              fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase",
            }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--color-primary)", display: "inline-block" }} />
              AI-Powered Materials Intelligence
            </div>

            <h1 style={{
              fontSize: "clamp(2.8rem, 6vw, 4.5rem)", fontWeight: 800, lineHeight: 1.08,
              letterSpacing: "-0.035em", marginBottom: 24,
            }}>
              Predict Any Alloy&apos;s{" "}
              <span className="gradient-text">Properties</span>
              <br />in Seconds.
            </h1>

            <p style={{
              fontSize: "1.15rem", color: "var(--color-muted)", lineHeight: 1.7,
              marginBottom: 40, maxWidth: 580,
            }}>
              ALLOY IQ combines physics-informed ML, SHAP explainability, multi-objective
              inverse design, and Blender microstructure visualization — the only platform
              a materials scientist will actually trust.
            </p>

            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <Link href="/predict">
                <button className="btn-glow" style={{ fontSize: "1rem", padding: "14px 32px" }}>
                  Start Predicting →
                </button>
              </Link>
              <Link href="/inverse">
                <button className="btn-ghost" style={{ fontSize: "1rem", padding: "13px 28px" }}>
                  Inverse Design
                </button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats bar ─────────────────────────────────────────────────────── */}
      <section style={{
        borderTop: "1px solid var(--color-border)", borderBottom: "1px solid var(--color-border)",
        background: "var(--color-surface)", padding: "32px 0",
      }}>
        <div className="container" style={{ display: "flex", justifyContent: "space-around", flexWrap: "wrap", gap: 24 }}>
          {STATS.map(({ value, label }) => (
            <div key={label} style={{ textAlign: "center" }}>
              <div style={{ fontSize: "2.4rem", fontWeight: 800, letterSpacing: "-0.03em" }} className="gradient-text">
                {value}
              </div>
              <div style={{ fontSize: "0.85rem", color: "var(--color-muted)", marginTop: 4, fontWeight: 500 }}>
                {label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────────────────────── */}
      <section className="section">
        <div className="container">
          <div style={{ textAlign: "center", marginBottom: 60 }}>
            <h2 style={{ fontSize: "clamp(1.8rem, 4vw, 2.6rem)", fontWeight: 800, letterSpacing: "-0.025em", marginBottom: 16 }}>
              Built for the{" "}
              <span className="gradient-text">Rigorous Engineer</span>
            </h2>
            <p style={{ fontSize: "1rem", color: "var(--color-muted)", maxWidth: 540, margin: "0 auto" }}>
              Every feature maps to a real pain point in materials qualification workflows.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24 }}>
            {FEATURES.map(f => <FeatureCard key={f.title} {...f} />)}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ────────────────────────────────────────────────────── */}
      <section className="section" style={{ paddingTop: 0 }}>
        <div className="container">
          <div style={{
            background: "linear-gradient(135deg, rgba(99,130,255,0.12) 0%, rgba(0,229,255,0.08) 100%)",
            border: "1px solid rgba(99,130,255,0.2)", borderRadius: 24,
            padding: "60px 48px", textAlign: "center",
          }}>
            <h2 style={{ fontSize: "clamp(1.6rem, 3.5vw, 2.4rem)", fontWeight: 800, letterSpacing: "-0.025em", marginBottom: 16 }}>
              One avoided test campaign pays for{" "}
              <span className="gradient-text">a year of access.</span>
            </h2>
            <p style={{ fontSize: "1rem", color: "var(--color-muted)", marginBottom: 36, maxWidth: 480, margin: "0 auto 36px" }}>
              A single physical coupon testing campaign costs $200k+. ALLOY IQ narrows your candidate space before you commission a single sample.
            </p>
            <Link href="/predict">
              <button className="btn-glow" style={{ fontSize: "1rem", padding: "14px 36px" }}>
                Try a Free Prediction →
              </button>
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer style={{
        borderTop: "1px solid var(--color-border)", padding: "28px 0",
        textAlign: "center", color: "var(--color-muted)", fontSize: "0.82rem",
      }}>
        <div className="container">
          © {new Date().getFullYear()} ALLOY IQ · AI-Powered Materials Science Platform
        </div>
      </footer>
    </main>
  );
}
