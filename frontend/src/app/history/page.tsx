"use client";

import { useEffect, useState } from "react";
import { Clock, FlaskConical, TrendingUp, Zap, Loader2 } from "lucide-react";
import axios from "axios";

interface HistoryItem {
  job_id: string;
  alloy_family: string;
  property: string;
  prediction: number | null;
  unit?: string;
  confidence?: string;
  created_at: string;
  status: string;
}

const FAMILY_ICONS: Record<string, React.ReactNode> = {
  steel:    <FlaskConical size={16} />,
  hea:      <Zap size={16} />,
  aluminum: <TrendingUp size={16} />,
};
const FAMILY_COLORS: Record<string, string> = {
  steel: "#6382FF", hea: "#00E5FF", aluminum: "#FFB547",
};

function timeAgo(iso: string) {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
  return `${Math.floor(diff/86400)}d ago`;
}

export default function HistoryPage() {
  const [filter, setFilter] = useState<string>("all");
  const [historyData, setHistoryData] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchHistory() {
      try {
        const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const { data } = await axios.get(`${API_URL}/history`);
        // Assign units and confidences since backend GET /history doesn't return full details yet
        const enriched = data.map((item: any) => ({
          ...item,
          unit: item.property.includes("pren") ? "PREN" : item.property.includes("toughness") ? "MPa√m" : item.property.includes("hardness") ? "HV" : "MPa",
          confidence: "high" // placeholder for history list
        }));
        setHistoryData(enriched);
      } catch (e) {
        console.error("Failed to load history", e);
      } finally {
        setLoading(false);
      }
    }
    fetchHistory();
  }, []);

  const families = ["all", "steel", "hea", "aluminum"];
  const filtered = filter === "all" ? historyData : historyData.filter(j => j.alloy_family === filter);

  return (
    <main style={{ paddingTop: 80 }}>
      <div className="container section" style={{ paddingTop: 40 }}>
        <h1 style={{ fontSize: "clamp(1.8rem, 3.5vw, 2.5rem)", fontWeight: 800, letterSpacing: "-0.03em", marginBottom: 8 }}>
          Prediction <span className="gradient-text">History</span>
        </h1>
        <p style={{ color: "var(--color-muted)", fontSize: "0.95rem", marginBottom: 32 }}>
          All your past predictions, stored and searchable.
        </p>

        {/* Summary cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, marginBottom: 36 }}>
          {[
            { label: "Total Predictions", value: historyData.length, icon: <Clock size={18} /> },
            { label: "Steel",    value: historyData.filter(j => j.alloy_family === "steel").length,    icon: <FlaskConical size={18} /> },
            { label: "HEA",      value: historyData.filter(j => j.alloy_family === "hea").length,      icon: <Zap size={18} /> },
            { label: "Aluminum", value: historyData.filter(j => j.alloy_family === "aluminum").length, icon: <TrendingUp size={18} /> },
          ].map(({ label, value, icon }) => (
            <div key={label} className="glass" style={{ padding: "20px 24px", display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{ color: "var(--color-primary)", opacity: 0.8 }}>{icon}</div>
              <div>
                <div style={{ fontSize: "1.8rem", fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1 }} className="gradient-text">
                  {value}
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--color-muted)", marginTop: 2 }}>{label}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Filter */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
          {families.map(f => (
            <button key={f} onClick={() => setFilter(f)}
              style={{
                padding: "7px 18px", borderRadius: 8, cursor: "pointer", fontSize: "0.82rem", fontWeight: 600,
                border: filter === f ? "1px solid var(--color-primary)" : "1px solid var(--color-border)",
                background: filter === f ? "rgba(99,130,255,0.15)" : "var(--color-surface-2)",
                color: filter === f ? "var(--color-primary)" : "var(--color-muted)",
                transition: "all 0.15s", textTransform: "capitalize",
              }}>
              {f === "all" ? "All Families" : f.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Table */}
        <div className="glass" style={{ overflow: "hidden" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
                {["Family", "Property", "Prediction", "Confidence", "Time", ""].map(col => (
                  <th key={col} style={{
                    padding: "14px 20px", textAlign: "left", fontSize: "0.75rem",
                    fontWeight: 700, color: "var(--color-muted)", letterSpacing: "0.06em", textTransform: "uppercase",
                  }}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((job, i) => (
                <tr key={job.job_id} style={{
                  borderBottom: i < filtered.length - 1 ? "1px solid var(--color-border)" : "none",
                  transition: "background 0.15s",
                }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = "rgba(99,130,255,0.04)"}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = "transparent"}
                >
                  <td style={{ padding: "14px 20px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ color: FAMILY_COLORS[job.alloy_family] }}>
                        {FAMILY_ICONS[job.alloy_family]}
                      </span>
                      <span style={{ fontSize: "0.85rem", fontWeight: 600, textTransform: "capitalize" }}>
                        {job.alloy_family.toUpperCase()}
                      </span>
                    </div>
                  </td>
                  <td style={{ padding: "14px 20px", fontSize: "0.85rem", color: "var(--color-text)" }}>
                    {job.property.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                  </td>
                  <td style={{ padding: "14px 20px" }}>
                    <span style={{ fontWeight: 700, color: "var(--color-primary)", fontSize: "0.95rem" }}>
                      {job.prediction ? job.prediction.toFixed(1) : "—"}
                    </span>
                    <span style={{ fontSize: "0.78rem", color: "var(--color-muted)", marginLeft: 4 }}>
                      {job.unit}
                    </span>
                  </td>
                  <td style={{ padding: "14px 20px" }}>
                    <span className={`badge badge-${job.confidence}`}>
                      {job.confidence}
                    </span>
                  </td>
                  <td style={{ padding: "14px 20px", fontSize: "0.8rem", color: "var(--color-muted)" }}>
                    {timeAgo(job.created_at)}
                  </td>
                  <td style={{ padding: "14px 20px" }}>
                    <button className="btn-ghost" style={{ padding: "5px 12px", fontSize: "0.78rem" }}>
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}
