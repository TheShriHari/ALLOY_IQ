"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Atom, Menu, X } from "lucide-react";

const NAV_LINKS = [
  { href: "/",             label: "Home"      },
  { href: "/predict",      label: "Predict"   },
  { href: "/inverse",      label: "Inverse Design" },
  { href: "/microstructure", label: "Microstructure" },
  { href: "/history",      label: "History"   },
];

export function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
      background: "rgba(8, 11, 20, 0.85)", backdropFilter: "blur(20px)",
      borderBottom: "1px solid rgba(99, 130, 255, 0.12)",
    }}>
      <div className="container" style={{ display: "flex", alignItems: "center", height: 64, gap: 32 }}>
        {/* Logo */}
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none" }}>
          <div style={{
            width: 34, height: 34, borderRadius: 10,
            background: "linear-gradient(135deg, #6382FF, #00E5FF)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Atom size={18} color="#fff" />
          </div>
          <span style={{ fontWeight: 700, fontSize: "1.1rem", letterSpacing: "-0.02em" }}>
            <span className="gradient-text">ALLOY</span>
            <span style={{ color: "#E8EEFF" }}> IQ</span>
          </span>
        </Link>

        {/* Desktop links */}
        <div style={{ display: "flex", gap: 4, marginLeft: "auto" }} className="hidden-mobile">
          {NAV_LINKS.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link key={href} href={href} style={{
                padding: "6px 14px", borderRadius: 8, fontSize: "0.875rem", fontWeight: 500,
                textDecoration: "none",
                color: active ? "#fff" : "var(--color-muted)",
                background: active ? "rgba(99, 130, 255, 0.15)" : "transparent",
                border: active ? "1px solid rgba(99, 130, 255, 0.25)" : "1px solid transparent",
                transition: "all 0.15s",
              }}>
                {label}
              </Link>
            );
          })}
        </div>

        {/* CTA */}
        <Link href="/predict" style={{ marginLeft: 16 }} className="hidden-mobile">
          <button className="btn-glow" style={{ padding: "8px 20px", fontSize: "0.85rem" }}>
            Start Predicting →
          </button>
        </Link>

        {/* Mobile toggle */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", color: "#fff" }}
          className="show-mobile"
        >
          {mobileOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div style={{
          background: "var(--color-surface)", borderTop: "1px solid var(--color-border)",
          padding: "16px 24px 20px",
        }}>
          {NAV_LINKS.map(({ href, label }) => (
            <Link key={href} href={href}
              onClick={() => setMobileOpen(false)}
              style={{
                display: "block", padding: "10px 0", fontSize: "0.95rem",
                color: pathname === href ? "var(--color-primary)" : "var(--color-text)",
                textDecoration: "none", borderBottom: "1px solid var(--color-border)",
              }}>
              {label}
            </Link>
          ))}
        </div>
      )}

      <style>{`
        @media (max-width: 768px) {
          .hidden-mobile { display: none !important; }
          .show-mobile   { display: flex !important; }
        }
        @media (min-width: 769px) {
          .show-mobile { display: none !important; }
        }
      `}</style>
    </nav>
  );
}
