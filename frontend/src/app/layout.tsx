import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "ALLOY IQ — AI-Powered Materials Property Prediction",
  description:
    "Predict, interpret, and optimize mechanical and corrosion properties of steels, high-entropy alloys, and aluminum alloys with SHAP explainability and inverse design.",
  keywords: ["materials science", "alloy design", "machine learning", "SHAP", "metallurgy"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-[#080B14] text-white antialiased">
        <Navbar />
        {children}
      </body>
    </html>
  );
}
