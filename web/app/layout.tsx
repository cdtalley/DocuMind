import "./globals.css";
import type { Metadata } from "next";
import { DM_Sans, Outfit } from "next/font/google";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  display: "swap"
});

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
  display: "swap"
});

export const metadata: Metadata = {
  title: "DocuMind · RAG over a document library",
  description:
    "FastAPI + ChromaDB + Ollama: ingest PDF/DOCX/TXT, optional arXiv fetch, vector search, mode-specific prompts, citations, optional FLARE-style second retrieval. Next.js dashboard."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${dmSans.variable} ${outfit.variable} ${dmSans.className}`}>{children}</body>
    </html>
  );
}
