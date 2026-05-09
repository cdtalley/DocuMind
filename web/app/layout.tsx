import "./globals.css";
import type { Metadata } from "next";
import { DM_Sans } from "next/font/google";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  display: "swap"
});

export const metadata: Metadata = {
  title: "DocuMind · Enterprise research RAG",
  description:
    "Local-first retrieval-augmented generation over your paper library: ChromaDB vectors, Ollama LLMs, FastAPI services, citation-backed answers, and audit-oriented query modes."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${dmSans.variable} ${dmSans.className}`}>{children}</body>
    </html>
  );
}
