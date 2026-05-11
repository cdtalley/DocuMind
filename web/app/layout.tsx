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
  title: "DocuMind · Retrieval-augmented research library",
  description:
    "FastAPI: Chroma (cosine) persistent index, Ollama embeddings and chat, mode-specific retrieval budgets, keyword-weighted rerank, optional FLARE follow-up. OpenAPI v1, live/ready health split. Next.js operator UI."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${dmSans.variable} ${outfit.variable} ${dmSans.className}`}>{children}</body>
    </html>
  );
}
