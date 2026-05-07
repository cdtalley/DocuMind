import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DocuMind",
  description: "Research paper intelligence with local Ollama RAG"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
