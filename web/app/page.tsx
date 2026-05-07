"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Source = {
  doc_id: string;
  paper_title: string;
  year: string;
  section: string;
  content_preview: string;
  distance: number;
};

type PaperCard = {
  doc_id: string;
  filename: string;
  title: string;
  authors: string;
  year: string;
  arxiv_id: string;
  chunk_count: number;
};

type HealthPayload = {
  ollama_available: boolean;
  llm_model: string;
  embedding_model: string;
  collection_stats: { paper_count: number; total_chunks: number; collection_name: string };
};

const modes = [
  { label: "General Q&A", value: "general" },
  { label: "Compare Methods", value: "compare" },
  { label: "Methodology Deep Dive", value: "methodology" },
  { label: "Dataset Finder", value: "datasets" },
  { label: "Reproduce Results", value: "reproduce" }
];

export default function HomePage() {
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [papers, setPapers] = useState<PaperCard[]>([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("general");
  const [section, setSection] = useState("All Sections");
  const [topK, setTopK] = useState(6);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [confidence, setConfidence] = useState(0);
  const [arxivId, setArxivId] = useState("");
  const [notice, setNotice] = useState("");
  const [apiHealthy, setApiHealthy] = useState(true);

  const libraryStats = useMemo(
    () => ({
      totalPapers: papers.length,
      totalChunks: papers.reduce((acc, paper) => acc + paper.chunk_count, 0)
    }),
    [papers]
  );

  const fetchJson = async <T,>(path: string, options?: RequestInit): Promise<T> => {
    const response = await fetch(`${API_BASE_URL}${path}`, options);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const payload = (await response.json()) as { detail?: string };
        if (payload?.detail) detail = payload.detail;
      } catch {
        // ignore json parse failures
      }
      throw new Error(detail);
    }
    return (await response.json()) as T;
  };

  const refresh = async () => {
    try {
      const [healthPayload, papersPayload] = await Promise.all([
        fetchJson<HealthPayload>("/health"),
        fetchJson<PaperCard[]>("/api/v1/papers")
      ]);
      setHealth(healthPayload);
      setPapers(papersPayload);
      setApiHealthy(true);
    } catch {
      setApiHealthy(false);
      setHealth(null);
      setPapers([]);
      setNotice(`API unreachable at ${API_BASE_URL}. Start backend: uvicorn app.main:app --reload`);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const askPapers = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setNotice("");
    try {
      const data = await fetchJson<{
        answer: string;
        sources: Source[];
        confidence: number;
      }>("/api/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: topK,
          query_mode: mode,
          section_filter: section === "All Sections" ? null : section
        })
      });
      setAnswer(data.answer);
      setSources(data.sources || []);
      setConfidence(Number(data.confidence || 0));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  const uploadFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setLoading(true);
    setNotice("");
    try {
      for (const file of Array.from(files)) {
        const form = new FormData();
        form.append("file", file);
        await fetchJson(`/api/v1/ingest`, { method: "POST", body: form });
      }
      setNotice("Upload complete.");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const fetchArxiv = async () => {
    if (!arxivId.trim()) return;
    setLoading(true);
    setNotice("");
    try {
      const data = await fetchJson<{ title: string }>("/api/v1/fetch-arxiv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId.trim() })
      });
      setNotice(`Fetched: ${data.title}`);
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "ArXiv fetch failed");
    } finally {
      setLoading(false);
    }
  };

  const deletePaper = async (docId: string) => {
    try {
      await fetchJson(`/api/v1/papers/${docId}`, { method: "DELETE" });
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Delete failed");
    }
  };

  return (
    <main className="layout">
      <aside className="sidebar">
        <h1>🔬 DocuMind</h1>
        <p className="pill">Interview Demo Build</p>
        <div className="grid" style={{ marginTop: 16 }}>
          <div className="card">
            <strong>Ollama</strong>
            <p>{apiHealthy ? (health?.ollama_available ? "Online" : "Offline") : "API Down"}</p>
            <p>LLM: {health?.llm_model ?? "-"}</p>
            <p>Embed: {health?.embedding_model ?? "-"}</p>
          </div>
          <div className="card">
            <strong>Collection</strong>
            <p>Papers: {health?.collection_stats.paper_count ?? libraryStats.totalPapers}</p>
            <p>Chunks: {health?.collection_stats.total_chunks ?? libraryStats.totalChunks}</p>
          </div>
          <button onClick={() => void refresh()}>Refresh Status</button>
        </div>
      </aside>

      <section className="content grid">
        <div className="card">
          <h2>Ask Across Papers</h2>
          <form className="grid two" onSubmit={askPapers}>
            <div style={{ gridColumn: "1 / span 2" }}>
              <label>Question</label>
              <textarea
                rows={4}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What datasets were used for tabular fraud detection?"
              />
            </div>
            <div>
              <label>Mode</label>
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                {modes.map((m) => (
                  <option value={m.value} key={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label>Section</label>
              <select value={section} onChange={(e) => setSection(e.target.value)}>
                {["All Sections", "abstract", "introduction", "methodology", "experiments", "results", "conclusion"].map(
                  (s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  )
                )}
              </select>
            </div>
            <div>
              <label>Top K</label>
              <input
                type="number"
                min={3}
                max={15}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>
            <div style={{ alignSelf: "end" }}>
              <button type="submit" disabled={loading}>
                {loading ? "Running..." : "Run Query"}
              </button>
            </div>
          </form>
          {answer && (
            <div className="card" style={{ marginTop: 16, borderLeft: "4px solid #22c55e" }}>
              <h3>Answer</h3>
              <p>{answer}</p>
              <p>Confidence: {(confidence * 100).toFixed(0)}%</p>
              {sources.map((source, index) => (
                <details key={`${source.doc_id}-${index}`} style={{ marginTop: 8 }}>
                  <summary>
                    Source {index + 1}: {source.paper_title} ({source.year}) — {source.section}
                  </summary>
                  <p>{source.content_preview}</p>
                </details>
              ))}
            </div>
          )}
        </div>

        <div className="grid two">
          <div className="card">
            <h2>Upload Papers</h2>
            <input type="file" multiple accept=".pdf,.docx,.txt" onChange={(e) => void uploadFiles(e.target.files)} />
          </div>
          <div className="card">
            <h2>Fetch from ArXiv</h2>
            <input value={arxivId} onChange={(e) => setArxivId(e.target.value)} placeholder="1706.03762" />
            <button style={{ marginTop: 10 }} onClick={() => void fetchArxiv()}>
              Fetch
            </button>
          </div>
        </div>

        <div className="card">
          <h2>Paper Library</h2>
          {papers.length === 0 ? (
            <p>Your library is empty. Upload a paper or fetch one from ArXiv.</p>
          ) : (
            <div className="grid">
              {papers.map((paper) => (
                <div className="card" key={paper.doc_id}>
                  <strong>📄 {paper.title}</strong>
                  <p>
                    {paper.authors} — {paper.year} — Chunks: {paper.chunk_count}
                  </p>
                  {paper.arxiv_id && (
                    <a href={`https://arxiv.org/abs/${paper.arxiv_id}`} target="_blank">
                      arXiv:{paper.arxiv_id}
                    </a>
                  )}
                  <button style={{ marginTop: 10 }} onClick={() => void deletePaper(paper.doc_id)}>
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {notice && <div className="card">{notice}</div>}
      </section>
    </main>
  );
}
