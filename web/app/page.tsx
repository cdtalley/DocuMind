"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getApiBaseUrl } from "../lib/api-base";
/** Increment when shipping visible UI or diagnostics changes. */
const DASHBOARD_UI_VERSION = "1.1.0";

export type ShowcaseScenario = {
  id: string;
  label: string;
  description: string;
  query: string;
  mode: string;
  topK: number;
  /** When set, applies the FLARE follow-up retrieval toggle for this scenario. */
  useFlare?: boolean;
};

const SHOWCASE_SCENARIOS: ShowcaseScenario[] = [
  {
    id: "baseline",
    label: "Baseline: evidence-first summary",
    description: "Grounded summary with explicit coverage limits.",
    query: `Using ONLY the retrieved encyclopedia-style passages, write a concise answer for a general reader.

Rules:
- Every non-trivial claim must be traceable to a cited **Article title** from the context.
- If the passages disagree or omit a subtopic, say so explicitly — do not invent facts.
- End with a short "Coverage" line: what themes the excerpts did and did not support.`,
    mode: "general",
    topK: 10
  },
  {
    id: "compare_articles",
    label: "Compare articles",
    description: "Structured contrast across articles the retriever surfaced — compare mode + FLARE.",
    query: `You are synthesizing ONLY from the retrieved article excerpts (public index).

Produce this outline in Markdown:
## At a glance
## Where the articles agree
## Where they disagree or leave gaps (quote titles)
## Comparison table (GFM): Theme | What article A states (title) | What article B states (title) | Confidence from excerpts only
## What a reader should verify next

Rules: Use **Article title** strings exactly as they appear in context. If the set lacks a second article on a row, write "not in excerpt set". No external facts.

Screenshot / regression note: Prefer compact tables. If fewer than three distinct article titles appear in the excerpts, say the comparison is partial and enumerate every title you did see. If any theme has thin evidence, label it and avoid filler.`,
    mode: "compare",
    topK: 16,
    useFlare: true
  },
  {
    id: "themes",
    label: "Cross-article themes",
    description: "Cluster recurring topics and cite which articles support each theme.",
    query: `From the retrieved passages only, cluster the major themes (history, geography, institutions, science concepts, etc. — whatever the excerpts actually contain).

For each theme: 2–4 bullets, each bullet tied to a specific **Article title** from the context. If a theme appears weakly, label it "thin evidence" and explain why.`,
    mode: "compare",
    topK: 14
  },
  {
    id: "entities",
    label: "People, places, institutions",
    description: "Entity-centric inventory grounded in chunk text.",
    query: `List notable people, places, organizations, dates, and laws or treaties mentioned in the retrieved excerpts.

Format as a table: Entity | Role in excerpts (one line) | Article title(s) that mention it.

Do not add entities not present in the context. If the excerpt set is sparse, say so up front.`,
    mode: "datasets",
    topK: 12
  },
  {
    id: "timeline",
    label: "Chronology from excerpts",
    description: "Orders dated statements; reproduce mode favors extraction discipline.",
    query: `Extract every dated or ordered historical statement you can support from the retrieved text. Output a chronological bullet list: date or era — what happened — **Article title**.

If dates conflict between articles, show both versions and the titles. If dating is vague, mark as "approximate / unclear in excerpts".`,
    mode: "reproduce",
    topK: 12
  }
];

const DEFAULT_SCENARIO = SHOWCASE_SCENARIOS[0];

const INITIAL_PUBLIC_QUERY = DEFAULT_SCENARIO.query;

type Source = {
  doc_id: string;
  paper_title: string;
  authors: string;
  year: string;
  section: string;
  chunk_index: number;
  page_number: number;
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

type CollectionStatsPayload = {
  paper_count: number;
  total_chunks: number;
  collection_name: string;
};

type LibrariesPayload = {
  public: CollectionStatsPayload;
  papers: CollectionStatsPayload;
  default_library: string;
};

type QueryResponse = {
  answer: string;
  sources: Source[];
  confidence: number;
  has_answer: boolean;
  query_mode?: string;
  chunks_searched?: number;
  model_used?: string;
  flare_enabled?: boolean;
  flare_followup_retrieval?: boolean;
  library?: string;
};

type DiagnosticsPayload = {
  api_version: string;
  openapi_disabled: boolean;
  git_sha: string;
  app_env: string;
  process_started_at_utc: string;
  uptime_seconds: number;
  python_version: string;
  seed_sample_docs: boolean;
  sample_corpus_version: string;
  default_library: string;
  chroma_persist_basename: string;
  chroma_collection_public: string;
  chroma_collection_papers: string;
  chunk_size: number;
  chunk_overlap: number;
  top_k_default: number;
  relevance_threshold_papers: number;
  public_relevance_threshold: number;
  keyword_rerank_weight_papers: number;
  public_keyword_rerank_weight: number;
  enable_fallback_retrieval: boolean;
  flare_active_retrieval_default: boolean;
  public_chunks: number;
  public_docs: number;
  papers_chunks: number;
  papers_docs: number;
};

function formatUptime(totalSec: number): string {
  const s = Math.max(0, Math.floor(totalSec % 60));
  const m = Math.max(0, Math.floor(totalSec / 60) % 60);
  const h = Math.floor(totalSec / 3600);
  if (h > 48) return `${Math.floor(h / 24)}d ${h % 24}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

const modes = [
  { label: "General Q&A", value: "general" },
  { label: "Compare across articles", value: "compare" },
  { label: "Topic deep dive", value: "methodology" },
  { label: "Entity & fact inventory", value: "datasets" },
  { label: "Chronology / provenance", value: "reproduce" }
];

type NoticeTone = "info" | "success" | "error";

/** Rich Markdown mapping: section cards, sticky tables, callouts — tuned for long RAG answers. */
const MARKDOWN_COMPONENTS: Components = {
  h1: ({ children }) => (
    <div className="md-section md-section--major">
      <div className="md-section__accent" aria-hidden />
      <h1 className="md-h md-h1">{children}</h1>
    </div>
  ),
  h2: ({ children }) => (
    <div className="md-section md-section--major">
      <div className="md-section__accent" aria-hidden />
      <h2 className="md-h md-h2">{children}</h2>
    </div>
  ),
  h3: ({ children }) => <h3 className="md-h3">{children}</h3>,
  h4: ({ children }) => <h4 className="md-h4">{children}</h4>,
  p: ({ children }) => <p className="md-p">{children}</p>,
  ul: ({ children }) => <ul className="md-ul">{children}</ul>,
  ol: ({ children }) => <ol className="md-ol">{children}</ol>,
  li: ({ children, className }) => (
    <li className={className ? `md-li ${className}` : "md-li"}>{children}</li>
  ),
  blockquote: ({ children }) => <blockquote className="md-callout">{children}</blockquote>,
  hr: () => <hr className="md-hr" />,
  table: ({ children }) => (
    <div className="md-table-shell">
      <table className="md-table">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead>{children}</thead>,
  tbody: ({ children }) => <tbody>{children}</tbody>,
  tr: ({ children }) => <tr className="md-tr">{children}</tr>,
  th: ({ children }) => <th className="md-th">{children}</th>,
  td: ({ children }) => <td className="md-td">{children}</td>,
  pre: ({ children }) => <pre className="md-pre">{children}</pre>,
  code: ({ className, children, ...props }) => {
    const isBlock = Boolean(className?.includes("language-"));
    return (
      <code className={isBlock ? `md-code-block ${className ?? ""}` : "md-code-inline"} {...props}>
        {children}
      </code>
    );
  },
  a: ({ href, children }) => (
    <a href={href} className="md-a" target="_blank" rel="noreferrer noopener">
      {children as ReactNode}
    </a>
  ),
  strong: ({ children }) => <strong className="md-strong">{children}</strong>
};

export default function HomePage() {
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [papers, setPapers] = useState<PaperCard[]>([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState(INITIAL_PUBLIC_QUERY);
  const [mode, setMode] = useState(DEFAULT_SCENARIO.mode);
  const [topK, setTopK] = useState(DEFAULT_SCENARIO.topK);
  const [useFlare, setUseFlare] = useState(Boolean(DEFAULT_SCENARIO.useFlare));
  const [flareFollowUp, setFlareFollowUp] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [confidence, setConfidence] = useState(0);
  const [hasAnswer, setHasAnswer] = useState(true);
  const [chunksSearched, setChunksSearched] = useState<number | null>(null);
  const [notice, setNotice] = useState("");
  const [noticeTone, setNoticeTone] = useState<NoticeTone>("info");
  const [apiHealthy, setApiHealthy] = useState(true);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [modelUsed, setModelUsed] = useState<string | null>(null);
  const [libraries, setLibraries] = useState<LibrariesPayload | null>(null);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsPayload | null>(null);
  const [bootstrapping, setBootstrapping] = useState(true);

  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  const libraryStats = useMemo(
    () => ({
      totalPapers: papers.length,
      totalChunks: papers.reduce((acc, paper) => acc + paper.chunk_count, 0)
    }),
    [papers]
  );

  const fetchJson = async <T,>(path: string, options?: RequestInit): Promise<T> => {
    const response = await fetch(`${apiBaseUrl}${path}`, options);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const payload = (await response.json()) as { detail?: string };
        if (payload?.detail) detail = String(payload.detail);
      } catch {
        // ignore
      }
      throw new Error(detail);
    }
    return (await response.json()) as T;
  };

  const refresh = async (): Promise<boolean> => {
    try {
      const [healthPayload, papersPayload, librariesPayload, diagPayload] = await Promise.all([
        fetchJson<HealthPayload>("/health"),
        fetchJson<PaperCard[]>(`/api/v1/papers?library=${encodeURIComponent("public")}`),
        fetchJson<LibrariesPayload>("/api/v1/libraries"),
        fetchJson<DiagnosticsPayload>("/api/v1/diagnostics")
      ]);
      setHealth(healthPayload);
      setPapers(papersPayload);
      setLibraries(librariesPayload);
      setDiagnostics(diagPayload);
      setApiHealthy(true);
      setLastSync(new Date());
      setNotice("");
      return true;
    } catch {
      setApiHealthy(false);
      setHealth(null);
      setPapers([]);
      setLibraries(null);
      setDiagnostics(null);
      setNotice(
        `API unreachable at ${apiBaseUrl}. Run: .\\start_documind.ps1 from the project root (or uvicorn on port 8001).`
      );
      setNoticeTone("error");
      return false;
    }
  };

  useEffect(() => {
    let cancelled = false;
    const bootstrap = async () => {
      for (let attempt = 0; attempt < 36 && !cancelled; attempt++) {
        const ok = await refresh();
        if (ok) break;
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
      if (!cancelled) setBootstrapping(false);
    };
    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  const runQuery = useCallback(async () => {
    setLoading(true);
    setNotice("");
    setNoticeTone("info");
    try {
      const data = await fetchJson<QueryResponse>("/api/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          library: "public",
          top_k: topK,
          query_mode: mode,
          section_filter: null,
          use_flare: useFlare
        })
      });
      setAnswer(data.answer);
      setSources(data.sources || []);
      setConfidence(Number(data.confidence ?? 0));
      setHasAnswer(data.has_answer !== false);
      setChunksSearched(typeof data.chunks_searched === "number" ? data.chunks_searched : null);
      setModelUsed(typeof data.model_used === "string" ? data.model_used : null);
      setFlareFollowUp(data.flare_followup_retrieval === true);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Query failed");
      setNoticeTone("error");
    } finally {
      setLoading(false);
    }
  }, [query, topK, mode, useFlare]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Enter" || (!e.ctrlKey && !e.metaKey)) return;
      if (!(e.target instanceof HTMLTextAreaElement)) return;
      e.preventDefault();
      if (!loading) void runQuery();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [loading, runQuery]);

  const askPapers = async (event: FormEvent) => {
    event.preventDefault();
    await runQuery();
  };

  const applyScenario = (s: ShowcaseScenario) => {
    setQuery(s.query);
    setMode(s.mode);
    setTopK(s.topK);
    setUseFlare(typeof s.useFlare === "boolean" ? s.useFlare : false);
    setNotice(`Loaded scenario: ${s.label} (public index)`);
    setNoticeTone("success");
  };

  const loadDemoPreset = () => {
    applyScenario(DEFAULT_SCENARIO);
  };

  const uploadFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setLoading(true);
    setNotice("");
    try {
      for (const file of Array.from(files)) {
        const form = new FormData();
        form.append("file", file);
        form.append("library", "public");
        await fetchJson(`/api/v1/ingest`, { method: "POST", body: form });
      }
      setNotice("Upload complete — public index refreshed.");
      setNoticeTone("success");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Upload failed");
      setNoticeTone("error");
    } finally {
      setLoading(false);
    }
  };

  const copyAnswer = async () => {
    if (!answer) return;
    try {
      await navigator.clipboard.writeText(answer);
      setNotice("Synthesis copied to clipboard.");
      setNoticeTone("success");
    } catch {
      setNotice("Clipboard unavailable in this context.");
      setNoticeTone("error");
    }
  };

  const deletePaper = async (docId: string) => {
    try {
      await fetchJson(`/api/v1/papers/${docId}?library=public`, { method: "DELETE" });
      setNotice("Article removed from public index.");
      setNoticeTone("success");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Delete failed");
      setNoticeTone("error");
    }
  };

  const noticeClass =
    noticeTone === "error" ? "notice notice--error" : noticeTone === "success" ? "notice notice--success" : "notice";
  const noticeA11y =
    noticeTone === "error"
      ? { role: "alert" as const, "aria-live": "assertive" as const }
      : notice
        ? { role: "status" as const, "aria-live": "polite" as const }
        : {};

  const ollamaOk = Boolean(apiHealthy && health?.ollama_available);
  const apiDocsUrl = `${apiBaseUrl.replace(/\/$/, "")}/docs`;
  const syncLabel = lastSync
    ? lastSync.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "—";

  const indexedPapers = health?.collection_stats.paper_count ?? libraryStats.totalPapers;
  const indexedChunks = health?.collection_stats.total_chunks ?? libraryStats.totalChunks;
  const publicArticleCount = libraries?.public.paper_count ?? indexedPapers;
  const publicChunkCount = libraries?.public.total_chunks ?? indexedChunks;

  return (
    <div className="app-root">
      <a href="#workspace" className="skip-link">
        Skip to workspace
      </a>
      <header className="enterprise-topbar" role="banner">
        <div className="enterprise-topbar__brand">
          <span className="enterprise-topbar__logo" aria-hidden />
          <div>
            <div className="enterprise-topbar__title">DocuMind</div>
            <div className="enterprise-topbar__subtitle">
              Public index · FastAPI · Ollama · Chroma · UI v{DASHBOARD_UI_VERSION}
              {libraries ? (
                <span className="enterprise-topbar__default-lib">
                  {" "}
                  · Console queries <strong>public</strong> · Papers index:{" "}
                  <strong>{libraries.papers.total_chunks.toLocaleString()}</strong> vectors
                </span>
              ) : null}
            </div>
          </div>
        </div>
        <div className="enterprise-topbar__status" aria-live="polite">
          <span className={`status-chip ${apiHealthy ? "status-chip--ok" : "status-chip--bad"}`}>
            <span className="status-dot" /> API
          </span>
          <span className={`status-chip ${ollamaOk ? "status-chip--ok" : apiHealthy ? "status-chip--warn" : "status-chip--bad"}`}>
            <span className="status-dot" /> Inference
          </span>
          {libraries ? (
            <>
              <span
                className="status-chip status-chip--neutral status-chip--stat"
                title={libraries.public.collection_name}
              >
                <span className="status-dot" /> Public · {libraries.public.paper_count.toLocaleString()} docs ·{" "}
                {libraries.public.total_chunks.toLocaleString()} chk
              </span>
              <span
                className="status-chip status-chip--neutral status-chip--stat"
                title={libraries.papers.collection_name}
              >
                <span className="status-dot" /> Papers · {libraries.papers.paper_count.toLocaleString()} docs ·{" "}
                {libraries.papers.total_chunks.toLocaleString()} chk
              </span>
            </>
          ) : (
            <span className="status-chip status-chip--neutral">
              <span className="status-dot" /> {indexedPapers} docs · {indexedChunks.toLocaleString()} chunks
            </span>
          )}
          {diagnostics ? (
            <span
              className="status-chip status-chip--neutral status-chip--stat"
              title={`Process started ${diagnostics.process_started_at_utc || "—"} (UTC). GET /api/v1/diagnostics`}
            >
              <span className="status-dot" /> API v{diagnostics.api_version} · up {formatUptime(diagnostics.uptime_seconds)}
            </span>
          ) : null}
          <span className="enterprise-topbar__sync">Synced {syncLabel}</span>
        </div>
        <div className="enterprise-topbar__links">
          <a className="topbar-link" href={apiDocsUrl} target="_blank" rel="noreferrer">
            OpenAPI
          </a>
          <code className="topbar-code">{apiBaseUrl}</code>
        </div>
      </header>

      <p className="app-context-line">
        Same REST surface as integrations: <code>/health</code>, <code>/api/v1/diagnostics</code>,{" "}
        <code>/api/v1/libraries</code>, <code>/api/v1/query</code>.
      </p>

      <main className="layout">
        <aside className="sidebar">
          <h1 className="sidebar-title">Status</h1>
          <p className="sidebar-tagline">
            <code>/health</code>, <code>/api/v1/diagnostics</code>, <code>/api/v1/libraries</code>,{" "}
            <code>/api/v1/papers?library=public</code>. Refresh after bulk index or ingest.
          </p>
          <p className="pill">Ollama · Chroma · FastAPI</p>
          <p className="api-hint">Dashboard uses this endpoint for all requests.</p>
          <div className="grid" style={{ marginTop: 16 }}>
            <div className="card card--inset">
              <strong className="sidebar-card-label">Inference stack</strong>
              <p className={`sidebar-status ${ollamaOk ? "sidebar-status--ok" : ""}`}>
                {apiHealthy ? (health?.ollama_available ? "Operational" : "Degraded — check Ollama") : "Unreachable"}
              </p>
              <p className="sidebar-metric">LLM · {health?.llm_model ?? "—"}</p>
              <p className="sidebar-metric">Embed · {health?.embedding_model ?? "—"}</p>
            </div>
            <div className="card card--inset">
              <strong className="sidebar-card-label">Diagnostics</strong>
              {diagnostics ? (
                <>
                  <p className="sidebar-metric">
                    <strong>Runtime</strong> · {diagnostics.app_env} · {diagnostics.python_version}
                  </p>
                  {diagnostics.git_sha ? (
                    <p className="sidebar-metric">
                      <strong>Build</strong> · <code className="topbar-code">{diagnostics.git_sha}</code>
                    </p>
                  ) : null}
                  <p className="sidebar-metric">
                    <strong>Chroma volume</strong> ·{" "}
                    <code className="topbar-code">{diagnostics.chroma_persist_basename}</code>
                    {diagnostics.openapi_disabled ? (
                      <>
                        {" "}
                        · OpenAPI off
                      </>
                    ) : null}
                  </p>
                  <div className="diagnostics-kv">
                    <span>Default library</span>
                    <code>{diagnostics.default_library}</code>
                  </div>
                  <div className="diagnostics-kv">
                    <span>Sample bundle ver</span>
                    <code>{diagnostics.sample_corpus_version}</code>
                  </div>
                  <div className="diagnostics-kv">
                    <span>Chunk / overlap</span>
                    <code>
                      {diagnostics.chunk_size} / {diagnostics.chunk_overlap}
                    </code>
                  </div>
                  <div className="diagnostics-kv">
                    <span>Default top_k (settings)</span>
                    <code>{diagnostics.top_k_default}</code>
                  </div>
                  <details className="diag-details">
                    <summary>Retrieval thresholds (live)</summary>
                    <div className="diagnostics-kv">
                      <span>Public cosine gate</span>
                      <code>{diagnostics.public_relevance_threshold}</code>
                    </div>
                    <div className="diagnostics-kv">
                      <span>Public keyword W</span>
                      <code>{diagnostics.public_keyword_rerank_weight}</code>
                    </div>
                    <div className="diagnostics-kv">
                      <span>Papers cosine gate</span>
                      <code>{diagnostics.relevance_threshold_papers}</code>
                    </div>
                    <div className="diagnostics-kv">
                      <span>Papers keyword W</span>
                      <code>{diagnostics.keyword_rerank_weight_papers}</code>
                    </div>
                    <div className="diagnostics-kv">
                      <span>Fallback retrieval</span>
                      <code>{diagnostics.enable_fallback_retrieval ? "on" : "off"}</code>
                    </div>
                    <div className="diagnostics-kv">
                      <span>FLARE default (env)</span>
                      <code>{diagnostics.flare_active_retrieval_default ? "on" : "off"}</code>
                    </div>
                  </details>
                  <details className="diag-details">
                    <summary>Index counts (diagnostics echo)</summary>
                    <div className="diagnostics-kv">
                      <span>Public docs / vectors</span>
                      <code>
                        {diagnostics.public_docs.toLocaleString()} / {diagnostics.public_chunks.toLocaleString()}
                      </code>
                    </div>
                    <div className="diagnostics-kv">
                      <span>Papers docs / vectors</span>
                      <code>
                        {diagnostics.papers_docs.toLocaleString()} / {diagnostics.papers_chunks.toLocaleString()}
                      </code>
                    </div>
                  </details>
                  {diagnostics.seed_sample_docs ? (
                    <p className="sidebar-status">
                      SEED_SAMPLE_DOCS=true: bundled papers may be indexing at startup.
                    </p>
                  ) : null}
                </>
              ) : (
                <p className="sidebar-metric" style={{ color: "var(--text-muted)" }}>
                  {apiHealthy ? "Diagnostics loading…" : "Unavailable until API connects."}
                </p>
              )}
            </div>
            <div className="card card--inset sidebar-indices">
              <strong className="sidebar-card-label">Vector indices</strong>
              {libraries ? (
                <>
                  <div className="sidebar-index-row">
                    <span className="sidebar-index-name">{libraries.public.collection_name}</span>
                    <span className="sidebar-index-stats">
                      {libraries.public.paper_count.toLocaleString()} docs ·{" "}
                      {libraries.public.total_chunks.toLocaleString()} vectors
                    </span>
                  </div>
                  <div className="sidebar-index-row">
                    <span className="sidebar-index-name">{libraries.papers.collection_name}</span>
                    <span className="sidebar-index-stats">
                      {libraries.papers.paper_count.toLocaleString()} docs ·{" "}
                      {libraries.papers.total_chunks.toLocaleString()} vectors
                    </span>
                  </div>
                </>
              ) : (
                <>
                  <p className="sidebar-metric">Docs · {indexedPapers}</p>
                  <p className="sidebar-metric">Chunks · {indexedChunks.toLocaleString()}</p>
                  <p className="sidebar-collection">{health?.collection_stats.collection_name ?? "—"}</p>
                </>
              )}
            </div>
            <p className="corpus-raw-note">
              <strong>Indexed vs raw.</strong> Cards reflect Chroma only. Large <code>.txt</code> trees under{" "}
              <code>data/wiki_txt_build/</code> are not listed here. Sync with{" "}
              <code>scripts/bulk_index_public.py</code> (checkpointed); verify counts with{" "}
              <code>/api/v1/libraries</code>.
            </p>
            <button type="button" className="btn-ghost" onClick={() => void refresh()}>
              Refresh status
            </button>
          </div>
        </aside>

        <section className="content grid">
          <div className="card card--hero">
            <div className="card-hero-head">
              <div>
                <h2 className="card-hero-title">Public corpus retrieval</h2>
                <p className="card-hero-lead">
                  Queries use the <strong>public</strong> collection. Bulk indexing:{" "}
                  <code>scripts/bulk_index_public.py</code> (checkpointed). Counts: <code>/api/v1/libraries</code>.
                  Modes change retrieval budget and prompts; responses are Markdown with sources; optional second
                  retrieval pass when enabled.
                </p>
              </div>
              <span className="kbd-hint" title="Submit from the question field">
                Ctrl+Enter
              </span>
            </div>
            <div className="hero-metrics" aria-label="Public index snapshot">
              {bootstrapping && !libraries ? (
                <p className="hero-metrics-foot">Connecting to API and loading index counts…</p>
              ) : libraries ? (
                <>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{libraries.public.paper_count.toLocaleString()}</div>
                    <div className="hero-metric__label">Articles (public)</div>
                  </div>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{libraries.public.total_chunks.toLocaleString()}</div>
                    <div className="hero-metric__label">Vectors (public)</div>
                  </div>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{modes.length}</div>
                    <div className="hero-metric__label">Retrieval modes</div>
                  </div>
                  <p className="hero-metrics-foot">
                    Collection <strong>{libraries.public.collection_name}</strong> · Papers collection remains on the API
                    for non-console clients
                  </p>
                </>
              ) : (
                <>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{publicArticleCount.toLocaleString()}</div>
                    <div className="hero-metric__label">Articles (public)</div>
                  </div>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{publicChunkCount.toLocaleString()}</div>
                    <div className="hero-metric__label">Vectors (public)</div>
                  </div>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{modes.length}</div>
                    <div className="hero-metric__label">Retrieval modes</div>
                  </div>
                </>
              )}
            </div>
          </div>

          <div
            id="workspace"
            className="card card--workspace"
            aria-busy={loading}
            aria-label="Query workspace"
            tabIndex={-1}
          >
            {loading && <div className="loading-strip" aria-hidden />}

            <div className="card card--inset showcase-section">
              <span className="section-eyebrow">Presets</span>
              <strong>Scenario queries</strong>
              <p className="showcase-section__intro">
                Each card sets prompt, mode, and Top K. All requests use <code>library=public</code>. Run query or
                Submit.
              </p>
              <div className="showcase-grid">
                {SHOWCASE_SCENARIOS.map((s, idx) => (
                  <button
                    key={s.id}
                    type="button"
                    className="showcase-btn"
                    data-scenario={s.id}
                    onClick={() => applyScenario(s)}
                    disabled={loading}
                  >
                    <span className="showcase-btn__idx">Scenario {String(idx + 1).padStart(2, "0")}</span>
                    <span className="showcase-btn__label">{s.label}</span>
                    <span className="showcase-btn__desc">{s.description}</span>
                  </button>
                ))}
              </div>
            </div>

          <div className="grid two workspace-actions">
            <button type="button" className="btn-ghost" onClick={loadDemoPreset} disabled={loading}>
              Reset to baseline
            </button>
            <button type="button" className="btn-cta" onClick={() => void runQuery()} disabled={loading}>
              {loading ? "Running…" : "Run query"}
            </button>
          </div>

          <form className="grid two" onSubmit={askPapers}>
            <div className="form-span-2">
              <label htmlFor="query-input">Question</label>
              <textarea
                id="query-input"
                name="query"
                rows={5}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question grounded in the public index…"
                className="query-textarea"
              />
            </div>
            <div>
              <label htmlFor="query-mode">Mode</label>
              <select id="query-mode" name="query_mode" value={mode} onChange={(e) => setMode(e.target.value)}>
                {modes.map((m) => (
                  <option value={m.value} key={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="top-k">Top K</label>
              <input
                id="top-k"
                name="top_k"
                type="number"
                min={3}
                max={24}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>
            <div className="form-span-2">
              <label className="flare-check">
                <input
                  id="use-flare"
                  name="use_flare"
                  type="checkbox"
                  checked={useFlare}
                  onChange={(e) => setUseFlare(e.target.checked)}
                  disabled={mode === "datasets"}
                />
                <span>
                  FLARE-style active retrieval (extra draft + possible second search). Off for the entity-inventory
                  mode.
                </span>
              </label>
            </div>
            <div className="form-actions-end">
              <button type="submit" className="btn-cta" disabled={loading}>
                {loading ? "Running…" : "Submit"}
              </button>
            </div>
          </form>

          {answer && (
            <div className={`card answer-panel ${!hasAnswer ? "answer-panel--muted" : ""}`} style={{ marginTop: 20 }}>
              <div className="answer-panel__head">
                <h3 className="answer-panel__title" id="synthesis-heading">
                  Synthesis
                </h3>
                <div className="answer-panel__actions">
                  <button type="button" className="btn-ghost btn-compact" onClick={() => void copyAnswer()}>
                    Copy Markdown
                  </button>
                </div>
                <div className="answer-panel__meta">
                  {modelUsed ? (
                    <span className="answer-meta-pill">
                      Model <strong>{modelUsed}</strong>
                    </span>
                  ) : null}
                  {chunksSearched != null ? (
                    <span className="answer-meta-pill">
                      Pool <strong>{chunksSearched}</strong> chunks
                    </span>
                  ) : null}
                  <span className="answer-meta-pill">
                    Mode <strong>{mode}</strong>
                  </span>
                  <span className="answer-meta-pill answer-meta-pill--muted">
                    Citations <strong>{sources.length}</strong>
                  </span>
                  {flareFollowUp ? (
                    <span className="answer-meta-pill" title="Second embedding search merged after forward-looking draft">
                      FLARE <strong>2nd pass</strong>
                    </span>
                  ) : null}
                </div>
              </div>

              {!hasAnswer && (
                <div className="notice notice--warn" style={{ marginTop: 12 }}>
                  No grounded answer from the public index — raise Top K, switch mode, bulk-index more articles, or
                  rephrase.
                </div>
              )}

              <div className="prose-answer" style={{ marginTop: 12 }} aria-labelledby="synthesis-heading">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
                  {answer}
                </ReactMarkdown>
              </div>

              <div className="confidence-row">
                <span id="confidence-label" style={{ fontSize: 13, color: "var(--text-muted)" }}>
                  Confidence
                </span>
                <progress
                  value={Math.min(1, Math.max(0, confidence))}
                  max={1}
                  aria-labelledby="confidence-label"
                  aria-valuenow={Math.round(Math.min(1, Math.max(0, confidence)) * 100)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
                <span style={{ fontSize: 13, fontWeight: 600 }}>{(confidence * 100).toFixed(0)}%</span>
              </div>

              {sources.length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <h4 className="sources-heading">Sources</h4>
                  {sources.map((source, index) => (
                    <details key={`${source.doc_id}-${index}`} className="source-block">
                      <summary>
                        {index + 1}. {source.paper_title}
                        {source.year ? ` (${source.year})` : ""} · {source.section}
                      </summary>
                      <span className="source-meta">
                        Distance {(source.distance ?? 0).toFixed(4)} · chunk {source.chunk_index} · page{" "}
                        {source.page_number}
                      </span>
                      <p style={{ margin: "8px 0 0", color: "var(--text-muted)", fontSize: 13 }}>
                        {source.content_preview}
                      </p>
                    </details>
                  ))}
                </div>
              )}
            </div>
          )}
          </div>

        <div className="card">
          <h2 className="card-h2">Ingest (public)</h2>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 0 }}>
            Small files only. Large corpora: <code>scripts/bulk_index_public.py</code> with a checkpoint.
          </p>
          <label htmlFor="ingest-files" className="visually-hidden">
            Choose PDF, Word, or text files to index
          </label>
          <input
            id="ingest-files"
            type="file"
            multiple
            accept=".pdf,.docx,.txt"
            onChange={(e) => void uploadFiles(e.target.files)}
          />
        </div>

        <div className="card">
          <div className="library-card-header">
            <h2 className="card-h2">Articles in public index</h2>
            <span className="library-count">{publicArticleCount.toLocaleString()} in index</span>
          </div>
          {bootstrapping && papers.length === 0 ? (
            <p style={{ color: "var(--text-muted)" }}>Loading articles from the public index…</p>
          ) : papers.length === 0 ? (
            <p style={{ color: "var(--text-muted)" }}>
              No public vectors yet — run <code>scripts/bulk_index_public.py</code> against your <code>.txt</code>{" "}
              shards, or ingest a small file above. Restart the API with an empty index to auto-seed curated demo
              articles (<code>SEED_PUBLIC_IF_EMPTY=true</code>).
            </p>
          ) : null}
          <div className="library-grid">
            {papers.map((paper) => (
              <div className="card card--inset library-card" key={paper.doc_id}>
                <strong>{paper.title}</strong>
                <p className="card-meta">
                  {(paper.authors || "—") + (paper.year ? ` · ${paper.year}` : "")} · {paper.chunk_count} chunks
                </p>
                {paper.arxiv_id && (
                  <a href={`https://arxiv.org/abs/${paper.arxiv_id}`} target="_blank" rel="noreferrer">
                    arXiv:{paper.arxiv_id}
                  </a>
                )}
                <button type="button" className="btn-ghost" style={{ marginTop: 12 }} onClick={() => void deletePaper(paper.doc_id)}>
                  Remove from index
                </button>
              </div>
            ))}
          </div>
        </div>

        {notice ? (
          <div className={noticeClass} {...noticeA11y}>
            {notice}
          </div>
        ) : null}
        </section>
      </main>
    </div>
  );
}
