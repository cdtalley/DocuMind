"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8001";

export type ShowcaseScenario = {
  id: string;
  label: string;
  description: string;
  query: string;
  mode: string;
  section: string;
  topK: number;
  /** When set, applies the FLARE follow-up retrieval toggle for this scenario. */
  useFlare?: boolean;
};

const SHOWCASE_SCENARIOS: ShowcaseScenario[] = [
  {
    id: "gold",
    label: "Gold: demo & QA",
    description:
      "Tuned for the bundled v7 library: dense benchmark keywords + compare outline + max Top K + FLARE. Use for screenshots and stakeholder demos.",
    query: `DEMO — Cross-paper benchmark audit (grounded synthesis only).

Retrieval anchors: GLUE SuperGLUE SQuAD ImageNet CIFAR MNIST Cora Citeseer PubMed LibriSpeech C4 Open Images IEEE-CIS MovieLens transformers vision CNN tabular gradient boosting graph convolution time series drift calibration ECE AdamW LoRA retrieval augmentation.

Using ONLY the retrieved excerpts, produce the full compare-mode outline: ## At a glance; ## Narrative overview; ## Comparison table as a GitHub-flavored markdown table with columns: Method / paradigm | Paper (exact title from excerpt) | Datasets or benchmarks named in text | Reported claim or metric if stated | Limitation or scope | Why a practitioner would care; ## Mechanism & objective contrast (subsections for losses/objectives and for data & evaluation); ## Trade-offs & decision guide.

Rules: No invented paper titles or datasets. If a name appears in a Keywords: line in the excerpt, you may treat it as in-scope. Where the library has no chunk for a theme, write explicitly that the excerpt set does not cover it — do not speculate.`,
    mode: "compare",
    section: "All Sections",
    topK: 24,
    useFlare: true
  },
  {
    id: "flagship",
    label: "Flagship: research landscape",
    description:
      "Cross-paper brief: method families, datasets/benchmarks, metrics — grounded in your library.",
    query: `Brief a technical lead using ONLY the retrieved context from my library:
(1) Cluster findings by area: NLP/Transformers, vision, tabular boosting, graph representation learning, probabilistic time series, and RLHF/alignment where present.
(2) For each area, list named datasets or benchmarks (e.g., GLUE, ImageNet, LibriSpeech, Cora, IEEE-CIS) and what they measure.
(3) Cite exact paper titles for each bullet.
If an area has no supporting chunk, state that clearly.`,
    mode: "compare",
    section: "All Sections",
    topK: 18
  },
  {
    id: "datasets",
    label: "Dataset map",
    description: "Structured inventory of datasets and where they appear.",
    query:
      "Produce a cross-paper dataset inventory: every named benchmark or dataset in the context, which paper mentions it, and one-line usage from the passage.",
    mode: "datasets",
    section: "All Sections",
    topK: 18
  },
  {
    id: "reproduce",
    label: "Repro blueprint",
    description: "What you would need to re-run experiments (datasets, architecture hints, metrics).",
    query:
      "Across the papers in context, what concrete artifacts would I need to reproduce reported results: datasets, model families, optimizers/schedules mentioned, and evaluation metrics?",
    mode: "reproduce",
    section: "All Sections",
    topK: 14
  },
  {
    id: "methodology",
    label: "Methods & training",
    description: "Implementation-focused extraction (optimizers, objectives, architectures).",
    query:
      "Summarize training objectives and model architectures described in context: contrastive losses, CTC, Transformers, graph convolutions, gradient boosting — tie each to paper title.",
    mode: "methodology",
    section: "All Sections",
    topK: 14
  }
];

const DEFAULT_SCENARIO = SHOWCASE_SCENARIOS[0];

const INITIAL_PUBLIC_QUERY =
  "What do the retrieved passages state? Summarize using only that evidence and cite **Article title** from the context.";

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

const PRESET_LIBRARY: PaperCard[] = [
  {
    doc_id: "__catalog__",
    filename: "Public index",
    title: "Public library (Wikipedia-scale) — index with scripts/bulk_index_public.py or ingest here",
    authors: "Primary corpus",
    year: "",
    arxiv_id: "",
    chunk_count: 0
  }
];

const modes = [
  { label: "General Q&A", value: "general" },
  { label: "Compare Methods", value: "compare" },
  { label: "Methodology Deep Dive", value: "methodology" },
  { label: "Dataset Finder", value: "datasets" },
  { label: "Reproduce Results", value: "reproduce" }
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
  const [section, setSection] = useState(DEFAULT_SCENARIO.section);
  const [topK, setTopK] = useState(DEFAULT_SCENARIO.topK);
  const [useFlare, setUseFlare] = useState(false);
  const [corpusLibrary, setCorpusLibrary] = useState<"public" | "papers">("public");
  const [flareFollowUp, setFlareFollowUp] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [confidence, setConfidence] = useState(0);
  const [hasAnswer, setHasAnswer] = useState(true);
  const [chunksSearched, setChunksSearched] = useState<number | null>(null);
  const [arxivId, setArxivId] = useState("");
  const [notice, setNotice] = useState("");
  const [noticeTone, setNoticeTone] = useState<NoticeTone>("info");
  const [apiHealthy, setApiHealthy] = useState(true);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [modelUsed, setModelUsed] = useState<string | null>(null);
  const [libraries, setLibraries] = useState<LibrariesPayload | null>(null);

  const libraryStats = useMemo(
    () => ({
      totalPapers: papers.length,
      totalChunks: papers.reduce((acc, paper) => acc + paper.chunk_count, 0)
    }),
    [papers]
  );
  const displayPapers = papers.length > 0 ? papers : PRESET_LIBRARY;

  const fetchJson = async <T,>(path: string, options?: RequestInit): Promise<T> => {
    const response = await fetch(`${API_BASE_URL}${path}`, options);
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

  const refresh = async () => {
    try {
      const [healthPayload, papersPayload, librariesPayload] = await Promise.all([
        fetchJson<HealthPayload>("/health"),
        fetchJson<PaperCard[]>(`/api/v1/papers?library=${encodeURIComponent(corpusLibrary)}`),
        fetchJson<LibrariesPayload>("/api/v1/libraries")
      ]);
      setHealth(healthPayload);
      setPapers(papersPayload);
      setLibraries(librariesPayload);
      setApiHealthy(true);
      setLastSync(new Date());
      setNotice("");
    } catch {
      setApiHealthy(false);
      setHealth(null);
      setPapers([]);
      setLibraries(null);
      setNotice(
        `API unreachable at ${API_BASE_URL}. Run: .\\start_documind.ps1 from the project root (or uvicorn on port 8001).`
      );
      setNoticeTone("error");
    }
  };

  useEffect(() => {
    void refresh();
  }, [corpusLibrary]);

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
          library: corpusLibrary,
          top_k: topK,
          query_mode: mode,
          section_filter: section === "All Sections" ? null : section,
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
  }, [query, topK, mode, section, useFlare, corpusLibrary]);

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
    setSection(s.section);
    setTopK(s.topK);
    setCorpusLibrary("papers");
    if (typeof s.useFlare === "boolean") setUseFlare(s.useFlare);
    setNotice(`Loaded scenario: ${s.label} (papers library)`);
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
        form.append("library", corpusLibrary);
        await fetchJson(`/api/v1/ingest`, { method: "POST", body: form });
      }
      setNotice("Upload complete — library refreshed.");
      setNoticeTone("success");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Upload failed");
      setNoticeTone("error");
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
      setNotice(`Fetched and indexed: ${data.title}`);
      setNoticeTone("success");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "ArXiv fetch failed");
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
      await fetchJson(`/api/v1/papers/${docId}?library=${encodeURIComponent(corpusLibrary)}`, { method: "DELETE" });
      setNotice("Paper removed.");
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
  const apiDocsUrl = `${API_BASE_URL.replace(/\/$/, "")}/docs`;
  const syncLabel = lastSync
    ? lastSync.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "—";

  const indexedPapers = health?.collection_stats.paper_count ?? libraryStats.totalPapers;
  const indexedChunks = health?.collection_stats.total_chunks ?? libraryStats.totalChunks;
  const totalDualChunks =
    libraries != null ? libraries.public.total_chunks + libraries.papers.total_chunks : indexedChunks;

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
              Dual-library RAG (public + papers) · Chroma · Ollama · FastAPI
              {libraries ? (
                <span className="enterprise-topbar__default-lib">
                  {" "}
                  · default query library: <strong>{libraries.default_library}</strong>
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
          <span className="enterprise-topbar__sync">Synced {syncLabel}</span>
        </div>
        <div className="enterprise-topbar__links">
          <a className="topbar-link" href={apiDocsUrl} target="_blank" rel="noreferrer">
            OpenAPI
          </a>
          <code className="topbar-code">{API_BASE_URL}</code>
        </div>
      </header>

      <div className="demo-trust-bar" role="list" aria-label="Capabilities">
        <span className="demo-trust-bar__item" role="listitem">
          Answers cite retrieved chunks
        </span>
        <span className="demo-trust-bar__item" role="listitem">
          Runs on localhost by default
        </span>
        <span className="demo-trust-bar__item" role="listitem">
          Five query modes (incl. compare / datasets)
        </span>
        <span className="demo-trust-bar__item" role="listitem">
          Ingest API + optional arXiv pull
        </span>
        <span className="demo-trust-bar__item" role="listitem">
          Live dual-index snapshot via /api/v1/libraries
        </span>
      </div>

      <main className="layout">
        <aside className="sidebar">
          <h1 className="sidebar-title">Control plane</h1>
          <p className="sidebar-tagline">
            Same data the UI uses: <code>/health</code> and <code>/api/v1/papers</code>. Refresh after ingest or
            re-index.
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
            <button type="button" className="btn-ghost" onClick={() => void refresh()}>
              Refresh status
            </button>
          </div>
        </aside>

        <section className="content grid">
          <div className="card card--hero">
            <div className="card-hero-head">
              <div>
                <h2 className="card-hero-title">Cross-document retrieval</h2>
                <p className="card-hero-lead">
                  Modes alter retrieval budget, section scope, and prompt shape. Response: Markdown synthesis,
                  confidence score, source cards (chunk text, cosine distance). Optional second retrieval pass (FLARE-style).
                  Load a scenario, run query, expand Sources.
                </p>
              </div>
              <span className="kbd-hint" title="Submit from the question field">
                Ctrl+Enter
              </span>
            </div>
            <div className="hero-metrics" aria-label="Index snapshot">
              {libraries ? (
                <>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{libraries.public.total_chunks.toLocaleString()}</div>
                    <div className="hero-metric__label">Public index chunks</div>
                  </div>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{libraries.papers.total_chunks.toLocaleString()}</div>
                    <div className="hero-metric__label">Papers index chunks</div>
                  </div>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{modes.length}</div>
                    <div className="hero-metric__label">Retrieval modes</div>
                  </div>
                  <p className="hero-metrics-foot">
                    <strong>{totalDualChunks.toLocaleString()}</strong> total vectors across both Chroma collections
                  </p>
                </>
              ) : (
                <>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{indexedPapers}</div>
                    <div className="hero-metric__label">Indexed documents</div>
                  </div>
                  <div className="hero-metric">
                    <div className="hero-metric__value">{indexedChunks.toLocaleString()}</div>
                    <div className="hero-metric__label">Vector chunks</div>
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
              <span className="section-eyebrow">Sample prompts</span>
              <strong>Showcase scenarios</strong>
              <p className="showcase-section__intro">
                Long prompts tuned for a mixed-topic corpus. A card loads query text, mode, section filter, and Top K;
                then use Run query or Submit.
              </p>
              <div className="showcase-grid">
                {SHOWCASE_SCENARIOS.map((s, idx) => (
                  <button
                    key={s.id}
                    type="button"
                    className="showcase-btn"
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
              Reset to flagship
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
                placeholder={INITIAL_PUBLIC_QUERY.slice(0, 120) + "…"}
                className="query-textarea"
              />
            </div>
            <div>
              <label htmlFor="corpus-library">Library</label>
              <select
                id="corpus-library"
                name="library"
                value={corpusLibrary}
                onChange={(e) => setCorpusLibrary(e.target.value as "public" | "papers")}
              >
                <option value="public">Public (Wikipedia-scale index)</option>
                <option value="papers">Papers (PDFs / arXiv / legacy bundle)</option>
              </select>
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
              <label htmlFor="section-filter">Section</label>
              <select
                id="section-filter"
                name="section_filter"
                value={section}
                onChange={(e) => setSection(e.target.value)}
              >
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
                  FLARE-style active retrieval (extra draft + possible second search). Off for Dataset Finder mode.
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
                  No grounded answer from the current index — try uploading more papers, raising Top K, or rephrasing.
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

        <div className="grid two">
          <div className="card">
            <h2 className="card-h2">Upload papers</h2>
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
            <h2 className="card-h2">Fetch from arXiv</h2>
            <label htmlFor="arxiv-id-input">arXiv ID</label>
            <input
              id="arxiv-id-input"
              name="arxiv_id"
              value={arxivId}
              onChange={(e) => setArxivId(e.target.value)}
              placeholder="1706.03762"
            />
            <button type="button" className="ingest-actions btn-cta" onClick={() => void fetchArxiv()}>
              Fetch PDF
            </button>
          </div>
        </div>

        <div className="card">
          <div className="library-card-header">
            <h2 className="card-h2">{corpusLibrary === "public" ? "Public corpus" : "Paper library"}</h2>
            <span className="library-count">{indexedPapers} in index</span>
          </div>
          {papers.length === 0 ? (
            <p style={{ color: "var(--text-muted)" }}>
              No vectors yet — start the API with Ollama so bundled samples index, or upload / fetch above.
            </p>
          ) : null}
          <div className="library-grid">
            {displayPapers.map((paper) => (
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
                {!paper.doc_id.startsWith("preset_") && paper.doc_id !== "__catalog__" && (
                  <button type="button" className="btn-ghost" style={{ marginTop: 12 }} onClick={() => void deletePaper(paper.doc_id)}>
                    Remove from library
                  </button>
                )}
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
