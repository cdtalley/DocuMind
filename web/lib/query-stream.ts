import { getApiBaseUrl } from "./api-base";

export type QueryStreamSource = {
  doc_id: string;
  paper_title: string;
  authors: string;
  year: string;
  section: string;
  page_number: number;
  chunk_index: number;
  content_preview: string;
  distance: number;
};

export type QueryStreamRetrieval = {
  sources: QueryStreamSource[];
  confidence: number;
  has_answer: boolean;
  query: string;
  query_mode: string;
  model_used: string;
  chunks_searched: number;
  flare_enabled: boolean;
  flare_followup_retrieval: boolean;
  retrieval_strategy: string;
  retrieval_passes: number;
  library: string;
};

export type QueryStreamDone = QueryStreamRetrieval & {
  answer: string;
};

type StreamHandlers = {
  onRetrieval: (payload: QueryStreamRetrieval) => void;
  onToken: (text: string) => void;
  onDone: (payload: QueryStreamDone) => void;
  onError: (message: string) => void;
};

function parseSseBlock(block: string): { event: string; data: string } | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  return { event, data: dataLines.join("\n") };
}

export async function streamQuery(
  body: Record<string, unknown>,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload?.detail) detail = String(payload.detail);
    } catch {
      // ignore
    }
    handlers.onError(detail);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    handlers.onError("Streaming not supported in this browser.");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const parsed = parseSseBlock(part.trim());
      if (!parsed) continue;
      try {
        const payload = JSON.parse(parsed.data) as Record<string, unknown>;
        if (parsed.event === "retrieval") {
          handlers.onRetrieval(payload as QueryStreamRetrieval);
        } else if (parsed.event === "token") {
          handlers.onToken(String(payload.text ?? ""));
        } else if (parsed.event === "done") {
          handlers.onDone(payload as QueryStreamDone);
        } else if (parsed.event === "error") {
          handlers.onError(String(payload.detail ?? "Stream failed"));
        }
      } catch {
        handlers.onError("Malformed stream payload from API.");
      }
    }
  }
}
