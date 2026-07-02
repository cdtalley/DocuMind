/**
 * Same-origin API path for SSR + client (Next.js rewrites → FastAPI).
 * Must not branch on `typeof window` — that causes hydration mismatches.
 */
export const API_BASE_PATH = "/documind-api";

export function getApiBaseUrl(): string {
  return API_BASE_PATH;
}
