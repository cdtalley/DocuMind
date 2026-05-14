import time
from typing import Any

import requests


class OllamaConnectionError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        llm_model: str,
        embedding_model: str,
        *,
        request_timeout_sec: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._timeout = float(request_timeout_sec)

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.1) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        last_err: BaseException | None = None
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()
                msg = data.get("message") or {}
                content = msg.get("content") if isinstance(msg, dict) else None
                if not content:
                    raise ValueError(f"Unexpected Ollama response shape: {repr(data)[:500]}")
                return str(content)
            except Exception as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise OllamaConnectionError(
                        f"Ollama LLM error at {self.base_url} after retries: {last_err!s}"
                    ) from last_err
        raise AssertionError("unreachable")

    def embed(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.embedding_model, "prompt": text}
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=self._timeout)
                response.raise_for_status()
                return response.json()["embedding"]
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise OllamaConnectionError(
                        f"Ollama LLM not reachable at {self.base_url}. Run: ollama serve"
                    ) from exc
        raise OllamaConnectionError(
            f"Ollama LLM not reachable at {self.base_url}. Run: ollama serve"
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def health_check(self) -> dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=min(30.0, self._timeout))
            response.raise_for_status()
            data = response.json()
            models = [item.get("name", "") for item in data.get("models", []) if item.get("name")]
            return {"available": True, "models": models}
        except Exception:
            return {"available": False, "models": []}
