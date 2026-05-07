import time
from typing import Any

import requests


class OllamaConnectionError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str, llm_model: str, embedding_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.1) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise OllamaConnectionError(
                        f"Ollama LLM not reachable at {self.base_url}. Run: ollama serve"
                    )
        raise OllamaConnectionError(
            f"Ollama LLM not reachable at {self.base_url}. Run: ollama serve"
        )

    def embed(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.embedding_model, "prompt": text}
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=120)
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
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            models = [item.get("name", "") for item in data.get("models", []) if item.get("name")]
            return {"available": True, "models": models}
        except Exception:
            return {"available": False, "models": []}
