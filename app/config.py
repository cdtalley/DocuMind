from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documind_papers"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TOP_K_RESULTS: int = 6
    RELEVANCE_THRESHOLD: float = 0.45
    MAX_FILE_SIZE_MB: int = 50
    ARXIV_BASE_URL: str = "https://export.arxiv.org/pdf"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
