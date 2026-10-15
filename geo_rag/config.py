from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    storage_path: Path
    min_score: float = 0.08
    semantic_min_score: float = 0.42
    max_file_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 480
    chunk_overlap: int = 80
    semantic_enabled: bool = False
    reranker_enabled: bool = False
    generation_enabled: bool = False
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    reranker_model: str = "BAAI/bge-reranker-base"
    model_cache_path: Path = Path("data/models")
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_reasoning_effort: str = "high"

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API")
        ai_default = bool(api_key)
        return cls(
            storage_path=Path(os.getenv("GEO_RAG_STORAGE", "data/storage/knowledge.json")),
            min_score=float(os.getenv("GEO_RAG_MIN_SCORE", "0.08")),
            semantic_min_score=float(os.getenv("GEO_RAG_SEMANTIC_MIN_SCORE", "0.42")),
            max_file_bytes=int(os.getenv("GEO_RAG_MAX_FILE_BYTES", str(10 * 1024 * 1024))),
            semantic_enabled=cls._env_flag("GEO_RAG_SEMANTIC_ENABLED", ai_default),
            reranker_enabled=cls._env_flag("GEO_RAG_RERANKER_ENABLED", ai_default),
            generation_enabled=cls._env_flag("GEO_RAG_GENERATION_ENABLED", ai_default),
            embedding_model=os.getenv("GEO_RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_device=os.getenv("GEO_RAG_EMBEDDING_DEVICE", "auto"),
            embedding_batch_size=int(os.getenv("GEO_RAG_EMBEDDING_BATCH_SIZE", "8")),
            reranker_model=os.getenv("GEO_RAG_RERANKER_MODEL", "BAAI/bge-reranker-base"),
            model_cache_path=Path(os.getenv("GEO_RAG_MODEL_CACHE", "data/models")),
            deepseek_api_key=api_key,
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
        )

