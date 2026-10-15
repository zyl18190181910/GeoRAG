"""Download the configured embedding model and run a small local smoke test."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_rag.config import Settings  # noqa: E402
from geo_rag.semantic import SentenceTransformerEmbedder  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.model_cache_path,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    vectors = embedder.encode(["中文地球物理反演", "multilingual retrieval"])
    if len(vectors) != 2 or not vectors[0]:
        raise RuntimeError("embedding smoke test returned invalid vectors")
    print(
        f"ready model={settings.embedding_model} "
        f"device={embedder._active_device} dimension={len(vectors[0])}"
    )


if __name__ == "__main__":
    main()
