"""Precompute and persist embeddings for the bundled public-paper corpus."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_rag.ablation import _load_public_paper_chunks  # noqa: E402
from geo_rag.config import Settings  # noqa: E402
from geo_rag.semantic import SentenceTransformerEmbedder  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    chunks = _load_public_paper_chunks(ROOT, settings)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.model_cache_path,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    vectors = embedder.encode_documents([chunk.text for chunk in chunks])
    dimension = len(vectors[0]) if vectors else 0
    print(
        f"ready model={settings.embedding_model} device={embedder._active_device} "
        f"documents={len(vectors)} dimension={dimension}"
    )


if __name__ == "__main__":
    main()
