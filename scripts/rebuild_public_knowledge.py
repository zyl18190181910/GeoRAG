"""Atomically rebuild the local knowledge store from the public-paper corpus."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_rag.ablation import _load_public_paper_chunks  # noqa: E402
from geo_rag.config import Settings  # noqa: E402
from geo_rag.storage import JsonKnowledgeStore  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    chunks = _load_public_paper_chunks(ROOT, settings)
    updated_at = datetime.now(UTC).isoformat()
    states: dict[str, dict] = {}
    for chunk in chunks:
        states.setdefault(
            chunk.source_id,
            {
                "enabled": True,
                "filename": chunk.metadata.get("filename"),
                "updated_at": updated_at,
            },
        )
    JsonKnowledgeStore(settings.storage_path).save(chunks, states)
    print(
        f"rebuilt sources={len(states)} chunks={len(chunks)} "
        f"storage={settings.storage_path}"
    )


if __name__ == "__main__":
    main()
