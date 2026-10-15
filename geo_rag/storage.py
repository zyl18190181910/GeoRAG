from __future__ import annotations

import json
from pathlib import Path

from .models import DocumentChunk


class JsonKnowledgeStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load_bundle(self) -> tuple[list[DocumentChunk], dict[str, dict]]:
        if not self.path.exists():
            return [], {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") not in {1, 2}:
            raise ValueError("unsupported knowledge-store schema")
        chunks = [DocumentChunk.from_dict(value) for value in payload.get("chunks", [])]
        states = {str(key): dict(value) for key, value in payload.get("sources", {}).items()}
        for chunk in chunks:
            states.setdefault(
                chunk.source_id,
                {
                    "enabled": True,
                    "filename": chunk.metadata.get("filename"),
                    "updated_at": None,
                },
            )
        return chunks, states

    def load(self) -> list[DocumentChunk]:
        return self.load_bundle()[0]

    def save(self, chunks: list[DocumentChunk], source_states: dict[str, dict] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "sources": source_states or {},
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

