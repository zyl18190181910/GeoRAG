from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from .models import DocumentChunk, SearchHit
from .retrieval import SparseTfidfRetriever


class TextEmbedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        cache_path: Path,
        *,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._active_device = "cpu"

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self.cache_path.mkdir(parents=True, exist_ok=True)
            requested = self.device.strip().lower()
            if requested not in {"auto", "cpu", "cuda"}:
                raise ValueError("embedding device must be auto, cpu, or cuda")
            device = (
                "cuda"
                if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
                else "cpu"
            )
            options = {
                "cache_folder": str(self.cache_path),
                "device": device,
            }
            if device == "cuda":
                # BGE-M3 is substantially larger than the previous encoder. FP16 keeps
                # it usable on common 4 GB laptop GPUs without changing vector shape.
                options["model_kwargs"] = {"torch_dtype": torch.float16}
            try:
                # Avoid a slow network metadata check after the model is cached.
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                    **options,
                )
            except OSError:
                # On the first run, allow Sentence Transformers to download it.
                self._model = SentenceTransformer(self.model_name, **options)
            self._active_device = device
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        options = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        try:
            vectors = self._load().encode(texts, **options)
        except RuntimeError as exc:
            if self._active_device != "cuda" or "out of memory" not in str(exc).lower():
                raise
            # A deterministic CPU fallback is preferable to disabling semantic search.
            import torch
            from sentence_transformers import SentenceTransformer

            self._model = None
            torch.cuda.empty_cache()
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_path),
                device="cpu",
                local_files_only=True,
            )
            self._active_device = "cpu"
            vectors = self._model.encode(texts, **options)
        return vectors.tolist()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode a stable document collection with an on-disk vector cache."""
        if not texts:
            return []
        digest = hashlib.sha256()
        digest.update(self.model_name.encode("utf-8"))
        for text in texts:
            digest.update(b"\0")
            digest.update(text.encode("utf-8"))
        cache_dir = self.cache_path / "vector_cache"
        cache_file = cache_dir / f"{digest.hexdigest()}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, list) and len(cached) == len(texts):
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        vectors = self.encode(texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(vectors, separators=(",", ":")), encoding="utf-8")
        temporary.replace(cache_file)
        return vectors


class SemanticRetriever:
    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    def fit(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        texts = [chunk.text for chunk in chunks]
        encode_documents = getattr(self.embedder, "encode_documents", None)
        self._vectors = (
            encode_documents(texts) if callable(encode_documents) else self.embedder.encode(texts)
        )

    def search(self, query: str, *, top_k: int = 4, min_score: float = -1.0) -> list[SearchHit]:
        if not self._chunks or not query.strip():
            return []
        query_vector = self.embedder.encode([query])[0]
        hits = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            if score >= min_score:
                hits.append(SearchHit(chunk=chunk, score=score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


class HybridRetriever:
    def __init__(
        self,
        *,
        sparse_min_score: float,
        semantic_min_score: float,
        semantic: SemanticRetriever | None = None,
    ) -> None:
        self.sparse = SparseTfidfRetriever()
        self.semantic = semantic
        self.sparse_min_score = sparse_min_score
        self.semantic_min_score = semantic_min_score
        self.semantic_error: str | None = None
        self._chunks: list[DocumentChunk] = []

    @property
    def semantic_ready(self) -> bool:
        return self.semantic is not None and self.semantic_error is None

    def fit(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self.sparse.fit(chunks)
        if self.semantic is not None:
            try:
                self.semantic.fit(chunks)
                self.semantic_error = None
            except Exception as exc:  # The sparse retriever remains a safe fallback.
                self.semantic_error = f"{type(exc).__name__}: {exc}"

    def search(self, query: str, *, top_k: int = 4) -> list[SearchHit]:
        if not self._chunks:
            return []
        candidate_count = max(top_k * 4, len(self._chunks))
        sparse_hits = self.sparse.search(query, top_k=candidate_count, min_score=0.0)
        if not self.semantic_ready:
            return [hit for hit in sparse_hits if hit.score >= self.sparse_min_score][:top_k]

        assert self.semantic is not None
        semantic_hits = self.semantic.search(query, top_k=candidate_count, min_score=-1.0)
        sparse_scores = {hit.chunk.chunk_id: hit.score for hit in sparse_hits}
        semantic_scores = {hit.chunk.chunk_id: hit.score for hit in semantic_hits}
        chunks = {chunk.chunk_id: chunk for chunk in self._chunks}

        combined: list[SearchHit] = []
        for chunk_id, chunk in chunks.items():
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            semantic_score = semantic_scores.get(chunk_id, -1.0)
            if sparse_score < self.sparse_min_score and semantic_score < self.semantic_min_score:
                continue
            score = 0.4 * sparse_score + 0.6 * max(semantic_score, 0.0)
            combined.append(SearchHit(chunk=chunk, score=score))
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:top_k]
