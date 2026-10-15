from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from .models import SearchHit


class PairReranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_path: Path) -> None:
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            self.cache_path.mkdir(parents=True, exist_ok=True)
            options = {
                "cache_folder": str(self.cache_path),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    local_files_only=True,
                    **options,
                )
            except (OSError, AttributeError, ValueError):
                # Transformers can raise AttributeError/ValueError when a
                # previous interrupted download left an incomplete snapshot.
                self._model = CrossEncoder(self.model_name, **options)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not hits or not query.strip():
            return []
        scores = self._load().predict(
            [(query, hit.chunk.text) for hit in hits],
            batch_size=min(16, len(hits)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        reranked = []
        for hit, raw_score in zip(hits, scores, strict=True):
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(raw_score)))))
            score = 0.25 * hit.score + 0.75 * probability
            reranked.append(SearchHit(chunk=hit.chunk, score=score))
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
