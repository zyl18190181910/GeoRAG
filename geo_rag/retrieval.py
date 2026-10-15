from __future__ import annotations

import math
import re
from collections import Counter

from .models import DocumentChunk, SearchHit


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_.+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")
_STOP_TOKENS = {
    "这个",
    "那个",
    "什么",
    "哪些",
    "多少",
    "如何",
    "怎么",
    "怎样",
    "为何",
    "为什么",
    "是否",
    "应该",
    "应当",
    "可以",
    "需要",
    "问题",
    "进行",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        value = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            chars = list(value)
            if len(chars) == 1:
                tokens.append(chars[0])
            else:
                tokens.extend(
                    token
                    for i in range(len(chars) - 1)
                    if (token := chars[i] + chars[i + 1]) not in _STOP_TOKENS
                )
        else:
            tokens.append(value)
    return tokens


class SparseTfidfRetriever:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._norms: list[float] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def fit(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        if not chunks:
            self._idf = {}
            self._vectors = []
            self._norms = []
            return

        token_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for counts in token_counts:
            document_frequency.update(counts.keys())

        count = len(chunks)
        self._idf = {
            token: math.log((1 + count) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self._vectors = [self._tfidf(counts) for counts in token_counts]
        self._norms = [self._norm(vector) for vector in self._vectors]

    def _tfidf(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {
            token: (frequency / total) * self._idf[token]
            for token, frequency in counts.items()
            if token in self._idf
        }

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def search(self, query: str, *, top_k: int = 4, min_score: float = 0.0) -> list[SearchHit]:
        if not self._chunks or not query.strip():
            return []
        query_vector = self._tfidf(Counter(tokenize(query)))
        query_norm = self._norm(query_vector)
        if query_norm == 0:
            return []
        anchor_tokens = {
            token for token in query_vector if len(token) >= 2 and re.search(r"[a-z0-9]", token)
        }

        hits: list[SearchHit] = []
        for chunk, vector, norm in zip(self._chunks, self._vectors, self._norms, strict=True):
            if norm == 0:
                continue
            dot = sum(value * vector.get(token, 0.0) for token, value in query_vector.items())
            score = dot / (query_norm * norm)
            if anchor_tokens:
                anchor_coverage = sum(token in vector for token in anchor_tokens) / len(anchor_tokens)
                score = min(1.0, score + 0.25 * anchor_coverage)
            if score >= min_score:
                hits.append(SearchHit(chunk=chunk, score=score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

