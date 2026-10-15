from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .chunking import chunk_text, split_sentences
from .config import Settings
from .generation import AnswerGenerator, DeepSeekGenerator, QueryPlan, heuristic_plan
from .models import (
    AnswerMode,
    AskResponse,
    Citation,
    ConversationTurn,
    DocumentChunk,
    SearchHit,
    SourceInfo,
)
from .reranking import CrossEncoderReranker, PairReranker
from .retrieval import tokenize
from .semantic import (
    HybridRetriever,
    SemanticRetriever,
    SentenceTransformerEmbedder,
    TextEmbedder,
)
from .storage import JsonKnowledgeStore


REFUSAL_MESSAGE = "现有知识库中没有足够依据回答这个问题，请补充相关资料后重试。"


class RagService:
    def __init__(
        self,
        settings: Settings,
        *,
        semantic_embedder: TextEmbedder | None = None,
        answer_generator: AnswerGenerator | None = None,
        reranker: PairReranker | None = None,
    ) -> None:
        self.settings = settings
        self.store = JsonKnowledgeStore(settings.storage_path)
        semantic = None
        if settings.semantic_enabled:
            embedder = semantic_embedder or SentenceTransformerEmbedder(
                settings.embedding_model,
                settings.model_cache_path,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
            )
            semantic = SemanticRetriever(embedder)
        self.retriever = HybridRetriever(
            sparse_min_score=settings.min_score,
            semantic_min_score=settings.semantic_min_score,
            semantic=semantic,
        )
        if reranker is not None:
            self.reranker = reranker
        elif settings.reranker_enabled:
            self.reranker = CrossEncoderReranker(
                settings.reranker_model,
                settings.model_cache_path,
            )
        else:
            self.reranker = None
        if answer_generator is not None:
            self.generator = answer_generator
        elif settings.generation_enabled and settings.deepseek_api_key:
            self.generator = DeepSeekGenerator(
                api_key=settings.deepseek_api_key,
                model=settings.deepseek_model,
                base_url=settings.deepseek_base_url,
                reasoning_effort=settings.deepseek_reasoning_effort,
            )
        else:
            self.generator = None
        self.last_planning_error: str | None = None
        self.last_reranker_error: str | None = None
        self.last_generation_error: str | None = None
        self._lock = threading.RLock()
        self._chunks, self._source_states = self.store.load_bundle()
        self._rebuild_index()

    @property
    def chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    @property
    def source_ids(self) -> list[str]:
        return sorted({chunk.source_id for chunk in self._chunks})

    @property
    def active_chunks(self) -> list[DocumentChunk]:
        return [
            chunk
            for chunk in self._chunks
            if self._source_states.get(chunk.source_id, {}).get("enabled", True)
        ]

    @property
    def active_source_ids(self) -> list[str]:
        return sorted({chunk.source_id for chunk in self.active_chunks})

    @property
    def semantic_enabled(self) -> bool:
        return self.retriever.semantic_ready

    @property
    def reranker_enabled(self) -> bool:
        return self.reranker is not None and self.last_reranker_error is None

    @property
    def generation_enabled(self) -> bool:
        return self.generator is not None

    def ingest(self, source_id: str, text: str, metadata: dict | None = None) -> int:
        return self.ingest_sections(source_id, [(text, {})], metadata=metadata)

    def ingest_sections(
        self,
        source_id: str,
        sections: Iterable[tuple[str, dict]],
        *,
        metadata: dict | None = None,
    ) -> int:
        common_metadata = dict(metadata or {})
        new_chunks: list[DocumentChunk] = []
        for text, section_metadata in sections:
            combined_metadata = {**common_metadata, **dict(section_metadata)}
            local_chunks = chunk_text(
                source_id,
                text,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
                metadata=combined_metadata,
            )
            for local_chunk in local_chunks:
                order = len(new_chunks)
                new_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{source_id}:{order}",
                        source_id=source_id,
                        text=local_chunk.text,
                        order=order,
                        metadata=local_chunk.metadata,
                    )
                )
        if not new_chunks:
            raise ValueError("document is empty after normalization")
        with self._lock:
            retained = [chunk for chunk in self._chunks if chunk.source_id != source_id]
            self._chunks = retained + new_chunks
            self._source_states[source_id] = {
                "enabled": True,
                "filename": common_metadata.get("filename"),
                "updated_at": datetime.now(UTC).isoformat(),
            }
            self._save_and_rebuild()
        return len(new_chunks)

    def source_details(self) -> list[SourceInfo]:
        counts: dict[str, int] = {}
        for chunk in self._chunks:
            counts[chunk.source_id] = counts.get(chunk.source_id, 0) + 1
        return [
            SourceInfo(
                source_id=source_id,
                chunks=counts[source_id],
                enabled=bool(self._source_states.get(source_id, {}).get("enabled", True)),
                filename=self._source_states.get(source_id, {}).get("filename"),
                updated_at=self._source_states.get(source_id, {}).get("updated_at"),
            )
            for source_id in sorted(counts)
        ]

    def set_source_enabled(self, source_id: str, enabled: bool) -> int:
        if source_id not in self.source_ids:
            raise KeyError(source_id)
        with self._lock:
            state = self._source_states.setdefault(source_id, {})
            state["enabled"] = enabled
            self._save_and_rebuild()
        return sum(chunk.source_id == source_id for chunk in self._chunks)

    def delete_source(self, source_id: str) -> int:
        removed = sum(chunk.source_id == source_id for chunk in self._chunks)
        if not removed:
            raise KeyError(source_id)
        with self._lock:
            self._chunks = [chunk for chunk in self._chunks if chunk.source_id != source_id]
            self._source_states.pop(source_id, None)
            self._save_and_rebuild()
        return removed

    def reindex(self) -> int:
        with self._lock:
            self._rebuild_index()
        return len(self.active_chunks)

    def _save_and_rebuild(self) -> None:
        self.store.save(self._chunks, self._source_states)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.retriever.fit(self.active_chunks)

    def ask(
        self,
        question: str,
        top_k: int = 6,
        *,
        answer_mode: AnswerMode = "auto",
        history: list[ConversationTurn] | None = None,
    ) -> AskResponse:
        started = time.perf_counter()
        turns = list(history or [])[-6:]
        plan = self._plan(question, answer_mode, turns)
        base_k = min(10, max(top_k, 6 if plan.answer_mode in {"summary", "analysis"} else 4))
        candidate_k = min(20, max(base_k * 3, 12))
        hits = self._search_queries(
            plan.queries,
            top_k=candidate_k,
        )
        hits = self._rerank(question, hits, top_k=candidate_k)
        if plan.answer_mode in {"summary", "analysis"}:
            hits = self._diversify_hits(hits, top_k=base_k)
        else:
            hits = hits[:base_k]
        hits = self._expand_context(
            hits,
            max_hits=min(10, base_k + (3 if plan.answer_mode in {"summary", "analysis"} else 1)),
        )
        if not hits:
            return AskResponse(
                answer=REFUSAL_MESSAGE,
                refused=True,
                citations=[],
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                mode="refusal",
                answer_mode=plan.answer_mode,
                rewritten_queries=list(plan.queries),
            )

        if self.generator is not None:
            try:
                generated = self.generator.generate(
                    question,
                    hits,
                    answer_mode=plan.answer_mode,
                    history=turns,
                )
                self.last_generation_error = None
                if not generated.supported:
                    return AskResponse(
                        answer=REFUSAL_MESSAGE,
                        refused=True,
                        citations=[],
                        latency_ms=round((time.perf_counter() - started) * 1000, 3),
                        mode="deepseek",
                        answer_mode=plan.answer_mode,
                        rewritten_queries=list(plan.queries),
                    )
                citation_ids = set(generated.citation_ids)
                cited_hits = [hit for hit in hits if hit.chunk.chunk_id in citation_ids]
                citations = self._citations(cited_hits)
                cited_answer = generated.answer
                inline_markers = {
                    f"[{value}]"
                    for hit in cited_hits
                    for value in (hit.chunk.chunk_id, hit.chunk.source_id)
                }
                if not any(marker in cited_answer for marker in inline_markers):
                    cited_answer += "\n\n依据：" + "、".join(
                        f"[{chunk_id}]" for chunk_id in generated.citation_ids
                    )
                return AskResponse(
                    answer=cited_answer,
                    refused=False,
                    citations=citations,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    mode="deepseek",
                    answer_mode=plan.answer_mode,
                    rewritten_queries=list(plan.queries),
                )
            except Exception as exc:  # Keep the local service useful during API outages.
                self.last_generation_error = f"{type(exc).__name__}: {exc}"

        return self._extractive_response(question, hits, started, plan)

    def _rerank(self, question: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if self.reranker is None or not hits:
            return hits[:top_k]
        try:
            reranked = self.reranker.rerank(question, hits, top_k=top_k)
            self.last_reranker_error = None
            return reranked
        except Exception as exc:  # Retrieval remains available if the local model fails.
            self.last_reranker_error = f"{type(exc).__name__}: {exc}"
            return hits[:top_k]

    def _plan(
        self,
        question: str,
        answer_mode: AnswerMode,
        history: list[ConversationTurn],
    ) -> QueryPlan:
        fallback = heuristic_plan(question, answer_mode)
        planner = getattr(self.generator, "plan", None)
        if planner is None:
            self.last_planning_error = None
            return fallback
        try:
            plan = planner(question, requested_mode=answer_mode, history=history)
            self.last_planning_error = None
            return plan
        except Exception as exc:  # Planning is an enhancement; retrieval must still work without it.
            self.last_planning_error = f"{type(exc).__name__}: {exc}"
            return fallback

    def _search_queries(
        self,
        queries: tuple[str, ...],
        *,
        top_k: int,
        diversify: bool = False,
    ) -> list[SearchHit]:
        merged: dict[str, SearchHit] = {}
        candidate_count = min(20, max(top_k * 2, 8))
        for query_index, query in enumerate(queries):
            for rank, hit in enumerate(self.retriever.search(query, top_k=candidate_count)):
                bonus = 0.04 / (rank + 1) + (0.01 if query_index == 0 else 0.0)
                scored = SearchHit(chunk=hit.chunk, score=hit.score + bonus)
                current = merged.get(hit.chunk.chunk_id)
                if current is None or scored.score > current.score:
                    merged[hit.chunk.chunk_id] = scored
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if not diversify:
            return ranked[:top_k]
        return self._diversify_hits(ranked, top_k=top_k)

    @staticmethod
    def _diversify_hits(ranked: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not ranked:
            return []
        selected: list[SearchHit] = []
        selected_ids: set[str] = set()
        selected_sources: set[str] = set()
        source_floor = ranked[0].score * 0.72

        # First cover distinct, still-relevant sources so repeated chunks from one
        # document cannot crowd every other project out of a summary.
        for hit in ranked:
            if hit.score < source_floor:
                break
            if hit.chunk.source_id not in selected_sources:
                selected.append(hit)
                selected_ids.add(hit.chunk.chunk_id)
                selected_sources.add(hit.chunk.source_id)
                if len(selected) >= top_k:
                    return selected

        # Then prefer non-adjacent sections. Immediate neighbours are added later
        # by _expand_context, so anchors should cover different parts first.
        for hit in ranked:
            if hit.chunk.chunk_id in selected_ids:
                continue
            too_close = any(
                existing.chunk.source_id == hit.chunk.source_id
                and abs(existing.chunk.order - hit.chunk.order) <= 1
                for existing in selected
            )
            if too_close:
                continue
            selected.append(hit)
            selected_ids.add(hit.chunk.chunk_id)
            if len(selected) >= top_k:
                return selected

        for hit in ranked:
            if hit.chunk.chunk_id in selected_ids:
                continue
            selected.append(hit)
            selected_ids.add(hit.chunk.chunk_id)
            if len(selected) >= top_k:
                break
        return selected

    def _expand_context(self, hits: list[SearchHit], *, max_hits: int) -> list[SearchHit]:
        if not hits or len(hits) >= max_hits:
            return hits[:max_hits]
        by_position = {(chunk.source_id, chunk.order): chunk for chunk in self.active_chunks}
        expanded = list(hits)
        seen = {hit.chunk.chunk_id for hit in expanded}
        for hit in hits:
            for offset in (-1, 1):
                neighbor = by_position.get((hit.chunk.source_id, hit.chunk.order + offset))
                if neighbor is None or neighbor.chunk_id in seen:
                    continue
                expanded.append(SearchHit(chunk=neighbor, score=hit.score * 0.82))
                seen.add(neighbor.chunk_id)
                if len(expanded) >= max_hits:
                    return expanded
        return expanded

    @staticmethod
    def _citations(hits) -> list[Citation]:
        return [
            Citation(
                source_id=hit.chunk.source_id,
                chunk_id=hit.chunk.chunk_id,
                score=round(hit.score, 4),
                excerpt=hit.chunk.text[:180],
                filename=hit.chunk.metadata.get("filename"),
                page=hit.chunk.metadata.get("page"),
                section=hit.chunk.metadata.get("section"),
            )
            for hit in hits
        ]

    def _extractive_response(
        self,
        question: str,
        hits,
        started: float,
        plan: QueryPlan,
    ) -> AskResponse:

        question_tokens = set(tokenize(question))
        candidates: list[tuple[float, str, str]] = []
        for hit in hits:
            for sentence in split_sentences(hit.chunk.text):
                sentence_tokens = set(tokenize(sentence))
                overlap = len(question_tokens & sentence_tokens) / max(len(question_tokens), 1)
                candidates.append((hit.score + overlap, sentence, hit.chunk.chunk_id))
        candidates.sort(key=lambda item: item[0], reverse=True)

        selected: list[tuple[str, str]] = []
        seen: set[str] = set()
        for _, sentence, chunk_id in candidates:
            normalized = re.sub(r"\s+", "", sentence)
            if normalized in seen:
                continue
            selected.append((sentence, chunk_id))
            seen.add(normalized)
            if len(selected) == 3:
                break

        answer = "根据知识库资料：" + " ".join(
            f"{sentence} [{chunk_id}]" for sentence, chunk_id in selected
        )
        warning = None
        if self.generator is not None and self.last_generation_error:
            error_type = self.last_generation_error.split(":", 1)[0]
            warning = f"DeepSeek 最终回答失败（{error_type}），已自动显示本地抽取结果。"
        return AskResponse(
            answer=answer,
            refused=False,
            citations=self._citations(hits),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            mode="extractive-fallback" if self.generator is not None else "extractive",
            answer_mode=plan.answer_mode,
            rewritten_queries=list(plan.queries),
            warning=warning,
        )

