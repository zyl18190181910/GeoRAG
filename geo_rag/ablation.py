from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .chunking import chunk_text
from .config import Settings
from .loaders import extract_sections_from_bytes, safe_source_id
from .models import DocumentChunk, SearchHit
from .reranking import CrossEncoderReranker
from .retrieval import SparseTfidfRetriever
from .semantic import HybridRetriever, SemanticRetriever, SentenceTransformerEmbedder


SearchFunction = Callable[[str, int], list[SearchHit]]


@dataclass(frozen=True)
class AblationCase:
    question: str
    expected_source: str
    evidence_terms: tuple[str, ...]


def load_ablation_cases(path: Path) -> list[AblationCase]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            cases.append(
                AblationCase(
                    question=str(value["question"]),
                    expected_source=str(value["expected_source"]),
                    evidence_terms=tuple(str(item) for item in value["evidence_terms"]),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid ablation case at line {number}") from exc
    return cases


def score_retrieval_method(
    cases: list[AblationCase],
    search: SearchFunction,
    *,
    top_k: int = 3,
) -> dict:
    if not cases:
        raise ValueError("ablation evaluation set is empty")

    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    failures: list[dict] = []

    for case in cases:
        started = time.perf_counter()
        hits = search(case.question, top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        first_rank = next(
            (
                rank
                for rank, hit in enumerate(hits, start=1)
                if hit.chunk.source_id == case.expected_source
                and any(term.lower() in hit.chunk.text.lower() for term in case.evidence_terms)
            ),
            None,
        )
        hit_at_1 += int(first_rank == 1)
        hit_at_k += int(first_rank is not None and first_rank <= top_k)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        if first_rank is None:
            failures.append(
                {
                    "question": case.question,
                    "expected_source": case.expected_source,
                    "evidence_terms": list(case.evidence_terms),
                    "returned_sources": [hit.chunk.source_id for hit in hits],
                }
            )

    count = len(cases)
    return {
        "hit_at_1": round(hit_at_1 / count, 4),
        "hit_at_k": round(hit_at_k / count, 4),
        "mrr": round(sum(reciprocal_ranks) / count, 4),
        "mean_latency_ms": round(sum(latencies) / count, 3),
        "failed_cases": len(failures),
        "failures": failures,
    }


def _load_sample_chunks(root: Path, settings: Settings) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in sorted((root / "data" / "sample_docs").glob("*.md")):
        chunks.extend(
            chunk_text(
                path.stem,
                path.read_text(encoding="utf-8"),
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                metadata={"filename": path.name},
            )
        )
    if not chunks:
        raise ValueError("sample documents are missing")
    return chunks


def _load_public_paper_chunks(root: Path, settings: Settings) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in sorted((root / "data" / "public_papers").glob("*.pdf")):
        source_id = safe_source_id(path.name)
        source_chunks: list[DocumentChunk] = []
        for section in extract_sections_from_bytes(path.name, path.read_bytes()):
            local_chunks = chunk_text(
                source_id,
                section.text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                metadata={"filename": path.name, **section.metadata},
            )
            for local_chunk in local_chunks:
                order = len(source_chunks)
                source_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{source_id}:{order}",
                        source_id=source_id,
                        text=local_chunk.text,
                        order=order,
                        metadata=local_chunk.metadata,
                    )
                )
        chunks.extend(source_chunks)
    if not chunks:
        raise ValueError(
            "public paper PDFs are missing; run scripts/download_public_papers.py first"
        )
    return chunks


def _load_ablation_assets(
    root: Path,
    settings: Settings,
) -> tuple[list[DocumentChunk], list[AblationCase], str]:
    public_case_path = root / "data" / "eval" / "public_paper_questions.jsonl"
    public_papers = list((root / "data" / "public_papers").glob("*.pdf"))
    if public_case_path.exists() and public_papers:
        return (
            _load_public_paper_chunks(root, settings),
            load_ablation_cases(public_case_path),
            "60道公开论文中英双语检索集",
        )
    return (
        _load_sample_chunks(root, settings),
        load_ablation_cases(root / "data" / "eval" / "ablation_questions.jsonl"),
        "原创语义改写检索集",
    )


def run_ablation_evaluation(settings: Settings, root: Path, *, top_k: int = 3) -> dict:
    chunks, cases, dataset_name = _load_ablation_assets(root, settings)

    sparse = SparseTfidfRetriever()
    sparse.fit(chunks)

    semantic = SemanticRetriever(
        SentenceTransformerEmbedder(
            settings.embedding_model,
            settings.model_cache_path,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
    )
    hybrid = HybridRetriever(
        sparse_min_score=settings.min_score,
        semantic_min_score=settings.semantic_min_score,
        semantic=semantic,
    )
    hybrid.fit(chunks)
    if not hybrid.semantic_ready:
        raise ValueError(f"semantic retriever unavailable: {hybrid.semantic_error}")

    reranker = CrossEncoderReranker(settings.reranker_model, settings.model_cache_path)
    warmup_question = cases[0].question
    warmup_candidates = hybrid.search(warmup_question, top_k=min(20, len(chunks)))
    reranker.rerank(warmup_question, warmup_candidates, top_k=top_k)

    def keyword_search(question: str, limit: int) -> list[SearchHit]:
        return sparse.search(question, top_k=limit, min_score=0.000001)

    def hybrid_search(question: str, limit: int) -> list[SearchHit]:
        return hybrid.search(question, top_k=limit)

    def reranked_search(question: str, limit: int) -> list[SearchHit]:
        candidates = hybrid.search(question, top_k=min(20, len(chunks)))
        return reranker.rerank(question, candidates, top_k=limit)

    definitions = [
        (
            "keyword",
            "关键词检索",
            "TF-IDF 词项匹配基线",
            keyword_search,
        ),
        (
            "hybrid",
            "混合检索",
            "TF-IDF + BGE 语义向量",
            hybrid_search,
        ),
        (
            "reranked",
            "混合检索 + 精排",
            "候选召回 + BGE Cross-Encoder",
            reranked_search,
        ),
    ]
    methods = []
    for method_id, name, description, search in definitions:
        metrics = score_retrieval_method(cases, search, top_k=top_k)
        methods.append(
            {
                "id": method_id,
                "name": name,
                "description": description,
                **metrics,
            }
        )

    baseline = methods[0]
    final = methods[-1]
    report = {
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_name,
        "cases": len(cases),
        "top_k": top_k,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "methods": methods,
        "improvement": {
            "hit_at_1_points": round((final["hit_at_1"] - baseline["hit_at_1"]) * 100, 1),
            "hit_at_k_points": round((final["hit_at_k"] - baseline["hit_at_k"]) * 100, 1),
            "mrr_points": round((final["mrr"] - baseline["mrr"]) * 100, 1),
        },
    }
    report_path = root / "reports" / "ablation_latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_ablation_report(root: Path) -> dict:
    path = root / "reports" / "ablation_latest.json"
    if not path.exists():
        return {"status": "not_run"}
    return json.loads(path.read_text(encoding="utf-8"))
