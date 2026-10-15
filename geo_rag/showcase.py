from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .models import AskResponse
from .service import RagService


@dataclass(frozen=True)
class ShowcaseCase:
    question: str
    answer_mode: str
    expected_sources: tuple[str, ...]
    required_points: tuple[str, ...]


def load_showcase_cases(path: Path) -> list[ShowcaseCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        cases.append(
            ShowcaseCase(
                question=str(value["question"]),
                answer_mode=str(value.get("answer_mode", "summary")),
                expected_sources=tuple(value["expected_sources"]),
                required_points=tuple(value["required_points"]),
            )
        )
    return cases


def score_showcase_case(
    case: ShowcaseCase,
    response: AskResponse,
    valid_chunk_ids: set[str],
) -> dict:
    returned_sources = {citation.source_id for citation in response.citations}
    expected_sources = set(case.expected_sources)
    source_coverage = len(returned_sources & expected_sources) / max(len(expected_sources), 1)
    matched_points = [point for point in case.required_points if point.lower() in response.answer.lower()]
    point_recall = len(matched_points) / max(len(case.required_points), 1)
    citation_valid = bool(response.citations) and all(
        citation.chunk_id in valid_chunk_ids for citation in response.citations
    )
    generated = response.mode == "deepseek" and not response.refused
    passed = generated and source_coverage == 1.0 and point_recall >= 0.6 and citation_valid
    return {
        "question": case.question,
        "answer_mode": case.answer_mode,
        "passed": passed,
        "generation_success": generated,
        "source_coverage": round(source_coverage, 3),
        "key_point_recall": round(point_recall, 3),
        "citation_valid": citation_valid,
        "expected_sources": list(case.expected_sources),
        "returned_sources": sorted(returned_sources),
        "matched_points": matched_points,
        "required_points": list(case.required_points),
        "latency_ms": response.latency_ms,
        "answer": response.answer,
    }


def run_showcase_evaluation(settings: Settings, root: Path) -> dict:
    if not settings.deepseek_api_key:
        raise ValueError("DeepSeek API key is required for showcase evaluation")
    storage_path = root / "data" / "storage" / "showcase_eval.json"
    eval_settings = replace(
        settings,
        storage_path=storage_path,
        semantic_enabled=True,
        generation_enabled=True,
    )
    service = RagService(eval_settings)
    sample_names = set()
    for path in sorted((root / "data" / "sample_docs").glob("*.md")):
        source_id = path.stem
        sample_names.add(source_id)
        service.ingest(source_id, path.read_text(encoding="utf-8"), {"filename": path.name})
    for source_id in list(service.source_ids):
        if source_id not in sample_names:
            service.delete_source(source_id)

    valid_chunk_ids = {chunk.chunk_id for chunk in service.chunks}
    details = []
    for case in load_showcase_cases(root / "data" / "eval" / "showcase_questions.jsonl"):
        response = service.ask(
            case.question,
            top_k=8,
            answer_mode=case.answer_mode,  # type: ignore[arg-type]
        )
        details.append(score_showcase_case(case, response, valid_chunk_ids))

    count = len(details) or 1
    report = {
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": len(details),
        "passed_cases": sum(item["passed"] for item in details),
        "generation_success_rate": round(
            sum(item["generation_success"] for item in details) / count, 3
        ),
        "source_coverage": round(sum(item["source_coverage"] for item in details) / count, 3),
        "key_point_recall": round(
            sum(item["key_point_recall"] for item in details) / count, 3
        ),
        "citation_validity": round(sum(item["citation_valid"] for item in details) / count, 3),
        "mean_latency_ms": round(sum(item["latency_ms"] for item in details) / count, 3),
        "reranker_enabled": service.reranker_enabled,
        "reranker_model": settings.reranker_model if service.reranker_enabled else None,
        "details": details,
    }
    report_path = root / "reports" / "showcase_latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_showcase_report(root: Path) -> dict:
    path = root / "reports" / "showcase_latest.json"
    if not path.exists():
        return {"status": "not_run"}
    return json.loads(path.read_text(encoding="utf-8"))
