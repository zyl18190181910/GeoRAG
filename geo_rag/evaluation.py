from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .service import RagService


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_sources: tuple[str, ...]
    should_refuse: bool = False


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            cases.append(
                EvalCase(
                    question=value["question"],
                    expected_sources=tuple(value.get("expected_sources", [])),
                    should_refuse=bool(value.get("should_refuse", False)),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation case at line {number}") from exc
    return cases


def evaluate(service: RagService, cases: list[EvalCase], top_k: int = 4) -> dict:
    if not cases:
        raise ValueError("evaluation set is empty")
    source_hits = 0
    source_cases = 0
    refusal_correct = 0
    latencies: list[float] = []
    details: list[dict] = []

    for case in cases:
        response = service.ask(case.question, top_k=top_k)
        returned_sources = {citation.source_id for citation in response.citations}
        source_hit = True
        if case.expected_sources:
            source_cases += 1
            source_hit = bool(returned_sources & set(case.expected_sources))
            source_hits += int(source_hit)
        refusal_match = response.refused == case.should_refuse
        refusal_correct += int(refusal_match)
        latencies.append(response.latency_ms)
        details.append(
            {
                "question": case.question,
                "refused": response.refused,
                "source_hit": source_hit,
                "returned_sources": sorted(returned_sources),
            }
        )

    return {
        "cases": len(cases),
        "source_hit_at_k": round(source_hits / source_cases, 4) if source_cases else None,
        "refusal_accuracy": round(refusal_correct / len(cases), 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 3),
        "details": details,
    }

