"""Validate that every public-paper evaluation case has matching source evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_rag.ablation import _load_public_paper_chunks  # noqa: E402
from geo_rag.config import Settings  # noqa: E402


def main() -> None:
    root = ROOT
    cases_path = root / "data" / "eval" / "public_paper_questions.jsonl"
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = _load_public_paper_chunks(root, Settings.from_env())

    missing: list[tuple[int, str, str, list[str]]] = []
    for index, case in enumerate(cases, start=1):
        source_texts = [
            chunk.text.casefold()
            for chunk in chunks
            if chunk.source_id == case["expected_source"]
        ]
        evidence_terms = [term.casefold() for term in case["evidence_terms"]]
        if not any(
            any(term in text for term in evidence_terms) for text in source_texts
        ):
            missing.append(
                (
                    index,
                    case["question"],
                    case["expected_source"],
                    case["evidence_terms"],
                )
            )

    print(f"cases={len(cases)} chunks={len(chunks)} missing={len(missing)}")
    for item in missing:
        print(item)

    if len(cases) != 60 or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
