import json
from dataclasses import replace

import pytest

from geo_rag import ablation
from geo_rag.ablation import (
    AblationCase,
    load_ablation_cases,
    load_ablation_report,
    run_ablation_evaluation,
    score_retrieval_method,
)
from geo_rag.models import DocumentChunk, SearchHit
from geo_rag.loaders import LoadedSection


def test_score_retrieval_method_calculates_hit_and_mrr():
    chunks = {
        "a": DocumentChunk("a:0", "a", "alpha", 0),
        "b": DocumentChunk("b:0", "b", "beta", 0),
    }
    cases = [
        AblationCase("first", "a", ("alpha",)),
        AblationCase("second", "b", ("beta",)),
    ]

    def search(question, top_k):
        if question == "first":
            return [SearchHit(chunks["b"], 0.9), SearchHit(chunks["a"], 0.8)][:top_k]
        return [SearchHit(chunks["b"], 0.9)][:top_k]

    result = score_retrieval_method(cases, search, top_k=3)
    assert result["hit_at_1"] == 0.5
    assert result["hit_at_k"] == 1.0
    assert result["mrr"] == 0.75
    assert result["failed_cases"] == 0


def test_load_ablation_report_before_first_run(tmp_path):
    assert load_ablation_report(tmp_path) == {"status": "not_run"}


def test_score_retrieval_method_records_failed_case():
    case = AblationCase("missing", "a", ("expected phrase",))

    result = score_retrieval_method([case], lambda question, top_k: [], top_k=3)

    assert result["hit_at_k"] == 0.0
    assert result["failed_cases"] == 1
    assert result["failures"][0]["expected_source"] == "a"


def test_score_retrieval_method_rejects_empty_set():
    with pytest.raises(ValueError, match="empty"):
        score_retrieval_method([], lambda question, top_k: [])


def test_load_ablation_cases_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_ablation_cases(path)


def test_run_ablation_evaluation_writes_report(monkeypatch, tmp_path, settings):
    sample_dir = tmp_path / "data" / "sample_docs"
    eval_dir = tmp_path / "data" / "eval"
    sample_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (sample_dir / "甲.md").write_text("甲资料包含目标证据。", encoding="utf-8")
    (sample_dir / "乙.md").write_text("乙资料包含另一段内容。", encoding="utf-8")
    (eval_dir / "ablation_questions.jsonl").write_text(
        json.dumps(
            {
                "question": "目标在哪里？",
                "expected_source": "甲",
                "evidence_terms": ["目标证据"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class FakeEmbedder:
        def encode(self, texts):
            return [[1.0, 0.0] if "目标" in text or "甲" in text else [0.0, 1.0] for text in texts]

    class FakeReranker:
        def __init__(self, model_name, cache_path):
            pass

        def rerank(self, question, hits, *, top_k):
            return hits[:top_k]

    monkeypatch.setattr(
        ablation,
        "SentenceTransformerEmbedder",
        lambda *args, **kwargs: FakeEmbedder(),
    )
    monkeypatch.setattr(ablation, "CrossEncoderReranker", FakeReranker)
    configured = replace(settings, model_cache_path=tmp_path / "models")

    report = run_ablation_evaluation(configured, tmp_path)

    assert report["status"] == "complete"
    assert report["cases"] == 1
    assert len(report["methods"]) == 3
    assert load_ablation_report(tmp_path)["dataset"] == "原创语义改写检索集"


def test_public_paper_assets_take_precedence(monkeypatch, tmp_path, settings):
    paper_dir = tmp_path / "data" / "public_papers"
    eval_dir = tmp_path / "data" / "eval"
    paper_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (paper_dir / "公开论文.pdf").write_bytes(b"%PDF-placeholder")
    (eval_dir / "public_paper_questions.jsonl").write_text(
        json.dumps(
            {
                "question": "论文说了什么？",
                "expected_source": "公开论文",
                "evidence_terms": ["原文证据"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ablation,
        "extract_sections_from_bytes",
        lambda filename, content: [LoadedSection("这里包含原文证据。", {"page": 3})],
    )

    chunks, cases, dataset = ablation._load_ablation_assets(tmp_path, settings)

    assert dataset == "60道公开论文中英双语检索集"
    assert cases[0].expected_source == "公开论文"
    assert chunks[0].source_id == "公开论文"
    assert chunks[0].metadata["page"] == 3
