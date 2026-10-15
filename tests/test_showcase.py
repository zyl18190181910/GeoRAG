import json
from dataclasses import replace

from geo_rag.config import Settings
from geo_rag.models import AskResponse, Citation, DocumentChunk
from geo_rag import showcase
from geo_rag.showcase import (
    ShowcaseCase,
    load_showcase_cases,
    load_showcase_report,
    run_showcase_evaluation,
    score_showcase_case,
)


def test_load_showcase_cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "归纳测试流程",
                "answer_mode": "summary",
                "expected_sources": ["规范"],
                "required_points": ["接口测试", "拒答"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    case = load_showcase_cases(path)[0]
    assert case.expected_sources == ("规范",)


def test_score_showcase_case_checks_coverage_points_and_citations():
    case = ShowcaseCase("归纳", "summary", ("规范",), ("接口测试", "拒答"))
    response = AskResponse(
        answer="检查接口测试，并验证资料不足时能够拒答。[规范:0]",
        refused=False,
        citations=[Citation(source_id="规范", chunk_id="规范:0", score=0.9, excerpt="片段")],
        latency_ms=1200,
        mode="deepseek",
        answer_mode="summary",
    )
    result = score_showcase_case(case, response, {"规范:0"})
    assert result["passed"] is True
    assert result["source_coverage"] == 1.0
    assert result["key_point_recall"] == 1.0


def test_run_and_load_showcase_report_with_isolated_service(monkeypatch, tmp_path, settings):
    sample_dir = tmp_path / "data" / "sample_docs"
    eval_dir = tmp_path / "data" / "eval"
    sample_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (sample_dir / "规范.md").write_text("接口测试需要验证拒答。", encoding="utf-8")
    (eval_dir / "showcase_questions.jsonl").write_text(
        json.dumps(
            {
                "question": "归纳检查项",
                "answer_mode": "summary",
                "expected_sources": ["规范"],
                "required_points": ["接口测试", "拒答"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self, active_settings):
            self.source_ids = []
            self.chunks = [DocumentChunk("规范:0", "规范", "接口测试需要验证拒答。", 0)]
            self.reranker_enabled = True

        def ingest(self, source_id, text, metadata):
            self.source_ids = [source_id]

        def delete_source(self, source_id):
            return 1

        def ask(self, question, top_k, answer_mode):
            return AskResponse(
                answer="检查接口测试与拒答。[规范:0]",
                refused=False,
                citations=[Citation(source_id="规范", chunk_id="规范:0", score=0.9, excerpt="片段")],
                latency_ms=800,
                mode="deepseek",
                answer_mode="summary",
            )

    monkeypatch.setattr(showcase, "RagService", FakeService)
    configured = replace(settings, deepseek_api_key="test", reranker_enabled=True)
    report = run_showcase_evaluation(configured, tmp_path)
    assert report["passed_cases"] == 1
    assert report["citation_validity"] == 1.0
    assert load_showcase_report(tmp_path)["status"] == "complete"
