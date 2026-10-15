from dataclasses import replace
from pathlib import Path

import pytest

from geo_rag.config import Settings
from geo_rag.evaluation import EvalCase, evaluate
from geo_rag.generation import GeneratedAnswer, QueryPlan
from geo_rag.models import DocumentChunk, SearchHit
from geo_rag.service import RagService


class MeaningEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if any(term in text for term in ("项目经历", "参与过哪些研究", "做过什么工作")):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class SummaryGenerator:
    def generate(self, question, hits, **kwargs):
        return GeneratedAnswer("参与了岩石识别研究。", True, (hits[0].chunk.chunk_id,))


class SourceCitationGenerator:
    def generate(self, question, hits, **kwargs):
        hit = hits[0]
        return GeneratedAnswer(
            f"参与了岩石识别研究 [{hit.chunk.source_id}]。",
            True,
            (hit.chunk.chunk_id,),
        )


class BrokenGenerator:
    def generate(self, question, hits, **kwargs):
        raise ConnectionError("offline")


class PlanningGenerator:
    def __init__(self):
        self.received_hits = []

    def plan(self, question, **kwargs):
        return QueryPlan("summary", (question, "项目经历 Python 图像识别"))

    def generate(self, question, hits, **kwargs):
        self.received_hits = hits
        return GeneratedAnswer("整理后的项目总结。", True, (hits[0].chunk.chunk_id,))


class PreferSecondReranker:
    def rerank(self, question, hits, *, top_k):
        return sorted(hits, key=lambda hit: hit.chunk.source_id != "second")[:top_k]


def test_service_refuses_when_store_is_empty(settings: Settings):
    response = RagService(settings).ask("什么是岩石分类？")
    assert response.refused is True
    assert response.citations == []


def test_service_ingests_and_answers_with_citation(settings: Settings):
    service = RagService(settings)
    service.ingest("rock", "低置信度岩石样本应标记为未知类别，不应强制分类。")
    response = service.ask("低置信度岩石样本怎样处理？")
    assert response.refused is False
    assert response.citations[0].source_id == "rock"
    assert "[rock:0]" in response.answer


def test_section_ingestion_preserves_precise_citation_location(settings: Settings):
    service = RagService(settings)
    service.ingest_sections(
        "report",
        [
            ("数据准备按采集区域划分。", {"page": 2}),
            ("模型评测需要检查混淆矩阵。", {"section": "模型评测"}),
        ],
        metadata={"filename": "地学报告.pdf"},
    )
    response = service.ask("数据准备怎样划分？")
    citation = response.citations[0]
    assert citation.filename == "地学报告.pdf"
    assert citation.page == 2
    assert citation.section is None


def test_ingesting_same_source_replaces_old_chunks(settings: Settings):
    service = RagService(settings)
    service.ingest("guide", "旧版本介绍岩石。")
    service.ingest("guide", "新版本介绍重力异常反演。")
    assert service.source_ids == ["guide"]
    assert all("旧版本" not in chunk.text for chunk in service.chunks)


def test_service_persists_across_instances(settings: Settings):
    first = RagService(settings)
    first.ingest("gravity", "再正演误差用于物理一致性验证。")
    second = RagService(settings)
    assert second.source_ids == ["gravity"]
    assert second.ask("再正演误差有什么用途？").refused is False


def test_disabled_source_state_persists(settings: Settings):
    first = RagService(settings)
    first.ingest("gravity", "再正演误差用于物理一致性验证。")
    first.set_source_enabled("gravity", False)
    second = RagService(settings)
    assert second.source_details()[0].enabled is False
    assert second.active_chunks == []


def test_service_rejects_blank_document(settings: Settings):
    service = RagService(settings)
    with pytest.raises(ValueError, match="empty"):
        service.ingest("blank", "  \n")


def test_semantic_retrieval_handles_paraphrased_question(settings: Settings):
    semantic_settings = replace(settings, semantic_enabled=True, semantic_min_score=0.4)
    service = RagService(semantic_settings, semantic_embedder=MeaningEmbedder())
    service.ingest("resume", "项目经历包括复杂岩石图像识别与模型评测。")
    service.ingest("other", "食堂开放时间为每天中午十二点。")
    response = service.ask("参与过哪些研究？")
    assert response.refused is False
    assert response.citations[0].source_id == "resume"


def test_generator_can_summarize_and_preserve_citation(settings: Settings):
    generated_settings = replace(settings, generation_enabled=True)
    service = RagService(generated_settings, answer_generator=SummaryGenerator())
    service.ingest("resume", "项目经历包括岩石图像识别研究。")
    response = service.ask("概括做过的研究")
    assert response.mode == "deepseek"
    assert response.answer.startswith("参与了岩石识别研究")
    assert response.citations[0].source_id == "resume"


def test_source_level_inline_citation_does_not_add_duplicate_evidence_line(settings: Settings):
    generated_settings = replace(settings, generation_enabled=True)
    service = RagService(generated_settings, answer_generator=SourceCitationGenerator())
    service.ingest("resume", "项目经历包括岩石图像识别研究。")

    response = service.ask("概括做过的研究")

    assert "[resume]" in response.answer
    assert "依据：" not in response.answer


def test_generator_failure_falls_back_to_local_answer(settings: Settings):
    generated_settings = replace(settings, generation_enabled=True)
    service = RagService(generated_settings, answer_generator=BrokenGenerator())
    service.ingest("resume", "项目经历包括岩石图像识别研究。")
    response = service.ask("项目经历包括什么？")
    assert response.mode == "extractive-fallback"
    assert response.refused is False
    assert "ConnectionError" in service.last_generation_error
    assert response.warning == "DeepSeek 最终回答失败（ConnectionError），已自动显示本地抽取结果。"


def test_planner_rewrites_query_before_retrieval(settings: Settings):
    generator = PlanningGenerator()
    generated_settings = replace(settings, generation_enabled=True)
    service = RagService(generated_settings, answer_generator=generator)
    service.ingest("resume", "项目经历使用 Python 完成复杂岩石图像识别。")
    response = service.ask("我以前具体干了啥？")
    assert response.refused is False
    assert response.answer_mode == "summary"
    assert response.rewritten_queries[-1] == "项目经历 Python 图像识别"
    assert generator.received_hits[0].chunk.source_id == "resume"


def test_summary_mode_adds_neighboring_context(settings: Settings):
    generator = PlanningGenerator()
    generated_settings = replace(settings, generation_enabled=True)
    service = RagService(generated_settings, answer_generator=generator)
    first = "项目经历概述：" + "使用图像算法开展识别工作，" * 10 + "。"
    second = "后续形成了模型训练、误差分析和系统集成能力，" * 9 + "。"
    service.ingest("resume", first + second)
    service.ask("概括项目经历", top_k=1, answer_mode="summary")
    assert {hit.chunk.order for hit in generator.received_hits} >= {0, 1}


def test_summary_diversifies_relevant_sources():
    ranked = [
        SearchHit(DocumentChunk("a:0", "a", "项目甲", 0), 0.90),
        SearchHit(DocumentChunk("a:1", "a", "项目甲重复", 1), 0.88),
        SearchHit(DocumentChunk("b:0", "b", "项目乙", 0), 0.84),
        SearchHit(DocumentChunk("c:0", "c", "项目丙", 0), 0.80),
    ]
    selected = RagService._diversify_hits(ranked, top_k=3)
    assert [hit.chunk.source_id for hit in selected] == ["a", "b", "c"]


def test_service_uses_reranker_before_answering(settings: Settings):
    service = RagService(settings, reranker=PreferSecondReranker())
    service.ingest("first", "岩石识别需要独立测试集。")
    service.ingest("second", "岩石识别需要检查混淆矩阵。")
    response = service.ask("岩石识别需要检查什么？", top_k=2)
    assert response.citations[0].source_id == "second"


def test_evaluation_reports_source_hit_and_refusal(settings: Settings):
    service = RagService(settings)
    service.ingest("rock", "Macro-F1 用于观察少数类别的识别效果。")
    cases = [
        EvalCase("Macro-F1 有什么作用？", ("rock",), False),
        EvalCase("火星天气如何？", (), True),
    ]
    result = evaluate(service, cases)
    assert result["source_hit_at_k"] == 1.0
    assert result["refusal_accuracy"] == 1.0

