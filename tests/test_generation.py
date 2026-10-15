import json
from types import SimpleNamespace

import httpx
import openai

from geo_rag.generation import DeepSeekGenerator, heuristic_plan
from geo_rag.models import DocumentChunk, SearchHit


def test_parse_json_accepts_markdown_fence():
    payload = DeepSeekGenerator._parse_json(
        '```json\n{"supported": true, "answer": "摘要", "citation_ids": ["doc:0"]}\n```'
    )
    assert payload == {
        "supported": True,
        "answer": "摘要",
        "citation_ids": ["doc:0"],
    }


def test_parse_json_rejects_non_json():
    try:
        DeepSeekGenerator._parse_json("not-json")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("invalid JSON should be rejected")


def test_heuristic_plan_detects_summary_and_analysis_modes():
    assert heuristic_plan("请归纳项目经历").answer_mode == "summary"
    assert heuristic_plan("分析我的能力是否匹配").answer_mode == "analysis"
    assert heuristic_plan("项目经历有哪些").answer_mode == "fact"


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    content = '{"supported": true, "answer": "归纳后的回答", "citation_ids": ["doc:0"]}'

    def __init__(self, **kwargs):
        self.options = kwargs
        self.chat = SimpleNamespace(completions=FakeCompletions(self.content))


def _hit() -> SearchHit:
    return SearchHit(DocumentChunk("doc:0", "doc", "项目经历包括岩石图像识别。", 0), 0.8)


def test_deepseek_generator_requests_grounded_json(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    generator = DeepSeekGenerator(
        api_key="secret-not-sent",
        model="deepseek-flash",
        base_url="https://example.invalid",
        reasoning_effort="high",
    )
    result = generator.generate("概括做过的研究", [_hit()])
    assert result.supported is True
    assert result.answer == "归纳后的回答"
    assert result.citation_ids == ("doc:0",)
    assert generator.client.chat.completions.last_request["response_format"] == {
        "type": "json_object"
    }
    assert generator.client.chat.completions.last_request["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_deepseek_generator_rebuilds_connection_and_retries(monkeypatch):
    attempts = {"count": 0}

    class FlakyCompletions(FakeCompletions):
        def create(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
            return super().create(**kwargs)

    class FlakyOpenAI(FakeOpenAI):
        def __init__(self, **kwargs):
            self.options = kwargs
            self.chat = SimpleNamespace(completions=FlakyCompletions(self.content))

    monkeypatch.setattr(openai, "OpenAI", FlakyOpenAI)
    monkeypatch.setattr("geo_rag.generation.time.sleep", lambda seconds: None)
    generator = DeepSeekGenerator(
        api_key="secret-not-sent",
        model="deepseek-flash",
        base_url="https://example.invalid",
    )

    result = generator.generate("概括项目", [_hit()])

    assert result.supported is True
    assert attempts["count"] == 2


def test_deepseek_generator_rejects_unknown_citation(monkeypatch):
    class UnknownCitationClient(FakeOpenAI):
        content = '{"supported": true, "answer": "无依据回答", "citation_ids": ["other:9"]}'

    monkeypatch.setattr(openai, "OpenAI", UnknownCitationClient)
    generator = DeepSeekGenerator(
        api_key="secret-not-sent",
        model="deepseek-flash",
        base_url="https://example.invalid",
    )
    try:
        generator.generate("问题", [_hit()])
    except ValueError as exc:
        assert "grounded citations" in str(exc)
    else:
        raise AssertionError("unknown citations must be rejected")


def test_deepseek_generator_recovers_bracketed_citation_from_answer(monkeypatch):
    class BracketedCitationClient(FakeOpenAI):
        content = '{"supported": true, "answer": "有依据的回答 [doc:0]", "citation_ids": []}'

    monkeypatch.setattr(openai, "OpenAI", BracketedCitationClient)
    generator = DeepSeekGenerator(
        api_key="secret-not-sent",
        model="deepseek-flash",
        base_url="https://example.invalid",
    )
    result = generator.generate("问题", [_hit()])
    assert result.citation_ids == ("doc:0",)


def test_deepseek_generator_recovers_missing_citation_by_text_overlap(monkeypatch):
    class MissingCitationClient(FakeOpenAI):
        content = '{"supported": true, "answer": "项目经历包括岩石图像识别。", "citation_ids": []}'

    monkeypatch.setattr(openai, "OpenAI", MissingCitationClient)
    generator = DeepSeekGenerator(
        api_key="secret-not-sent",
        model="deepseek-flash",
        base_url="https://example.invalid",
    )
    result = generator.generate("问题", [_hit()])
    assert result.citation_ids == ("doc:0",)


def test_parse_json_accepts_leading_explanation():
    payload = DeepSeekGenerator._parse_json('结果如下：\n{"supported": false, "answer": "", "citation_ids": []}')
    assert payload["supported"] is False


def test_deepseek_planner_keeps_original_query_and_respects_requested_mode(monkeypatch):
    class PlannerClient(FakeOpenAI):
        content = '{"answer_mode":"fact","queries":["项目经历 技术工具","研究实践"]}'

    monkeypatch.setattr(openai, "OpenAI", PlannerClient)
    generator = DeepSeekGenerator(
        api_key="secret-not-sent",
        model="deepseek-flash",
        base_url="https://example.invalid",
    )
    plan = generator.plan("我以前做过什么？", requested_mode="summary")
    assert plan.answer_mode == "summary"
    assert plan.queries == ("我以前做过什么？", "项目经历 技术工具", "研究实践")
