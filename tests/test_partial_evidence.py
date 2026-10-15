from types import SimpleNamespace

from geo_rag.generation import DeepSeekGenerator
from geo_rag.models import DocumentChunk, SearchHit


def test_partial_evidence_returns_grounded_answer_without_exposing_label():
    generator = object.__new__(DeepSeekGenerator)
    generator.model = "test-model"
    request = {}
    payload = (
        '{"supported": true, "coverage": "partial", '
        '"answer": "根据当前资料，只能确认项目采用了有限元法 [报告:0]。资料未覆盖完整分类。", '
        '"citation_ids": ["报告:0"]}'
    )
    def completion(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
        )

    generator._completion = completion
    hit = SearchHit(
        chunk=DocumentChunk(
            chunk_id="报告:0",
            source_id="报告",
            text="项目采用有限元法开展地球物理响应数值模拟。",
            order=0,
        ),
        score=0.9,
    )

    result = generator.generate(
        "地球物理数值模拟一般有哪些方法？",
        [hit],
        answer_mode="summary",
    )

    assert result.supported is True
    assert result.coverage == "partial"
    assert result.citation_ids == ("报告:0",)
    assert "partial" not in result.answer
    assert "资料未覆盖" in result.answer
    system_prompt = request["messages"][0]["content"]
    assert "通常有哪些" in system_prompt
    assert "不得把网格策略、并行技术" in system_prompt
    assert "即使是否定句" in system_prompt
    assert "就不得选择 none" in system_prompt
    assert "完整片段编号" in system_prompt
    assert "必须使用 Markdown 表格" in request["messages"][1]["content"]


def test_none_coverage_cannot_become_a_supported_answer():
    generator = object.__new__(DeepSeekGenerator)
    generator.model = "test-model"
    payload = (
        '{"supported": true, "coverage": "none", "answer": "", '
        '"citation_ids": []}'
    )
    generator._completion = lambda **_: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )

    result = generator.generate("资料完全没有涉及的问题", [])

    assert result.supported is False
    assert result.coverage == "none"
    assert result.citation_ids == ()
