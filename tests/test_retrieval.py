from geo_rag.models import DocumentChunk
from geo_rag.retrieval import SparseTfidfRetriever, tokenize


def _chunk(source: str, text: str, order: int = 0) -> DocumentChunk:
    return DocumentChunk(f"{source}:{order}", source, text, order)


def test_tokenize_keeps_latin_technical_terms():
    tokens = tokenize("Top-1 与 Macro-F1")
    assert "top-1" in tokens
    assert "macro-f1" in tokens


def test_tokenize_generates_chinese_bigrams():
    tokens = tokenize("岩石识别")
    assert "岩石" in tokens
    assert "识别" in tokens


def test_empty_retriever_returns_no_hits():
    retriever = SparseTfidfRetriever()
    assert retriever.search("岩石") == []


def test_retriever_ranks_relevant_chunk_first():
    retriever = SparseTfidfRetriever()
    retriever.fit([_chunk("rock", "低置信度岩石样本应进行未知类别拒识。"), _chunk("gravity", "重力异常可以用于地下结构研究。")])
    hits = retriever.search("低置信度样本如何拒识", top_k=2)
    assert hits[0].chunk.source_id == "rock"


def test_retriever_respects_min_score():
    retriever = SparseTfidfRetriever()
    retriever.fit([_chunk("rock", "岩石图像分类与识别。")])
    assert retriever.search("财务报表", min_score=0.5) == []


def test_technical_term_match_gets_anchor_bonus():
    retriever = SparseTfidfRetriever()
    retriever.fit(
        [
            _chunk("metrics", "Macro-F1 能反映少数类别的识别效果。"),
            _chunk("gravity", "反演问题需要物理约束。"),
        ]
    )
    hits = retriever.search("Macro-F1 适合观察什么问题？", top_k=2, min_score=0.08)
    assert hits[0].chunk.source_id == "metrics"


def test_cross_domain_question_is_rejected_at_default_threshold():
    retriever = SparseTfidfRetriever()
    retriever.fit(
        [
            _chunk("testing", "接口测试需要检查状态码、字段结构和错误信息。"),
            _chunk("workflow", "缺陷修复后应该进行回归测试并保存报告。"),
        ]
    )
    assert retriever.search("红烧肉应该放多少糖？", min_score=0.08) == []


def test_question_stopwords_do_not_become_features():
    tokens = tokenize("这个问题应该如何处理？")
    assert "这个" not in tokens
    assert "问题" not in tokens
    assert "应该" not in tokens
    assert "如何" not in tokens

