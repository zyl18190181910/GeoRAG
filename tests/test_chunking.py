import pytest

from geo_rag.chunking import chunk_text, normalize_text, split_sentences


def test_normalize_text_removes_blank_lines_and_extra_spaces():
    assert normalize_text("  第一行  \n\n 第二   行 ") == "第一行\n第二行"


def test_normalize_text_repairs_cjk_layout_spacing():
    assert normalize_text("多 头 注 意 力 机 制 Transformer model") == "多头注意力机制 Transformer model"


def test_split_sentences_supports_chinese_punctuation():
    assert split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]


def test_chunk_text_returns_empty_for_blank_text():
    assert chunk_text("source", "   \n") == []


def test_chunk_text_assigns_stable_ids_and_order():
    chunks = chunk_text("guide", "第一句。第二句。", chunk_size=80, overlap=10)
    assert chunks[0].chunk_id == "guide:0"
    assert chunks[0].order == 0


def test_chunk_text_splits_long_content():
    text = "。".join(["岩石图像识别需要验证" * 8 for _ in range(8)]) + "。"
    chunks = chunk_text("guide", text, chunk_size=100, overlap=20)
    assert len(chunks) > 2
    assert all(len(chunk.text) <= 140 for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("source", "有效文本。", chunk_size=100, overlap=100)

