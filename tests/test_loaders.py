import pytest

from geo_rag import loaders
from geo_rag.loaders import extract_sections_from_bytes, extract_text_from_bytes, safe_source_id


def test_safe_source_id_removes_unsafe_characters():
    assert safe_source_id("岩石 识别(终稿).md") == "岩石-识别-终稿"


def test_load_utf8_markdown():
    assert "岩石" in extract_text_from_bytes("guide.md", "# 岩石".encode("utf-8"))


def test_markdown_preserves_heading_as_section_metadata():
    sections = extract_sections_from_bytes(
        "guide.md",
        "# 数据准备\n按区域划分。\n## 模型评测\n检查混淆矩阵。".encode("utf-8"),
    )
    assert [section.metadata["section"] for section in sections] == ["数据准备", "模型评测"]
    assert "检查混淆矩阵" in sections[1].text


def test_load_gb18030_text():
    assert "重力" in extract_text_from_bytes("guide.txt", "重力异常".encode("gb18030"))


def test_reject_unsupported_file_type():
    with pytest.raises(ValueError, match="unsupported"):
        extract_text_from_bytes("image.png", b"not an image")


def test_reject_empty_text_file():
    with pytest.raises(ValueError, match="extractable"):
        extract_text_from_bytes("empty.txt", b"   ")


def test_pdf_preserves_page_number(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        pages = [FakePage("第一页内容"), FakePage(""), FakePage("第三页内容")]

    monkeypatch.setattr(loaders, "PdfReader", lambda stream: FakeReader())
    sections = extract_sections_from_bytes("report.pdf", b"fake")
    assert [section.metadata["page"] for section in sections] == [1, 3]

