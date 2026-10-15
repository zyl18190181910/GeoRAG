from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class LoadedSection:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def safe_source_id(filename: str) -> str:
    stem = Path(filename).stem.strip()
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", stem).strip("-")
    return cleaned[:100] or "document"


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("text file encoding must be UTF-8 or GB18030")


def _markdown_sections(text: str) -> list[LoadedSection]:
    sections: list[LoadedSection] = []
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        value = "\n".join(lines).strip()
        if value:
            metadata = {"section": heading} if heading else {}
            sections.append(LoadedSection(value, metadata))
        lines = []

    for line in text.splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if match:
            flush()
            heading = match.group(1).strip()
            lines.append(line)
        else:
            lines.append(line)
    flush()
    return sections


def extract_sections_from_bytes(filename: str, content: bytes) -> list[LoadedSection]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type: {suffix or '(none)'}")
    if suffix == ".txt":
        text = _decode_text(content)
        sections = [LoadedSection(text)]
    elif suffix == ".md":
        text = _decode_text(content)
        sections = _markdown_sections(text)
    else:
        reader = PdfReader(io.BytesIO(content))
        sections = [
            LoadedSection(text, {"page": number})
            for number, page in enumerate(reader.pages, start=1)
            if (text := (page.extract_text() or "").strip())
        ]
    if not sections or not any(section.text.strip() for section in sections):
        raise ValueError("document does not contain extractable text")
    return sections


def extract_text_from_bytes(filename: str, content: bytes) -> str:
    return "\n".join(section.text for section in extract_sections_from_bytes(filename, content))

