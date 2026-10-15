from __future__ import annotations

import re
from typing import Iterable

from .models import DocumentChunk


_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", re.UNICODE)
_CJK_LAYOUT_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")


def normalize_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        normalized = re.sub(r"[\t\u3000 ]+", " ", line).strip()
        # Chinese journal PDFs sometimes place a layout space between every glyph.
        # Remove CJK-to-CJK spaces only, so ordinary Latin word boundaries remain.
        normalized = _CJK_LAYOUT_SPACE_RE.sub("", normalized)
        lines.append(normalized)
    return "\n".join(line for line in lines if line).strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(normalized) if match.group(0).strip()]


def _tail_for_overlap(sentences: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        tail.insert(0, sentence)
        total += len(sentence)
        if total >= overlap:
            break
    return tail


def chunk_text(
    source_id: str,
    text: str,
    *,
    chunk_size: int = 480,
    overlap: int = 80,
    metadata: dict | None = None,
) -> list[DocumentChunk]:
    if chunk_size < 80:
        raise ValueError("chunk_size must be at least 80 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    sentences = split_sentences(text)
    if not sentences:
        return []

    groups: list[list[str]] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                groups.append(current)
                current = []
                current_length = 0
            step = chunk_size - overlap
            groups.extend([[sentence[i : i + chunk_size]] for i in range(0, len(sentence), step)])
            continue

        if current and current_length + len(sentence) > chunk_size:
            groups.append(current)
            current = _tail_for_overlap(current, overlap)
            current_length = sum(len(item) for item in current)
            while current and current_length + len(sentence) > chunk_size:
                removed = current.pop(0)
                current_length -= len(removed)
        current.append(sentence)
        current_length += len(sentence)

    if current:
        groups.append(current)

    common_metadata = dict(metadata or {})
    chunks: list[DocumentChunk] = []
    for order, group in enumerate(groups):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{source_id}:{order}",
                source_id=source_id,
                text="".join(group),
                order=order,
                metadata=common_metadata,
            )
        )
    return chunks

