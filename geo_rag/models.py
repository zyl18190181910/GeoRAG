from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_id: str
    text: str
    order: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            source_id=str(value["source_id"]),
            text=str(value["text"]),
            order=int(value["order"]),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SearchHit:
    chunk: DocumentChunk
    score: float


class IngestTextRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def clean_source_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_id cannot be blank")
        return cleaned


class IngestResponse(BaseModel):
    source_id: str
    chunks_added: int
    total_chunks: int


class SourceInfo(BaseModel):
    source_id: str
    chunks: int
    enabled: bool
    filename: str | None = None
    updated_at: str | None = None


class SourceToggleRequest(BaseModel):
    enabled: bool


class SourceActionResponse(BaseModel):
    source_id: str | None = None
    affected_chunks: int
    active_sources: int
    active_chunks: int


AnswerMode = Literal["auto", "fact", "summary", "analysis"]


class ConversationTurn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=5000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=10)
    answer_mode: AnswerMode = "auto"
    history: list[ConversationTurn] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question cannot be blank")
        return cleaned


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    score: float
    excerpt: str
    filename: str | None = None
    page: int | None = None
    section: str | None = None


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    mode: str = "extractive"
    answer_mode: str = "fact"
    rewritten_queries: list[str] = Field(default_factory=list)
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    sources: int
    chunks: int
    active_sources: int = 0
    active_chunks: int = 0
    semantic_enabled: bool = False
    reranker_enabled: bool = False
    generation_enabled: bool = False
    embedding_model: str | None = None
    reranker_model: str | None = None
    generation_model: str | None = None

