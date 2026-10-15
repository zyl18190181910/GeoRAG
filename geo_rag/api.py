from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .ablation import load_ablation_report, run_ablation_evaluation
from .config import Settings
from .loaders import extract_sections_from_bytes, safe_source_id
from .models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    SourceActionResponse,
    SourceInfo,
    SourceToggleRequest,
)
from .service import RagService
from .showcase import load_showcase_report, run_showcase_evaluation
from .web_ui import HOME_PAGE


ROOT = Path(__file__).resolve().parents[1]


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or Settings.from_env()
    service = RagService(active_settings)
    app = FastAPI(
        title="地学知识库问答与自动化评测系统",
        version="0.1.0",
        description="提供资料导入、检索、带出处回答和无答案拒答能力。",
    )
    app.state.rag_service = service

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> HTMLResponse:
        return HTMLResponse(HOME_PAGE)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            sources=len(service.source_ids),
            chunks=len(service.chunks),
            active_sources=len(service.active_source_ids),
            active_chunks=len(service.active_chunks),
            semantic_enabled=service.semantic_enabled,
            reranker_enabled=service.reranker_enabled,
            generation_enabled=service.generation_enabled,
            embedding_model=active_settings.embedding_model if service.semantic_enabled else None,
            reranker_model=active_settings.reranker_model if service.reranker_enabled else None,
            generation_model=active_settings.deepseek_model if service.generation_enabled else None,
        )

    @app.get("/sources", response_model=list[str])
    def sources() -> list[str]:
        return service.source_ids

    @app.get("/sources/details", response_model=list[SourceInfo])
    def source_details() -> list[SourceInfo]:
        return service.source_details()

    @app.patch("/sources/{source_id}", response_model=SourceActionResponse)
    def toggle_source(source_id: str, request: SourceToggleRequest) -> SourceActionResponse:
        try:
            affected = service.set_source_enabled(source_id, request.enabled)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="source not found") from exc
        return SourceActionResponse(
            source_id=source_id,
            affected_chunks=affected,
            active_sources=len(service.active_source_ids),
            active_chunks=len(service.active_chunks),
        )

    @app.delete("/sources/{source_id}", response_model=SourceActionResponse)
    def delete_source(source_id: str) -> SourceActionResponse:
        try:
            affected = service.delete_source(source_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="source not found") from exc
        return SourceActionResponse(
            source_id=source_id,
            affected_chunks=affected,
            active_sources=len(service.active_source_ids),
            active_chunks=len(service.active_chunks),
        )

    @app.post("/reindex", response_model=SourceActionResponse)
    def reindex() -> SourceActionResponse:
        affected = service.reindex()
        return SourceActionResponse(
            affected_chunks=affected,
            active_sources=len(service.active_source_ids),
            active_chunks=len(service.active_chunks),
        )

    @app.get("/evaluation/showcase")
    def showcase_report() -> dict:
        return load_showcase_report(ROOT)

    @app.post("/evaluation/showcase/run")
    def run_showcase() -> dict:
        try:
            return run_showcase_evaluation(active_settings, ROOT)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/evaluation/ablation")
    def ablation_report() -> dict:
        return load_ablation_report(ROOT)

    @app.post("/evaluation/ablation/run")
    def run_ablation() -> dict:
        try:
            return run_ablation_evaluation(active_settings, ROOT)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest/text", response_model=IngestResponse)
    def ingest_text(request: IngestTextRequest) -> IngestResponse:
        try:
            added = service.ingest(request.source_id, request.text, request.metadata)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(source_id=request.source_id, chunks_added=added, total_chunks=len(service.chunks))

    @app.post("/ingest/files", response_model=list[IngestResponse])
    async def ingest_files(files: list[UploadFile] = File(...)) -> list[IngestResponse]:
        results: list[IngestResponse] = []
        for upload in files:
            content = await upload.read()
            if len(content) > active_settings.max_file_bytes:
                raise HTTPException(status_code=413, detail=f"file too large: {upload.filename}")
            try:
                filename = upload.filename or "document"
                sections = extract_sections_from_bytes(filename, content)
                source_id = safe_source_id(filename)
                added = service.ingest_sections(
                    source_id,
                    [(section.text, section.metadata) for section in sections],
                    metadata={"filename": filename},
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            results.append(
                IngestResponse(source_id=source_id, chunks_added=added, total_chunks=len(service.chunks))
            )
        return results

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return service.ask(
            request.question,
            request.top_k,
            answer_mode=request.answer_mode,
            history=request.history,
        )

    return app


app = create_app()

