from fastapi.testclient import TestClient

from geo_rag import api as api_module
from geo_rag.api import create_app


def test_home_page_is_chinese_and_links_to_docs(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "地学资料智能问答" in response.text
    assert 'href="/docs"' in response.text
    assert "renderAnswer" in response.text
    assert "prepareAnswerDisplay" in response.text
    assert "displayIndex:index + 1" in response.text
    assert "firstNumberBySource" in response.text
    assert "片段 ${item.chunk_id}" not in response.text


def test_health_is_ok_for_empty_store(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "sources": 0,
        "chunks": 0,
        "active_sources": 0,
        "active_chunks": 0,
        "semantic_enabled": False,
        "reranker_enabled": False,
        "generation_enabled": False,
        "embedding_model": None,
        "reranker_model": None,
        "generation_model": None,
    }


def test_sources_starts_empty(client: TestClient):
    assert client.get("/sources").json() == []


def test_ingest_text_and_list_source(client: TestClient):
    response = client.post("/ingest/text", json={"source_id": "rock", "text": "岩石分类需要独立测试集。"})
    assert response.status_code == 200
    assert response.json()["chunks_added"] == 1
    assert client.get("/sources").json() == ["rock"]


def test_source_can_be_disabled_reindexed_and_deleted(client: TestClient):
    client.post("/ingest/text", json={"source_id": "rock", "text": "岩石识别使用独立测试集。"})
    details = client.get("/sources/details").json()
    assert details[0]["source_id"] == "rock"
    assert details[0]["enabled"] is True

    disabled = client.patch("/sources/rock", json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json()["active_sources"] == 0
    assert client.post("/ask", json={"question": "岩石识别怎样测试？"}).json()["refused"] is True

    assert client.post("/reindex").json()["active_chunks"] == 0
    assert client.patch("/sources/rock", json={"enabled": True}).json()["active_sources"] == 1
    deleted = client.delete("/sources/rock")
    assert deleted.json()["affected_chunks"] == 1
    assert client.get("/sources").json() == []


def test_missing_source_actions_return_404(client: TestClient):
    assert client.patch("/sources/missing", json={"enabled": False}).status_code == 404
    assert client.delete("/sources/missing").status_code == 404


def test_showcase_report_endpoints_without_report_or_key(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    local_client = TestClient(create_app(settings))
    assert local_client.get("/evaluation/showcase").json() == {"status": "not_run"}
    response = local_client.post("/evaluation/showcase/run")
    assert response.status_code == 400
    assert "DeepSeek API key" in response.json()["detail"]


def test_ablation_report_endpoint_before_first_run(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    local_client = TestClient(create_app(settings))
    assert local_client.get("/evaluation/ablation").json() == {"status": "not_run"}


def test_ablation_run_returns_clear_error(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(api_module, "ROOT", tmp_path)

    def fail(active_settings, root):
        raise ValueError("evaluation data missing")

    monkeypatch.setattr(api_module, "run_ablation_evaluation", fail)
    response = TestClient(create_app(settings)).post("/evaluation/ablation/run")
    assert response.status_code == 400
    assert response.json()["detail"] == "evaluation data missing"


def test_ask_returns_citations(client: TestClient):
    client.post("/ingest/text", json={"source_id": "rock", "text": "低置信度结果应该执行未知类别拒识。"})
    response = client.post("/ask", json={"question": "低置信度结果怎么处理？", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is False
    assert payload["citations"][0]["source_id"] == "rock"
    assert payload["answer_mode"] == "fact"
    assert payload["rewritten_queries"] == ["低置信度结果怎么处理？"]


def test_ask_accepts_answer_mode_and_history(client: TestClient):
    client.post("/ingest/text", json={"source_id": "rock", "text": "项目经历包括岩石图像识别。"})
    response = client.post(
        "/ask",
        json={
            "question": "把它概括一下",
            "answer_mode": "summary",
            "history": [{"question": "项目经历是什么？", "answer": "岩石图像识别。"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["answer_mode"] == "summary"


def test_ask_rejects_blank_question(client: TestClient):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_upload_markdown_file(client: TestClient):
    response = client.post(
        "/ingest/files",
        files=[("files", ("重力基础.md", "再正演用于物理一致性验证。".encode("utf-8"), "text/markdown"))],
    )
    assert response.status_code == 200
    assert response.json()[0]["source_id"] == "重力基础"


def test_uploaded_markdown_citation_includes_section(client: TestClient):
    client.post(
        "/ingest/files",
        files=[
            (
                "files",
                ("评测指南.md", "# 模型评测\n混淆矩阵用于定位类别误判。".encode("utf-8"), "text/markdown"),
            )
        ],
    )
    payload = client.post("/ask", json={"question": "什么可以定位类别误判？"}).json()
    assert payload["citations"][0]["filename"] == "评测指南.md"
    assert payload["citations"][0]["section"] == "模型评测"


def test_upload_rejects_unsupported_file(client: TestClient):
    response = client.post(
        "/ingest/files",
        files=[("files", ("image.png", b"not-an-image", "image/png"))],
    )
    assert response.status_code == 400

