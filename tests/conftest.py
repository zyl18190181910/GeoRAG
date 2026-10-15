from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geo_rag.api import create_app
from geo_rag.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(storage_path=tmp_path / "knowledge.json", min_score=0.08, chunk_size=180, chunk_overlap=30)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))

