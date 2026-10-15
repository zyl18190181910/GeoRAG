from pathlib import Path

import sentence_transformers
import torch

from geo_rag.models import DocumentChunk
from geo_rag.semantic import HybridRetriever, SemanticRetriever, SentenceTransformerEmbedder


class FakeArray:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeSentenceTransformer:
    loads = 0
    encodes = 0

    def __init__(self, model_name, cache_folder, device, local_files_only=False):
        FakeSentenceTransformer.loads += 1
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self.local_files_only = local_files_only

    def encode(self, texts, **kwargs):
        FakeSentenceTransformer.encodes += 1
        return FakeArray([[1.0, 0.0] for _ in texts])


def test_sentence_transformer_embedder_loads_once(monkeypatch, tmp_path: Path):
    FakeSentenceTransformer.loads = 0
    FakeSentenceTransformer.encodes = 0
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    embedder = SentenceTransformerEmbedder("fake-model", tmp_path / "models")
    assert embedder.encode(["一", "二"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert embedder.encode(["三"]) == [[1.0, 0.0]]
    assert FakeSentenceTransformer.loads == 1


def test_document_embeddings_are_cached(monkeypatch, tmp_path: Path):
    FakeSentenceTransformer.loads = 0
    FakeSentenceTransformer.encodes = 0
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    first = SentenceTransformerEmbedder("fake-model", tmp_path / "models")
    second = SentenceTransformerEmbedder("fake-model", tmp_path / "models")

    assert first.encode_documents(["甲", "乙"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert second.encode_documents(["甲", "乙"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert FakeSentenceTransformer.encodes == 1


class FailingEmbedder:
    def encode(self, texts):
        raise RuntimeError("model unavailable")


def test_hybrid_retriever_falls_back_when_semantic_model_fails():
    semantic = SemanticRetriever(FailingEmbedder())
    retriever = HybridRetriever(sparse_min_score=0.08, semantic_min_score=0.4, semantic=semantic)
    retriever.fit([DocumentChunk("rock:0", "rock", "岩石图像识别", 0)])
    assert retriever.semantic_ready is False
    assert "RuntimeError" in retriever.semantic_error
    assert retriever.search("岩石识别")[0].chunk.source_id == "rock"
