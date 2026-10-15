from pathlib import Path

import numpy as np
import sentence_transformers
import torch

from geo_rag.models import DocumentChunk, SearchHit
from geo_rag.reranking import CrossEncoderReranker


class FakeCrossEncoder:
    loads = 0

    def __init__(self, model_name, cache_folder, device, local_files_only=False):
        FakeCrossEncoder.loads += 1

    def predict(self, pairs, **kwargs):
        return np.array([-2.0, 3.0])


def test_cross_encoder_reranks_and_loads_once(monkeypatch, tmp_path: Path):
    FakeCrossEncoder.loads = 0
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reranker = CrossEncoderReranker("fake", tmp_path / "models")
    hits = [
        SearchHit(DocumentChunk("a:0", "a", "内容一", 0), 0.9),
        SearchHit(DocumentChunk("b:0", "b", "内容二", 0), 0.4),
    ]
    result = reranker.rerank("问题", hits, top_k=2)
    assert result[0].chunk.source_id == "b"
    reranker.rerank("问题", hits, top_k=1)
    assert FakeCrossEncoder.loads == 1
