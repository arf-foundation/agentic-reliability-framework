"""Tests for ProductionFAISSIndex."""
import numpy as np
import pytest
from agentic_reliability_framework.runtime.memory.faiss_index import ProductionFAISSIndex, create_faiss_index

def test_create_index():
    idx = create_faiss_index(dim=128)
    assert idx.index.d == 128

def test_add_and_search():
    idx = ProductionFAISSIndex(dim=128)
    vec = np.random.randn(128).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    added_id = idx.add(vec)
    assert added_id == 0
    distances, indices = idx.search(vec, k=1)
    assert indices[0] == 0
    assert distances[0] < 1e-3
