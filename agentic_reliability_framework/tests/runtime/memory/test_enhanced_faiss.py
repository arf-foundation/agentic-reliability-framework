"""Tests for EnhancedFAISSIndex."""
import numpy as np
import pytest
import asyncio
from agentic_reliability_framework.runtime.memory.faiss_index import ProductionFAISSIndex
from agentic_reliability_framework.runtime.memory.enhanced_faiss import EnhancedFAISSIndex
from agentic_reliability_framework.runtime.memory.constants import MemoryConstants

def test_search_async():
    base = ProductionFAISSIndex()
    enhanced = EnhancedFAISSIndex(base)
    vec = np.random.randn(MemoryConstants.VECTOR_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    base.add(vec)
    distances, indices = asyncio.run(enhanced.search_async(vec, k=1))
    assert indices[0] == 0
    assert distances[0] < 1e-3

def test_semantic_search_fallback():
    base = ProductionFAISSIndex()
    enhanced = EnhancedFAISSIndex(base)
    vec = np.random.randn(MemoryConstants.VECTOR_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    base.add(vec)
    base.texts.append("test text")
    results = enhanced.semantic_search("test query", k=1)
    assert len(results) == 1
    assert "similarity" in results[0]
    assert results[0]["text"] == "test text"
