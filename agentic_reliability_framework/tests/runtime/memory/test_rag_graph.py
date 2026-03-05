"""Tests for RAGGraphMemory."""
import pytest
from datetime import datetime, timezone
from agentic_reliability_framework.core.models.event import ReliabilityEvent, EventSeverity
from agentic_reliability_framework.runtime.memory.faiss_index import ProductionFAISSIndex
from agentic_reliability_framework.runtime.memory.rag_graph import RAGGraphMemory

@pytest.fixture
def rag_graph():
    faiss = ProductionFAISSIndex()
    return RAGGraphMemory(faiss)

@pytest.fixture
def sample_event():
    return ReliabilityEvent(
        component="test",
        latency_p99=150.0,
        error_rate=0.05,
        throughput=1000,
        cpu_util=0.6,
        memory_util=0.7,
        timestamp=datetime.now(timezone.utc),
        severity=EventSeverity.WARNING   # changed from MEDIUM to WARNING
    )

def test_store_and_find(rag_graph, sample_event):
    analysis = {"incident_summary": {"anomaly_confidence": 0.8}}
    inc_id = rag_graph.store_incident(sample_event, analysis)
    assert inc_id.startswith("inc_")
    similar = rag_graph.find_similar(sample_event, analysis, k=1)
    assert len(similar) == 1
    assert similar[0].incident_id == inc_id
    assert similar[0].metadata.get("similarity_score", 0) > 0

def test_historical_effectiveness(rag_graph, sample_event):
    inc_id = rag_graph.store_incident(sample_event, {})
    rag_graph.store_outcome(inc_id, ["restart_container"], success=True, resolution_time_minutes=2.5)
    stats = rag_graph.get_historical_effectiveness("restart_container", component_filter="test")
    assert stats["total_uses"] == 1
    assert stats["successful_uses"] == 1
    assert stats["success_rate"] == 1.0

def test_store_outcome(rag_graph, sample_event):
    inc_id = rag_graph.store_incident(sample_event, {})
    outcome_id = rag_graph.store_outcome(inc_id, ["restart"], success=True, resolution_time_minutes=5.0, lessons_learned=["test lesson"])
    assert outcome_id.startswith("out_")
    assert len(rag_graph.outcome_nodes) == 1
    outcome = list(rag_graph.outcome_nodes.values())[0]
    assert outcome.success is True
    assert outcome.actions_taken == ["restart"]
