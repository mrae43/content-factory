import pytest
from unittest.mock import MagicMock

from app.workers.agents import ResearchAgent, AgentActionStatus


def _make_agent():
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.2
    agent.llm = MagicMock()
    return agent


@pytest.mark.agent
async def test_returns_success_with_chunks(
    mock_vector_store,
    job_id,
    research_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    with chain_mock(research_schema_output):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert len(result.payload["chunks"]) == 2
    mock_vector_store.ingest_chunks.assert_awaited_once_with(
        job_id=job_id,
        chunks=research_schema_output.chunks,
        scope="LOCAL",
    )
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.metadata["model"] == "gemini-2.5-flash"


@pytest.mark.agent
async def test_returns_error_when_no_search_results(
    mock_vector_store,
    job_id,
):
    mock_vector_store.semantic_search.return_value = []
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No context retrieved" in result.reasoning
    assert result.confidence_score == 0.0


@pytest.mark.agent
async def test_returns_error_when_missing_vector_store(job_id):
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "vector_store": None,
        "job_id": job_id,
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "Vector store or job_id not provided" in result.reasoning
    assert result.confidence_score == 0.0
