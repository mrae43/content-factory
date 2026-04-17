import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.workers.agents import RedTeamAgent, AgentActionStatus


def _make_agent():
    agent = RedTeamAgent.__new__(RedTeamAgent)
    agent.model_name = "gemini-1.5-pro"
    agent.temperature = 0.1
    agent.llm = MagicMock()
    return agent


@pytest.mark.agent
async def test_returns_success_when_all_supported(
    mock_vector_store,
    job_id,
    red_team_verdict_supported,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 3.2% in 2024.",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    with chain_mock(red_team_verdict_supported):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["verdict"] == "SUPPORTED"
    assert len(result.payload["claims"]) == 1
    assert result.payload["claims"][0]["verdict"] == "SUPPORTED"


@pytest.mark.agent
async def test_returns_revision_needed_when_unsupported(
    mock_vector_store,
    job_id,
    red_team_verdict_unsupported,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 15% last year. New payment system launched.",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    with chain_mock(red_team_verdict_unsupported):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.REVISION_NEEDED
    assert result.payload["verdict"] == "UNSUPPORTED"
    avg_conf = sum(c.confidence for c in red_team_verdict_unsupported.claims) / len(
        red_team_verdict_unsupported.claims
    )
    assert result.confidence_score == avg_conf
    assert avg_conf < 0.8


@pytest.mark.agent
async def test_returns_revision_needed_when_contested(
    mock_vector_store,
    job_id,
    red_team_verdict_contested,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS controls 40% of global trade.",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    with chain_mock(red_team_verdict_contested):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.REVISION_NEEDED
    assert result.payload["verdict"] == "UNSUPPORTED"


@pytest.mark.agent
async def test_returns_escalate_when_no_sources(mock_vector_store, job_id):
    mock_vector_store.semantic_search.return_value = []
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 15% last year.",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ESCALATE
    assert "No research sources" in result.reasoning
    assert result.confidence_score == 0.0


@pytest.mark.agent
async def test_returns_escalate_on_llm_parse_error(
    mock_vector_store,
    job_id,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 3.2% in 2024.",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    with patch(
        "langchain_core.runnables.base.RunnableSequence.ainvoke",
        new_callable=AsyncMock,
        side_effect=Exception("Parse error"),
    ):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.ESCALATE
    assert "LLM output parsing failed" in result.reasoning
