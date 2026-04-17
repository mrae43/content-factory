import pytest
from unittest.mock import MagicMock

from app.workers.agents import CopywriterAgent, AgentActionStatus


def _make_agent():
    agent = CopywriterAgent.__new__(CopywriterAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.2
    agent.llm = MagicMock()
    return agent


@pytest.mark.agent
async def test_returns_success_with_script_and_storyboard(
    mock_vector_store,
    job_id,
    copywriter_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    with chain_mock(copywriter_schema_output):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert isinstance(result.payload["script_content"], str)
    assert isinstance(result.payload["storyboard"], list)
    assert len(result.payload["storyboard"]) == 2
    for scene in result.payload["storyboard"]:
        assert "visual_prompt" in scene
        assert "audio_cue" in scene


@pytest.mark.agent
async def test_returns_success_with_feedback_from_revision(
    mock_vector_store,
    job_id,
    copywriter_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "Previous script too vague",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    with chain_mock(copywriter_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert invoked_input["feedback"] == "Previous script too vague"


@pytest.mark.agent
async def test_returns_error_when_no_research_chunks(
    mock_vector_store,
    job_id,
):
    mock_vector_store.semantic_search.return_value = []
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No research chunks" in result.reasoning
    assert result.confidence_score == 0.0
