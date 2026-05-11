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
async def test_returns_success_with_script(
    copywriter_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "",
        "refined_context": "BRICS collective GDP grew 3.2% in 2024. A new payment system was announced in Q2.",
    }

    with chain_mock(copywriter_schema_output):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert isinstance(result.payload["script_content"], str)
    assert len(result.payload["script_content"]) > 0


@pytest.mark.agent
async def test_returns_success_with_feedback_from_revision(
    copywriter_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "Previous script too vague",
        "refined_context": "BRICS collective GDP grew 3.2% in 2024. A new payment system was announced in Q2.",
    }

    with chain_mock(copywriter_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert invoked_input["feedback"] == "Previous script too vague"


@pytest.mark.agent
async def test_returns_error_when_no_refined_context():
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "",
        "refined_context": "",
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No refined research context" in result.reasoning
    assert result.confidence_score == 0.0
