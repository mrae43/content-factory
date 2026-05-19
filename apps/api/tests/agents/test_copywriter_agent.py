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
        "evidence_sections": "## Retrieved Evidence\n\nChunk 1 (similarity: 0.92, source: WEB_SEARCH, relevance: HIGH):\nIMF report confirms 3.2% growth.\n",
        "story_directives": {
            "target_audience": "Investors",
            "tone": "analytical",
            "angle": "economic implications for emerging markets",
        },
    }

    with chain_mock(copywriter_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert isinstance(result.payload["script_content"], str)
    assert len(result.payload["script_content"]) > 0
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert "Target Audience: Investors" in invoked_input["story_directives"]
    assert "Tone: analytical" in invoked_input["story_directives"]
    assert "Angle: economic implications" in invoked_input["story_directives"]
    assert "IMF report confirms 3.2% growth" in invoked_input["evidence_sections"]


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
        "evidence_sections": "## Retrieved Evidence\n\nChunk 1 (similarity: 0.92, source: WEB_SEARCH, relevance: HIGH):\nIMF report confirms 3.2% growth.\n",
        "story_directives": {
            "target_audience": "General",
            "tone": "conversational",
            "angle": "",
        },
    }

    with chain_mock(copywriter_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert invoked_input["feedback"] == "Previous script too vague"
    assert "Target Audience: General" in invoked_input["story_directives"]
    assert "Tone: conversational" in invoked_input["story_directives"]
    assert "IMF report confirms 3.2% growth" in invoked_input["evidence_sections"]


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


@pytest.mark.agent
async def test_passes_evidence_sections_to_prompt(
    copywriter_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "",
        "refined_context": "BRICS collective GDP grew 3.2% in 2024.",
        "evidence_sections": "## Retrieved Evidence\n\nChunk 1 (similarity: 0.95): IMF confirms 3.2% growth.",
        "story_directives": {
            "target_audience": "General",
            "tone": "neutral",
            "angle": "",
        },
    }

    with chain_mock(copywriter_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert (
        "Chunk 1 (similarity: 0.95): IMF confirms 3.2% growth."
        in invoked_input["evidence_sections"]
    )


@pytest.mark.agent
async def test_handles_empty_evidence_sections(
    copywriter_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "",
        "refined_context": "BRICS collective GDP grew 3.2% in 2024.",
        "evidence_sections": "",
        "story_directives": {
            "target_audience": "General",
            "tone": "neutral",
            "angle": "",
        },
    }

    with chain_mock(copywriter_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert "No additional evidence was retrieved" in invoked_input["evidence_sections"]
