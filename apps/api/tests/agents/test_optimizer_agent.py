import pytest
from unittest.mock import MagicMock

from app.workers.optimizer import (
    ScriptOptimizerAgent,
    format_active_failures,
    format_optimization_history,
)
from app.workers.agents import AgentActionStatus


SAMPLE_ACTIVE_FAILURES = [
    {
        "uuid": "test-uuid-1",
        "text": "BRICS GDP grew 15% last year.",
        "latest_verdict": "UNSUPPORTED",
        "failure_reason": "No source confirms 15% GDP growth.",
    }
]

SAMPLE_OPTIMIZATION_HISTORY = [
    {
        "iteration": 1,
        "patches_applied": ["Replaced 15% with 3.2% based on IMF data"],
        "claims_snapshot": [
            {
                "uuid": "test-uuid-1",
                "text": "BRICS GDP grew 3.2% in 2024",
                "verdict": "UNSUPPORTED",
            }
        ],
    }
]

SAMPLE_REFINED_CONTEXT = "BRICS GDP grew 3.2% in 2024 according to IMF data."


def _make_agent():
    agent = ScriptOptimizerAgent.__new__(ScriptOptimizerAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.3
    agent.llm = MagicMock()
    agent.di_tools = {}
    return agent


@pytest.mark.agent
async def test_returns_success_with_patched_script(
    optimizer_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 15% last year. This is a big deal.",
        "active_failures": SAMPLE_ACTIVE_FAILURES,
        "optimization_history": SAMPLE_OPTIMIZATION_HISTORY,
        "refined_context": SAMPLE_REFINED_CONTEXT,
        "evidence_sections": "## Retrieved Evidence\n\nChunk 1 (similarity: 0.92, source: WEB_SEARCH, relevance: HIGH):\nBRICS GDP grew 3.2% in 2024 according to IMF.\n",
        "story_directives": {
            "target_audience": "Investors",
            "tone": "analytical",
            "angle": "economic risks",
        },
    }

    with chain_mock(optimizer_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert isinstance(result.payload["script_content"], str)
    assert isinstance(result.payload["patch_summary"], str)
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.metadata["agent"] == "optimizer"
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert "Target Audience: Investors" in invoked_input["story_directives"]
    assert "Tone: analytical" in invoked_input["story_directives"]
    assert "Angle: economic risks" in invoked_input["story_directives"]
    assert "BRICS GDP grew 3.2%" in invoked_input["retrieved_evidence"]


@pytest.mark.agent
async def test_returns_error_when_no_script_content():
    agent = _make_agent()
    context = {
        "script_content": "",
        "active_failures": SAMPLE_ACTIVE_FAILURES,
        "refined_context": SAMPLE_REFINED_CONTEXT,
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No script content" in result.reasoning
    assert result.confidence_score == 0.0


@pytest.mark.agent
async def test_returns_error_when_no_active_failures():
    agent = _make_agent()
    context = {
        "script_content": "Some script content.",
        "active_failures": [],
        "refined_context": SAMPLE_REFINED_CONTEXT,
    }

    result = await agent._execute(context)
    assert result.status == AgentActionStatus.ERROR
    assert "No active failures" in result.reasoning
    assert result.confidence_score == 0.0


@pytest.mark.agent
async def test_chain_receives_formatted_claims(
    optimizer_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 15% last year.",
        "active_failures": SAMPLE_ACTIVE_FAILURES,
        "optimization_history": SAMPLE_OPTIMIZATION_HISTORY,
        "refined_context": SAMPLE_REFINED_CONTEXT,
        "evidence_sections": "## Retrieved Evidence\n\nChunk 1 (similarity: 0.92): IMF data.",
    }

    with chain_mock(optimizer_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert "15%" in invoked_input["active_failures"]
    assert "UNSUPPORTED" in invoked_input["active_failures"]
    assert "Iteration 1" in invoked_input["optimization_history"]
    assert "Replaced 15%" in invoked_input["optimization_history"]
    assert invoked_input["original_script"] == "BRICS GDP grew 15% last year."
    assert invoked_input["refined_context"] == SAMPLE_REFINED_CONTEXT
    assert (
        "Chunk 1 (similarity: 0.92): IMF data." in invoked_input["retrieved_evidence"]
    )


@pytest.mark.agent
async def test_handles_empty_evidence_sections(
    optimizer_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 15% last year.",
        "active_failures": SAMPLE_ACTIVE_FAILURES,
        "optimization_history": SAMPLE_OPTIMIZATION_HISTORY,
        "refined_context": SAMPLE_REFINED_CONTEXT,
        "evidence_sections": "",
        "story_directives": {
            "target_audience": "General",
            "tone": "neutral",
            "angle": "",
        },
    }

    with chain_mock(optimizer_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert "No additional evidence was retrieved" in invoked_input["retrieved_evidence"]


def test_format_active_failures_single():
    result = format_active_failures(SAMPLE_ACTIVE_FAILURES)
    assert "Claim 1: BRICS GDP grew 15% last year." in result
    assert "Verdict: UNSUPPORTED" in result
    assert "Failure reason: No source confirms 15% GDP growth." in result


def test_format_active_failures_multiple():
    claims = [
        {
            "text": "GDP grew 15%.",
            "latest_verdict": "UNSUPPORTED",
            "failure_reason": "No evidence.",
        },
        {
            "text": "Payment system launched.",
            "latest_verdict": "CONTESTED",
            "failure_reason": "Sources disagree.",
        },
    ]
    result = format_active_failures(claims)
    assert "Claim 1:" in result
    assert "Claim 2:" in result
    assert "UNSUPPORTED" in result
    assert "CONTESTED" in result


def test_format_active_failures_missing_reason():
    claims = [{"text": "Some claim.", "latest_verdict": "UNSUPPORTED"}]
    result = format_active_failures(claims)
    assert "Failure reason: N/A" in result


def test_format_optimization_history_empty():
    result = format_optimization_history([])
    assert "No prior optimization history" in result


def test_format_optimization_history_with_data():
    history = [
        {
            "iteration": 1,
            "patches_applied": ["Replaced GDP metric"],
            "claims_snapshot": [{"text": "GDP grew 3.2%", "verdict": "SUPPORTED"}],
        }
    ]
    result = format_optimization_history(history)
    assert "Iteration 1" in result
    assert "Replaced GDP metric" in result
    assert "GDP grew 3.2%" in result
    assert "SUPPORTED" in result
