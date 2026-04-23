import pytest
from unittest.mock import MagicMock

from app.workers.optimizer import ScriptOptimizerAgent, format_failed_claims
from app.workers.agents import AgentActionStatus


SAMPLE_FAILED_CLAIMS = [
    {
        "claim_text": "BRICS GDP grew 15% last year.",
        "verdict": "UNSUPPORTED",
        "confidence": 0.3,
        "evidence_text": "No source confirms 15% GDP growth.",
    }
]

SAMPLE_REFINED_CONTEXT = "BRICS GDP grew 3.2% in 2024 according to IMF data."


def _make_agent():
    agent = ScriptOptimizerAgent.__new__(ScriptOptimizerAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.3
    agent.llm = MagicMock()
    return agent


@pytest.mark.agent
async def test_returns_success_with_patched_script(
    optimizer_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 15% last year. This is a big deal.",
        "failed_claims": SAMPLE_FAILED_CLAIMS,
        "refined_context": SAMPLE_REFINED_CONTEXT,
    }

    with chain_mock(optimizer_schema_output):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert isinstance(result.payload["script_content"], str)
    assert isinstance(result.payload["storyboard"], list)
    assert isinstance(result.payload["patch_summary"], str)
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.metadata["agent"] == "optimizer"


@pytest.mark.agent
async def test_returns_error_when_no_script_content():
    agent = _make_agent()
    context = {
        "script_content": "",
        "failed_claims": SAMPLE_FAILED_CLAIMS,
        "refined_context": SAMPLE_REFINED_CONTEXT,
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No script content" in result.reasoning
    assert result.confidence_score == 0.0


@pytest.mark.agent
async def test_returns_error_when_no_failed_claims():
    agent = _make_agent()
    context = {
        "script_content": "Some script content.",
        "failed_claims": [],
        "refined_context": SAMPLE_REFINED_CONTEXT,
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No failed claims" in result.reasoning
    assert result.confidence_score == 0.0


@pytest.mark.agent
async def test_chain_receives_formatted_claims(
    optimizer_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "BRICS GDP grew 15% last year.",
        "failed_claims": SAMPLE_FAILED_CLAIMS,
        "refined_context": SAMPLE_REFINED_CONTEXT,
    }

    with chain_mock(optimizer_schema_output) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    call_args = mock_ainvoke.call_args
    invoked_input = call_args[0][0]
    assert "15%" in invoked_input["failed_claims"]
    assert "UNSUPPORTED" in invoked_input["failed_claims"]
    assert invoked_input["original_script"] == "BRICS GDP grew 15% last year."
    assert invoked_input["refined_context"] == SAMPLE_REFINED_CONTEXT


def test_format_failed_claims_single():
    result = format_failed_claims(SAMPLE_FAILED_CLAIMS)
    assert "Claim 1: BRICS GDP grew 15% last year." in result
    assert "Verdict: UNSUPPORTED" in result
    assert "Confidence: 0.30" in result
    assert "Evidence: No source confirms 15% GDP growth." in result


def test_format_failed_claims_multiple():
    claims = [
        {
            "claim_text": "GDP grew 15%.",
            "verdict": "UNSUPPORTED",
            "confidence": 0.3,
            "evidence_text": "No evidence.",
        },
        {
            "claim_text": "Payment system launched.",
            "verdict": "CONTESTED",
            "confidence": 0.5,
            "evidence_text": "Sources disagree.",
        },
    ]
    result = format_failed_claims(claims)
    assert "Claim 1:" in result
    assert "Claim 2:" in result
    assert "UNSUPPORTED" in result
    assert "CONTESTED" in result


def test_format_failed_claims_missing_evidence():
    claims = [{"claim_text": "Some claim.", "verdict": "UNSUPPORTED"}]
    result = format_failed_claims(claims)
    assert "Evidence: N/A" in result
