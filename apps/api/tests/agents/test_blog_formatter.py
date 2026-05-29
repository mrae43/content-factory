import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from app.workers.agents import AgentActionStatus
from app.workers.formatters import (
    BlogFormatterAgent,
    BlogPlan,
    BlogOutlineSection,
    BlogFormatterOutput,
    BlogSection,
    SeoMeta,
)

_UUID_1 = str(uuid4())
_UUID_2 = str(uuid4())


def _make_agent():
    agent = BlogFormatterAgent.__new__(BlogFormatterAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.2
    agent.llm = MagicMock()
    return agent


def _blog_plan():
    return BlogPlan(
        proposed_title="The Rise of BRICS",
        proposed_subtitle="How emerging economies are reshaping finance",
        sections=[
            BlogOutlineSection(
                heading="Introduction",
                key_points=["GDP growth", "New payment system"],
                sources_to_cite=[_UUID_1],
            ),
            BlogOutlineSection(
                heading="Economic Impact",
                key_points=["3.2% GDP growth", "Trade partnerships"],
                sources_to_cite=[_UUID_2],
            ),
        ],
        target_tags=["BRICS", "economy", "finance"],
        cta_direction="Subscribe for more analysis",
    )


def _blog_output():
    return BlogFormatterOutput(
        title="The Rise of BRICS",
        subtitle="How emerging economies are reshaping finance",
        sections=[
            BlogSection(
                heading="Introduction",
                body="BRICS nations are driving a shift in global finance.",
                key_takeaway="BRICS GDP grew 3.2% in 2024.",
                word_count=10,
                sources_used=[_UUID_1],
            ),
            BlogSection(
                heading="Economic Impact",
                body="The collective GDP growth of 3.2% signals resilience.",
                key_takeaway="Growth surpasses expectations.",
                word_count=9,
                sources_used=[_UUID_2],
            ),
        ],
        seo_meta=SeoMeta(
            meta_title="The Rise of BRICS in 2025",
            meta_description="Explore how BRICS is reshaping global finance.",
            keywords=["BRICS", "economy", "GDP"],
        ),
        tags=["BRICS", "economy", "finance"],
        call_to_action="Subscribe for more analysis",
    )


def _blog_context():
    return {
        "script_content": "Did you know BRICS is reshaping global finance?",
        "refined_context": "BRICS GDP grew 3.2% in 2024.",
        "verified_claims": [
            {
                "claim_text": "BRICS GDP grew 3.2%",
                "verdict": "SUPPORTED",
                "evidence_text": "IMF data confirms.",
            }
        ],
    }


@pytest.mark.agent
async def test_blog_happy_path_two_phase(multi_chain_mock):
    agent = _make_agent()
    plan = _blog_plan()
    output = _blog_output()
    context = _blog_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["_format"] == "blog"
    assert result.payload["_version"] == 1
    assert result.payload["title"] == "The Rise of BRICS"
    assert len(result.payload["sections"]) == 2
    assert result.confidence_score == 0.9
    assert result.metadata["agent"] == "blog_formatter"
    assert result.metadata["planned_sections"] == 2
    assert result.metadata["generated_sections"] == 2


@pytest.mark.agent
async def test_blog_includes_seo_meta(multi_chain_mock):
    agent = _make_agent()
    plan = _blog_plan()
    output = _blog_output()
    context = _blog_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    seo = result.payload["seo_meta"]
    assert seo["meta_title"] == "The Rise of BRICS in 2025"
    assert seo["meta_description"] == "Explore how BRICS is reshaping global finance."
    assert "BRICS" in seo["keywords"]


@pytest.mark.agent
async def test_blog_includes_sources_used_per_section(multi_chain_mock):
    agent = _make_agent()
    plan = _blog_plan()
    output = _blog_output()
    context = _blog_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    sources_0 = result.payload["sections"][0]["sources_used"][0]
    sources_1 = result.payload["sections"][1]["sources_used"][0]
    assert str(sources_0) == _UUID_1
    assert str(sources_1) == _UUID_2


@pytest.mark.agent
async def test_blog_error_when_no_script_content():
    agent = _make_agent()
    context = {
        "script_content": "",
        "refined_context": "Some context",
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No script content" in result.reasoning
    assert result.confidence_score == 0.0


@pytest.mark.agent
async def test_blog_error_when_no_refined_context():
    agent = _make_agent()
    context = {
        "script_content": "Some script",
        "refined_context": "",
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No refined context" in result.reasoning
    assert result.confidence_score == 0.0


@pytest.mark.agent
async def test_blog_handles_empty_verified_claims(multi_chain_mock):
    agent = _make_agent()
    plan = _blog_plan()
    output = _blog_output()
    context = {
        "script_content": "Script text",
        "refined_context": "Context text",
        "verified_claims": [],
    }

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["_format"] == "blog"


@pytest.mark.agent
async def test_blog_handles_missing_verified_claims_key(multi_chain_mock):
    agent = _make_agent()
    plan = _blog_plan()
    output = _blog_output()
    context = {
        "script_content": "Script text",
        "refined_context": "Context text",
    }

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS


@pytest.mark.agent
async def test_blog_passes_correction_hint(multi_chain_mock):
    agent = _make_agent()
    plan = _blog_plan()
    output = _blog_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "correction_hint": "Add more sources",
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    assert len(calls) == 2
    for call in calls:
        invoked_input = call[0][0]
        assert invoked_input["correction_hint"] == "Add more sources"


@pytest.mark.agent
async def test_blog_formats_claims_text_correctly(multi_chain_mock):
    agent = _make_agent()
    plan = _blog_plan()
    output = _blog_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "verified_claims": [
            {
                "claim_text": "GDP grew 3.2%",
                "verdict": "SUPPORTED",
                "evidence_text": "IMF confirms.",
            },
            {
                "claim_text": "New payment system",
                "verdict": "SUPPORTED",
                "evidence_text": "Announced Q2.",
            },
        ],
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    claims_text = calls[0][0][0]["verified_claims"]
    assert "GDP grew 3.2% [SUPPORTED]" in claims_text
    assert "New payment system [SUPPORTED]" in claims_text


@pytest.mark.agent
async def test_blog_epistemic_ledger_in_hedge_block(
    multi_chain_mock,
):
    agent = _make_agent()
    plan = _blog_plan()
    output = _blog_output()
    context = {
        "script_content": "Script with uncertain claims.",
        "refined_context": "Context.",
        "verified_claims": [],
        "hedge_index": [
            {"claim_text": "GDP grew 3.2%", "verdict": "UNCERTAIN"},
        ],
        "epistemic_ledger": {
            "weak_passes": [
                {
                    "claim_text": "GDP grew 3.2%",
                    "verdict": "UNCERTAIN",
                    "confidence": 0.4,
                    "weakness_reason": "Limited data available",
                }
            ],
        },
    }

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
