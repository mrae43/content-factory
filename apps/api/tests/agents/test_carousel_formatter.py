import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from app.workers.agents import AgentActionStatus
from app.workers.formatters import (
    CarouselFormatterAgent,
    CarouselPlan,
    CarouselOutlineSlide,
    CarouselFormatterOutput,
    CarouselSlide,
)

_UUID_1 = str(uuid4())
_UUID_2 = str(uuid4())


def _make_agent():
    agent = CarouselFormatterAgent.__new__(CarouselFormatterAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.2
    agent.llm = MagicMock()
    return agent


def _carousel_plan():
    return CarouselPlan(
        thread_title="BRICS De-dollarization Thread",
        slides=[
            CarouselOutlineSlide(
                slide_number=1,
                purpose="Hook the reader",
                hook_type="question",
                key_claim="",
            ),
            CarouselOutlineSlide(
                slide_number=2,
                purpose="Present GDP data",
                hook_type="statistic",
                key_claim="BRICS GDP grew 3.2%",
            ),
            CarouselOutlineSlide(
                slide_number=3,
                purpose="CTA",
                hook_type="cta",
                key_claim="",
            ),
        ],
        hashtags=["BRICS", "economy"],
        cta_direction="Follow for more",
    )


def _carousel_output():
    return CarouselFormatterOutput(
        slides=[
            CarouselSlide(
                slide_number=1,
                text="Did you know BRICS nations are reshaping global finance?",
                visual_description="World map with BRICS highlighted",
                hook_type="question",
                sources_used=[_UUID_1],
            ),
            CarouselSlide(
                slide_number=2,
                text="BRICS GDP grew 3.2% in 2024, outpacing expectations.",
                visual_description="GDP growth chart",
                hook_type="statistic",
                sources_used=[_UUID_2],
            ),
            CarouselSlide(
                slide_number=3,
                text="Follow for more global economic insights!",
                visual_description="Subscribe CTA card",
                hook_type="cta",
                sources_used=[],
            ),
        ],
        thread_title="BRICS De-dollarization Thread",
        hashtags=["BRICS", "economy"],
        cta_slide="Follow for more global economic insights!",
    )


def _carousel_context():
    return {
        "script_content": "BRICS is reshaping global finance.",
        "refined_context": "BRICS GDP grew 3.2% in 2024.",
        "platform": "twitter",
        "verified_claims": [
            {
                "claim_text": "BRICS GDP grew 3.2%",
                "verdict": "SUPPORTED",
                "evidence_text": "IMF data.",
            }
        ],
    }


@pytest.mark.agent
async def test_carousel_happy_path_two_phase(multi_chain_mock):
    agent = _make_agent()
    plan = _carousel_plan()
    output = _carousel_output()
    context = _carousel_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["_format"] == "carousel"
    assert result.payload["_version"] == 1
    assert result.payload["thread_title"] == "BRICS De-dollarization Thread"
    assert len(result.payload["slides"]) == 3
    assert result.confidence_score == 0.9
    assert result.metadata["agent"] == "carousel_formatter"
    assert result.metadata["planned_slides"] == 3
    assert result.metadata["generated_slides"] == 3


@pytest.mark.agent
async def test_carousel_sets_empty_char_limit_violations(multi_chain_mock):
    agent = _make_agent()
    plan = _carousel_plan()
    output = _carousel_output()
    context = _carousel_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.payload["char_limit_violations"] == []


@pytest.mark.agent
async def test_carousel_includes_visual_descriptions(multi_chain_mock):
    agent = _make_agent()
    plan = _carousel_plan()
    output = _carousel_output()
    context = _carousel_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    for slide in result.payload["slides"]:
        assert "visual_description" in slide
        assert isinstance(slide["visual_description"], str)
        assert len(slide["visual_description"]) > 0


@pytest.mark.agent
async def test_carousel_includes_cta_slide(multi_chain_mock):
    agent = _make_agent()
    plan = _carousel_plan()
    output = _carousel_output()
    context = _carousel_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.payload["cta_slide"] == "Follow for more global economic insights!"


@pytest.mark.agent
async def test_carousel_error_when_no_script_content():
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
async def test_carousel_error_when_no_refined_context():
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
async def test_carousel_passes_platform_to_prompts(multi_chain_mock):
    agent = _make_agent()
    plan = _carousel_plan()
    output = _carousel_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "platform": "linkedin",
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    for call in calls:
        invoked_input = call[0][0]
        assert invoked_input["platform"] == "linkedin"


@pytest.mark.agent
async def test_carousel_defaults_platform_to_default_when_missing(multi_chain_mock):
    agent = _make_agent()
    plan = _carousel_plan()
    output = _carousel_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    for call in calls:
        invoked_input = call[0][0]
        assert invoked_input["platform"] == "default"


@pytest.mark.agent
async def test_carousel_passes_correction_hint(multi_chain_mock):
    agent = _make_agent()
    plan = _carousel_plan()
    output = _carousel_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "correction_hint": "Reduce slide text length",
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    for call in calls:
        invoked_input = call[0][0]
        assert invoked_input["correction_hint"] == "Reduce slide text length"


@pytest.mark.agent
async def test_carousel_handles_empty_verified_claims(multi_chain_mock):
    agent = _make_agent()
    plan = _carousel_plan()
    output = _carousel_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "verified_claims": [],
    }

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["_format"] == "carousel"
