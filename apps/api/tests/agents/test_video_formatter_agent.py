import pytest
from unittest.mock import MagicMock

from app.workers.agents import AgentActionStatus
from app.workers.formatters import (
    VideoFormatterAgent,
    VideoPlan,
    VideoSceneOutline,
    VideoFormatterOutput,
)
from app.schemas.formats import VideoScene


def _make_agent():
    agent = VideoFormatterAgent.__new__(VideoFormatterAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.2
    agent.llm = MagicMock()
    return agent


def _video_plan():
    return VideoPlan(
        proposed_title="BRICS De-dollarization Explained",
        scene_outline=[
            VideoSceneOutline(
                scene_number=1,
                purpose="Hook the audience with a striking question",
                key_visual="World map with BRICS nations glowing",
                duration_estimate=15.0,
            ),
            VideoSceneOutline(
                scene_number=2,
                purpose="Present GDP growth data",
                key_visual="Bar chart of BRICS GDP growth",
                duration_estimate=30.0,
            ),
            VideoSceneOutline(
                scene_number=3,
                purpose="CTA — close with a thought-provoking statement",
                key_visual="Currency symbols dissolving into digital network",
                duration_estimate=20.0,
            ),
        ],
        visual_style_direction="Cinematic documentary with golden hour lighting",
    )


def _video_output():
    return VideoFormatterOutput(
        scenes=[
            VideoScene(
                scene_number=1,
                narration_text="Did you know BRICS nations are quietly reshaping global finance?",
                visual_prompt="Cinematic drone shot over a world map with BRICS nations illuminated in gold",
                audio_cue="Tension build with low strings",
                duration_seconds=15.0,
            ),
            VideoScene(
                scene_number=2,
                narration_text="BRICS collective GDP grew 3.2% in 2024, outpacing expectations.",
                visual_prompt="Close-up of an animated bar chart showing GDP growth across BRICS nations",
                audio_cue="Data reveal with rising piano notes",
                duration_seconds=30.0,
            ),
            VideoScene(
                scene_number=3,
                narration_text="Will the dollar's dominance survive this shift? The answer may surprise you.",
                visual_prompt="Currency symbols dissolving into a digital payment network",
                audio_cue="Reflective piano with ambient fade-out",
                duration_seconds=20.0,
            ),
        ],
        total_duration_seconds=65.0,
        visual_style="Cinematic documentary with golden hour lighting and smooth camera transitions",
        audio_direction="Orchestral with electronic undertones",
        unified_visual_prompt="Open on a cinematic drone shot over a world map with BRICS nations illuminated in gold. Cut to a close-up of an animated bar chart showing GDP growth. Close on currency symbols dissolving into a digital payment network with warm amber lighting.",
    )


def _video_context():
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
async def test_video_happy_path_two_phase(multi_chain_mock):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
    context = _video_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["_format"] == "video"
    assert result.payload["_version"] == 1
    assert len(result.payload["scenes"]) == 3
    assert result.confidence_score == 0.9
    assert result.metadata["agent"] == "video_formatter"
    assert result.metadata["planned_scenes"] == 3
    assert result.metadata["generated_scenes"] == 3


@pytest.mark.agent
async def test_video_includes_visual_style(multi_chain_mock):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
    context = _video_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["visual_style"] == (
        "Cinematic documentary with golden hour lighting and smooth camera transitions"
    )
    assert result.payload["audio_direction"] == "Orchestral with electronic undertones"


@pytest.mark.agent
async def test_video_includes_scene_details(multi_chain_mock):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
    context = _video_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    scene = result.payload["scenes"][0]
    assert scene["scene_number"] == 1
    assert len(scene["narration_text"]) >= 10
    assert len(scene["visual_prompt"]) >= 10
    assert scene["duration_seconds"] == 15.0


@pytest.mark.agent
async def test_video_includes_total_duration(multi_chain_mock):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
    context = _video_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["total_duration_seconds"] == 65.0


@pytest.mark.agent
async def test_video_error_when_no_script_content():
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
async def test_video_error_when_no_refined_context():
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
async def test_video_handles_empty_verified_claims(multi_chain_mock):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
    context = {
        "script_content": "Script text",
        "refined_context": "Context text",
        "verified_claims": [],
    }

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["_format"] == "video"


@pytest.mark.agent
async def test_video_handles_missing_verified_claims_key(multi_chain_mock):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
    context = {
        "script_content": "Script text",
        "refined_context": "Context text",
    }

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS


@pytest.mark.agent
async def test_video_passes_correction_hint(multi_chain_mock):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "correction_hint": "Ensure all scenes have narration",
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    assert len(calls) == 2
    for call in calls:
        invoked_input = call[0][0]
        assert invoked_input["correction_hint"] == "Ensure all scenes have narration"


@pytest.mark.agent
async def test_video_formats_claims_text_correctly(multi_chain_mock):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
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
async def test_video_epistemic_ledger_in_hedge_block(
    multi_chain_mock,
):
    agent = _make_agent()
    plan = _video_plan()
    output = _video_output()
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
