import pytest
from unittest.mock import MagicMock

from app.workers.agents import AgentActionStatus
from app.workers.short_formatter import (
    ShortFormatterAgent,
    ShortPlan,
    ShortSceneOutline,
    ShortFormatterOutput,
    _resolve_voice_id,
    _resolve_subtitle_preset,
    _resolve_short_aspect_ratio,
)
from app.schemas.formats import ShortScene


def _make_agent():
    agent = ShortFormatterAgent.__new__(ShortFormatterAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.2
    agent.llm = MagicMock()
    return agent


def _short_plan(loopable: bool = True):
    return ShortPlan(
        proposed_title="BRICS De-dollarization in 60 Seconds",
        scene_outline=[
            ShortSceneOutline(
                scene_number=1,
                purpose="Hook with a provocative question",
                key_visual="World map with BRICS nations glowing gold",
                duration_estimate=5.0,
                suggested_asset_type="video_clip",
            ),
            ShortSceneOutline(
                scene_number=2,
                purpose="Present GDP data with a chart",
                key_visual="Animated bar chart of BRICS GDP",
                duration_estimate=8.0,
                suggested_asset_type="ken_burns",
            ),
            ShortSceneOutline(
                scene_number=3,
                purpose="Closer with a call-to-action",
                key_visual="Currency symbols dissolving into digital network",
                duration_estimate=5.0,
                suggested_asset_type="video_clip",
            ),
        ],
        visual_style_direction="Cinematic documentary with golden hour lighting",
        audio_direction="Orchestral with electronic undertones",
        music_mood="synthwave_hype",
        voice_id="voice_123",
        subtitle_preset="CENTER_POP_YELLOW",
        loop_hook="Wait — didn't we start with this exact question?"
        if loopable
        else None,
    )


def _short_output(loopable: bool = True):
    return ShortFormatterOutput(
        scenes=[
            ShortScene(
                scene_number=1,
                narration_text="Did you know BRICS nations are quietly reshaping global finance?",
                visual_prompt="Cinematic drone shot over a world map with BRICS nations illuminated in gold",
                asset_type="video_clip",
                kb_motion=None,
                sfx_cue=None,
                target_duration_seconds=5.0,
            ),
            ShortScene(
                scene_number=2,
                narration_text="BRICS collective GDP grew 3.2% in 2024, outpacing expectations.",
                visual_prompt="Close-up of an animated bar chart showing GDP growth across BRICS nations",
                asset_type="ken_burns",
                kb_motion="zoom_in",
                sfx_cue=None,
                target_duration_seconds=8.0,
            ),
            ShortScene(
                scene_number=3,
                narration_text="Will the dollar survive this shift? Follow for the full story.",
                visual_prompt="Currency symbols dissolving into a digital payment network",
                asset_type="video_clip",
                kb_motion=None,
                sfx_cue=None,
                target_duration_seconds=5.0,
            ),
        ],
        target_total_duration=18.0,
        visual_style="Cinematic documentary with golden hour lighting",
        audio_direction="Orchestral with electronic undertones",
        music_mood="synthwave_hype",
        voice_id="voice_123",
        subtitle_preset="CENTER_POP_YELLOW",
        loop_hook="Wait — didn't we start with this exact question?"
        if loopable
        else None,
    )


def _short_context():
    return {
        "script_content": "Did you know BRICS is reshaping global finance?",
        "refined_context": "BRICS GDP grew 3.2% in 2024.",
        "platform": "tiktok",
        "loopable": True,
        "voice_id": "voice_123",
        "verified_claims": [
            {
                "claim_text": "BRICS GDP grew 3.2%",
                "verdict": "SUPPORTED",
                "evidence_text": "IMF data confirms.",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Happy-path two-phase
# ---------------------------------------------------------------------------


@pytest.mark.agent
async def test_short_happy_path_two_phase(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan(loopable=True)
    output = _short_output(loopable=True)
    context = _short_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["_format"] == "short"
    assert result.payload["_version"] == 1
    assert len(result.payload["scenes"]) == 3
    assert result.confidence_score == 0.9
    assert result.metadata["agent"] == "short_formatter"
    assert result.metadata["planned_scenes"] == 3
    assert result.metadata["generated_scenes"] == 3
    assert result.metadata["loopable"] is True
    assert result.metadata["platform"] == "tiktok"


@pytest.mark.agent
async def test_short_includes_loop_hook_when_loopable(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan(loopable=True)
    output = _short_output(loopable=True)
    context = _short_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert (
        result.payload["loop_hook"]
        == "Wait — didn't we start with this exact question?"
    )


@pytest.mark.agent
async def test_short_omits_loop_hook_when_not_loopable(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan(loopable=False)
    plan.loop_hook = None
    output = _short_output(loopable=False)
    output.loop_hook = None
    context = {**_short_context(), "loopable": False}

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload.get("loop_hook") is None


@pytest.mark.agent
async def test_short_includes_voice_id_from_context(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = _short_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.payload["voice_id"] == "voice_123"


@pytest.mark.agent
async def test_short_includes_subtitle_preset(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = _short_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.payload["subtitle_preset"] == "CENTER_POP_YELLOW"


@pytest.mark.agent
async def test_short_includes_scene_details(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = _short_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    scene = result.payload["scenes"][0]
    assert scene["scene_number"] == 1
    assert len(scene["narration_text"]) >= 10
    assert len(scene["visual_prompt"]) >= 10
    assert scene["asset_type"] == "video_clip"
    assert scene["target_duration_seconds"] == 5.0

    scene2 = result.payload["scenes"][1]
    assert scene2["asset_type"] == "ken_burns"
    assert scene2["kb_motion"] == "zoom_in"


@pytest.mark.agent
async def test_short_includes_target_total_duration(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = _short_context()

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.payload["target_total_duration"] == 18.0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@pytest.mark.agent
async def test_short_error_when_no_script_content():
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
async def test_short_error_when_no_refined_context():
    agent = _make_agent()
    context = {
        "script_content": "Some script",
        "refined_context": "",
    }

    result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert "No refined context" in result.reasoning
    assert result.confidence_score == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.agent
async def test_short_handles_empty_verified_claims(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = {
        "script_content": "Script text",
        "refined_context": "Context text",
        "platform": "tiktok",
        "loopable": True,
        "voice_id": "voice_123",
        "verified_claims": [],
    }

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["_format"] == "short"


@pytest.mark.agent
async def test_short_handles_missing_verified_claims_key(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = {
        "script_content": "Script text",
        "refined_context": "Context text",
        "platform": "tiktok",
        "loopable": True,
        "voice_id": "voice_123",
    }

    with multi_chain_mock([plan, output]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS


@pytest.mark.agent
async def test_short_passes_correction_hint(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "platform": "tiktok",
        "loopable": True,
        "voice_id": "voice_123",
        "correction_hint": "Make the hook punchier",
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    assert len(calls) == 2
    for call in calls:
        invoked_input = call[0][0]
        assert invoked_input["correction_hint"] == "Make the hook punchier"


@pytest.mark.agent
async def test_short_formats_claims_text_correctly(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "platform": "tiktok",
        "loopable": True,
        "voice_id": "voice_123",
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
async def test_short_passes_platform_to_prompts(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "platform": "instagram",
        "loopable": True,
        "voice_id": "voice_123",
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    for call in calls:
        invoked_input = call[0][0]
        assert invoked_input["platform"] == "instagram"


@pytest.mark.agent
async def test_short_defaults_platform_to_default_when_missing(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = {
        "script_content": "Script",
        "refined_context": "Context",
        "loopable": True,
        "voice_id": "voice_123",
    }

    with multi_chain_mock([plan, output]) as mock_ainvoke:
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    calls = mock_ainvoke.call_args_list
    for call in calls:
        invoked_input = call[0][0]
        assert invoked_input["platform"] == "default"


@pytest.mark.agent
async def test_short_epistemic_ledger_in_hedge_block(multi_chain_mock):
    agent = _make_agent()
    plan = _short_plan()
    output = _short_output()
    context = {
        "script_content": "Script with uncertain claims.",
        "refined_context": "Context.",
        "platform": "tiktok",
        "loopable": True,
        "voice_id": "voice_123",
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


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestResolveVoiceId:
    def test_returns_directives_voice_id_when_set(self, monkeypatch):
        monkeypatch.setattr(
            "app.workers.short_formatter.DEFAULT_VOICE_MAP",
            {"tiktok": "default_tiktok_voice"},
        )
        directives = {"voice_id": "custom_voice"}
        assert _resolve_voice_id(directives, "tiktok") == "custom_voice"

    def test_falls_back_to_platform_default(self, monkeypatch):
        monkeypatch.setattr(
            "app.workers.short_formatter.DEFAULT_VOICE_MAP",
            {"tiktok": "default_tiktok_voice"},
        )
        directives = {}
        assert _resolve_voice_id(directives, "tiktok") == "default_tiktok_voice"

    def test_returns_empty_string_when_no_default(self, monkeypatch):
        monkeypatch.setattr(
            "app.workers.short_formatter.DEFAULT_VOICE_MAP",
            {},
        )
        directives = {}
        assert _resolve_voice_id(directives, "tiktok") == ""


class TestResolveSubtitlePreset:
    def test_returns_payload_preset_when_set(self):
        payload = {"subtitle_preset": "NEON_BOXED"}
        assert _resolve_subtitle_preset(payload, "tiktok") == "NEON_BOXED"

    def test_falls_back_to_platform_default(self):
        payload = {}
        assert _resolve_subtitle_preset(payload, "youtube") == "CLEAN_WHITE_LOWER"

    def test_falls_back_to_global_default_for_unknown_platform(self):
        payload = {}
        assert (
            _resolve_subtitle_preset(payload, "unknown_platform") == "CENTER_POP_YELLOW"
        )


class TestResolveShortAspectRatio:
    def test_tiktok_is_9_16(self):
        assert _resolve_short_aspect_ratio("tiktok") == "9:16"

    def test_youtube_is_9_16(self):
        assert _resolve_short_aspect_ratio("youtube") == "9:16"

    def test_instagram_is_4_5(self):
        assert _resolve_short_aspect_ratio("instagram") == "4:5"

    def test_twitter_falls_back_to_existing_map(self):
        assert _resolve_short_aspect_ratio("twitter") == "16:9"

    def test_unknown_falls_back_to_16_9(self):
        assert _resolve_short_aspect_ratio("unknown") == "16:9"
