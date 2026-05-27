import pytest
from unittest.mock import MagicMock

from app.workers.agents import AssetStudioAgent, AgentActionStatus


def _make_agent():
    agent = AssetStudioAgent.__new__(AssetStudioAgent)
    agent.model_name = "gemini-2.5-flash"
    agent.temperature = 0.2
    agent.llm = MagicMock()
    return agent


@pytest.mark.agent
async def test_returns_success_with_prompts_and_mock_url(
    asset_studio_context,
    studio_prompt_schema_output,
    chain_mock,
):
    agent = _make_agent()

    with chain_mock(studio_prompt_schema_output):
        result = await agent._execute(asset_studio_context)

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["video_url"].startswith("s3://factory/renders/")
    assert "visual_prompts" in result.payload["prompts"]
    assert "audio_prompts" in result.payload["prompts"]


@pytest.mark.agent
async def test_returns_success_with_empty_script_and_scenes(
    job_id,
    studio_prompt_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "",
        "scenes": [],
        "visual_style": "",
        "job_id": job_id,
    }

    with chain_mock(studio_prompt_schema_output):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS


@pytest.mark.agent
async def test_mock_url_contains_job_id(
    job_id,
    studio_prompt_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "job_id": job_id,
        "script_content": "test",
        "scenes": [],
        "visual_style": "",
    }

    with chain_mock(studio_prompt_schema_output):
        result = await agent._execute(context)

    assert str(job_id) in result.payload["video_url"]
    assert result.confidence_score == 0.9
    assert "model" in result.metadata
    assert "synth_id_enabled" in result.metadata


@pytest.mark.agent
async def test_underspecified_scene_returns_error(
    job_id,
    chain_mock,
):
    from app.workers.agents import StudioPromptSchema

    agent = _make_agent()
    error_output = StudioPromptSchema(
        status=AgentActionStatus.ERROR,
        visual_prompts=[],
        audio_prompts="",
        reasoning="Scene input is underspecified: generic visual_prompt, empty audio_cue, zero duration.",
    )
    context = {
        "script_content": "Something happened.",
        "scenes": [
            {
                "scene_number": 1,
                "narration_text": "A thing happened.",
                "visual_prompt": "A thing",
                "audio_cue": "",
                "duration_seconds": 0,
            }
        ],
        "visual_style": "vague style",
        "job_id": job_id,
    }

    with chain_mock(error_output):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.ERROR
    assert any(
        word in result.reasoning.lower()
        for word in ["underspecified", "ambiguous", "generic"]
    )
