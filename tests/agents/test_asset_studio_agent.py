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
async def test_returns_success_with_empty_script_and_storyboard(
    job_id,
    studio_prompt_schema_output,
    chain_mock,
):
    agent = _make_agent()
    context = {
        "script_content": "",
        "storyboard": [],
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
        "storyboard": [],
    }

    with chain_mock(studio_prompt_schema_output):
        result = await agent._execute(context)

    assert str(job_id) in result.payload["video_url"]
    assert result.confidence_score == 0.9
    assert "model" in result.metadata
    assert "synth_id_enabled" in result.metadata
