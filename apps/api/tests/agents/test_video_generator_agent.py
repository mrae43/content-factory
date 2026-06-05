from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.workers.agents import AgentActionStatus
from app.workers.video_generator_agent import VideoGeneratorAgent
from app.services.tools import Tool


def _make_gen_result(job_id: str | None = "test-job-123") -> dict:
    return {"job_id": job_id}


def _make_poll_result(
    status: str = "completed",
    download_url: str | None = "https://example.com/video.mp4",
    failure_reason: str | None = None,
) -> dict:
    return {
        "status": status,
        "download_url": download_url,
        "failure_reason": failure_reason,
    }


def _make_gen_tool(side_effect=None, return_value=None):
    mock_gen = MagicMock()
    if return_value is not None:
        mock_gen.callable = AsyncMock(return_value=return_value)
    elif side_effect is not None:
        mock_gen.callable = AsyncMock(side_effect=side_effect)
    else:
        mock_gen.callable = AsyncMock(return_value=_make_gen_result())
    return Tool(
        name="generate_video",
        description="Test gen",
        callable=mock_gen.callable,
        permissions={"*"},
    )


def _make_poll_tool(side_effect=None):
    mock_poll = MagicMock()
    if side_effect is not None:
        mock_poll.callable = AsyncMock(side_effect=side_effect)
    else:
        mock_poll.callable = AsyncMock(return_value=_make_poll_result())
    return Tool(
        name="poll_video",
        description="Test poll",
        callable=mock_poll.callable,
        permissions={"*"},
    )


def _make_upload_tool(return_url: str = "/api/videos/test.mp4"):
    mock_up = MagicMock()
    mock_up.callable = AsyncMock(return_value=return_url)
    return Tool(
        name="upload_video",
        description="Test upload",
        callable=mock_up.callable,
        permissions={"*"},
    )


def _make_agent(gen_tool=None, poll_tool=None, upload_tool=None):
    agent = VideoGeneratorAgent.__new__(VideoGeneratorAgent)
    agent.di_tools = {}
    if gen_tool:
        agent.di_tools["generate_video"] = gen_tool
    if poll_tool:
        agent.di_tools["poll_video"] = poll_tool
    if upload_tool:
        agent.di_tools["upload_video"] = upload_tool
    return agent


def _format_payload(
    prompt: str = "A cinematic documentary visual style with tension-building and climax scenes.",
    duration: float = 90.0,
) -> dict:
    return {
        "unified_visual_prompt": prompt,
        "total_duration_seconds": duration,
        "scenes": [],
        "visual_style": "cinematic documentary",
        "audio_direction": "tension-building",
    }


@pytest.fixture(autouse=True)
def _patch_settings():
    with patch("app.workers.video_generator_agent.settings") as mock:
        mock.video_gen_model = "minimax/video-01-director"
        mock.video_gen_poll_interval_seconds = 0.01
        mock.video_gen_max_poll_retries = 60
        yield mock


@pytest.fixture(autouse=True)
def _patch_aiohttp():
    """Mock aiohttp download so tests don't make real HTTP calls."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.read = AsyncMock(return_value=b"fake-video-bytes")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        yield


@pytest.mark.agent
async def test_happy_path_generates_and_uploads_video():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    poll_tool = _make_poll_tool()
    upload_tool = _make_upload_tool(return_url="/api/videos/final.mp4")

    agent = _make_agent(gen_tool=gen_tool, poll_tool=poll_tool, upload_tool=upload_tool)
    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(),
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["video_url"] == "/api/videos/final.mp4"
    assert result.payload["duration"] == 90.0
    assert gen_tool.callable.call_count == 1
    assert poll_tool.callable.call_count >= 1
    assert upload_tool.callable.call_count == 1


@pytest.mark.agent
async def test_returns_error_when_format_payload_not_a_dict():
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": "not-a-dict",
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert "must be a dict" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_missing_unified_visual_prompt():
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": {"total_duration_seconds": 90.0},
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert "unified_visual_prompt" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_gen_tool_missing():
    poll_tool = _make_poll_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(poll_tool=poll_tool, upload_tool=upload_tool)
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(),
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert "generate_video" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_poll_tool_missing():
    gen_tool = _make_gen_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(),
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert "poll_video" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_upload_tool_missing():
    gen_tool = _make_gen_tool()
    poll_tool = _make_poll_tool()

    agent = _make_agent(gen_tool=gen_tool, poll_tool=poll_tool)
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(),
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert "upload_video" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_polls_exhausted():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    max_retries = 3
    poll_tool = _make_poll_tool(
        side_effect=[_make_poll_result(status="processing", download_url=None)]
        * max_retries
    )
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, poll_tool=poll_tool, upload_tool=upload_tool)

    with patch(
        "app.workers.video_generator_agent.settings.video_gen_max_poll_retries",
        max_retries,
    ):
        with patch(
            "app.workers.video_generator_agent.settings.video_gen_poll_interval_seconds",
            0.005,
        ):
            result = await agent.run(
                {
                    "job_id": jid,
                    "format_payload": _format_payload(),
                }
            )

    assert result.status == AgentActionStatus.ERROR
    assert "timed out" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_poll_returns_failed():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    poll_tool = _make_poll_tool(
        side_effect=[
            _make_poll_result(
                status="failed",
                download_url=None,
                failure_reason="Model overloaded",
            )
        ]
    )
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, poll_tool=poll_tool, upload_tool=upload_tool)

    with patch(
        "app.workers.video_generator_agent.settings.video_gen_poll_interval_seconds",
        0.005,
    ):
        result = await agent.run(
            {
                "job_id": jid,
                "format_payload": _format_payload(),
            }
        )

    assert result.status == AgentActionStatus.ERROR
    assert "Model overloaded" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_no_job_id_from_gen():
    gen_tool = _make_gen_tool(return_value={"job_id": None})
    poll_tool = _make_poll_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, poll_tool=poll_tool, upload_tool=upload_tool)
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(),
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert "No job_id" in result.reasoning


@pytest.mark.agent
async def test_upload_failure_still_returns_success_with_download_url():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    poll_tool = _make_poll_tool()
    upload_tool = _make_upload_tool()
    upload_tool.callable = AsyncMock(side_effect=Exception("Upload failed"))

    agent = _make_agent(gen_tool=gen_tool, poll_tool=poll_tool, upload_tool=upload_tool)
    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(),
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["video_url"] == "https://example.com/video.mp4"
    assert "upload failed" in result.reasoning


@pytest.mark.agent
async def test_gen_tool_side_effect():
    gen_tool = _make_gen_tool(side_effect=Exception("API unavailable"))
    poll_tool = _make_poll_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, poll_tool=poll_tool, upload_tool=upload_tool)
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(),
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert "API unavailable" in result.reasoning
