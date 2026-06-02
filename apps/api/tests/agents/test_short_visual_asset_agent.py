from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.workers.agents import AgentActionStatus
from app.workers.short_visual_asset_agent import ShortVisualAssetAgent
from app.services.tools import Tool


def _make_gen_video_result(job_id: str | None = "test-video-job-123") -> dict:
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


def _make_gen_image_result(
    success: bool = True,
    image_bytes: bytes | None = b"img",
    failure_reason: str | None = None,
) -> dict:
    return {
        "success": success,
        "image_bytes": image_bytes,
        "width": 1088,
        "height": 1920,
        "failure_reason": failure_reason,
        "prompt_used": "test prompt",
    }


def _make_gen_video_tool(side_effect=None, return_value=None):
    mock_gen = MagicMock()
    if return_value is not None:
        mock_gen.callable = AsyncMock(return_value=return_value)
    elif side_effect is not None:
        mock_gen.callable = AsyncMock(side_effect=side_effect)
    else:
        mock_gen.callable = AsyncMock(return_value=_make_gen_video_result())
    return Tool(
        name="generate_video",
        description="Test gen video",
        callable=mock_gen.callable,
        permissions={"*"},
    )


def _make_poll_video_tool(side_effect=None):
    mock_poll = MagicMock()
    if side_effect is not None:
        mock_poll.callable = AsyncMock(side_effect=side_effect)
    else:
        mock_poll.callable = AsyncMock(return_value=_make_poll_result())
    return Tool(
        name="poll_video",
        description="Test poll video",
        callable=mock_poll.callable,
        permissions={"*"},
    )


def _make_upload_video_tool(return_url: str = "/api/videos/scene.mp4"):
    mock_up = MagicMock()
    mock_up.callable = AsyncMock(return_value=return_url)
    return Tool(
        name="upload_video",
        description="Test upload video",
        callable=mock_up.callable,
        permissions={"*"},
    )


def _make_gen_image_tool(side_effect=None, return_value=None):
    mock_gen = MagicMock()
    if side_effect is not None:
        mock_gen.callable = AsyncMock(side_effect=side_effect)
    elif return_value is not None:
        mock_gen.callable = AsyncMock(return_value=return_value)
    else:
        mock_gen.callable = AsyncMock(return_value=_make_gen_image_result())
    return Tool(
        name="generate_image",
        description="Test gen image",
        callable=mock_gen.callable,
        permissions={"*"},
    )


def _make_upload_image_tool(return_url: str = "/api/images/scene.png"):
    mock_up = MagicMock()
    mock_up.callable = AsyncMock(return_value=return_url)
    return Tool(
        name="upload_image",
        description="Test upload image",
        callable=mock_up.callable,
        permissions={"*"},
    )


def _make_agent(
    gen_video_tool=None,
    poll_video_tool=None,
    upload_video_tool=None,
    gen_image_tool=None,
    upload_image_tool=None,
):
    agent = ShortVisualAssetAgent.__new__(ShortVisualAssetAgent)
    agent.di_tools = {}
    if gen_video_tool:
        agent.di_tools["generate_video"] = gen_video_tool
    if poll_video_tool:
        agent.di_tools["poll_video"] = poll_video_tool
    if upload_video_tool:
        agent.di_tools["upload_video"] = upload_video_tool
    if gen_image_tool:
        agent.di_tools["generate_image"] = gen_image_tool
    if upload_image_tool:
        agent.di_tools["upload_image"] = upload_image_tool
    return agent


def _scene(
    n: int = 1,
    asset_type: str = "video_clip",
    visual_prompt: str = "A dramatic cityscape at sunset",
    narration: str = "The city never sleeps.",
    kb_motion: str | None = None,
    target_duration: float = 5.0,
) -> dict:
    scene = {
        "scene_number": n,
        "narration_text": narration,
        "visual_prompt": visual_prompt,
        "asset_type": asset_type,
        "target_duration_seconds": target_duration,
    }
    if kb_motion is not None:
        scene["kb_motion"] = kb_motion
    return scene


def _format_payload(scenes: list[dict]) -> dict:
    return {
        "format": "short",
        "version": 1,
        "scenes": scenes,
        "target_total_duration": 30.0,
        "visual_style": "cinematic",
        "audio_direction": "upbeat",
        "music_mood": "synthwave_hype",
        "voice_id": "test-voice",
        "subtitle_preset": "CENTER_POP_YELLOW",
    }


@pytest.fixture(autouse=True)
def _patch_settings():
    with patch("app.workers.short_visual_asset_agent.settings") as mock:
        mock.video_gen_model = ""
        mock.video_gen_poll_interval_seconds = 0.005
        mock.video_gen_max_poll_retries = 60
        mock.image_gen_slide_delay = 0.0
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


# ───────────────────────────
# Happy paths
# ───────────────────────────


@pytest.mark.agent
async def test_all_video_clip_scenes_success():
    jid = uuid4()
    scenes = [_scene(1, "video_clip"), _scene(2, "video_clip")]
    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(return_url="/vids/scene.mp4"),
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=_make_upload_image_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.metadata["successful_scenes"] == 2
    assert result.metadata["failed_scenes"] == 0
    urls = result.payload["scene_urls"]
    assert len(urls) == 2
    assert urls[0]["asset_type"] == "video_clip"
    assert urls[1]["asset_type"] == "video_clip"
    updated = result.payload["updated_format_payload"]["scenes"]
    assert updated[0]["video_url"] == "/vids/scene.mp4"
    assert updated[1]["video_url"] == "/vids/scene.mp4"


@pytest.mark.agent
async def test_all_ken_burns_scenes_success():
    jid = uuid4()
    scenes = [
        _scene(1, "ken_burns", kb_motion="zoom_in"),
        _scene(2, "ken_burns", kb_motion="pan_left"),
    ]
    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=_make_upload_image_tool(return_url="/imgs/scene.png"),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "instagram",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.metadata["successful_scenes"] == 2
    urls = result.payload["scene_urls"]
    assert len(urls) == 2
    assert urls[0]["asset_type"] == "ken_burns"
    assert urls[1]["asset_type"] == "ken_burns"
    updated = result.payload["updated_format_payload"]["scenes"]
    assert updated[0]["image_url"] == "/imgs/scene.png"
    assert updated[1]["image_url"] == "/imgs/scene.png"


@pytest.mark.agent
async def test_mixed_video_and_ken_burns_scenes():
    jid = uuid4()
    scenes = [
        _scene(1, "video_clip"),
        _scene(2, "ken_burns", kb_motion="zoom_out"),
    ]
    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(return_url="/vids/scene.mp4"),
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=_make_upload_image_tool(return_url="/imgs/scene.png"),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.metadata["successful_scenes"] == 2
    urls = result.payload["scene_urls"]
    assert urls[0]["asset_type"] == "video_clip"
    assert urls[1]["asset_type"] == "ken_burns"


# ───────────────────────────
# Fallback behaviour
# ───────────────────────────


@pytest.mark.agent
async def test_video_fallback_to_ken_burns_after_retry():
    """Video fails twice, falls back to Ken Burns, which succeeds."""
    jid = uuid4()
    scenes = [_scene(1, "video_clip")]

    # generate_video always fails
    gen_video_tool = _make_gen_video_tool(side_effect=Exception("API down"))
    gen_image_tool = _make_gen_image_tool()
    upload_image_tool = _make_upload_image_tool(return_url="/imgs/fallback.png")

    agent = _make_agent(
        gen_video_tool=gen_video_tool,
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=gen_image_tool,
        upload_image_tool=upload_image_tool,
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.metadata["successful_scenes"] == 1
    updated = result.payload["updated_format_payload"]["scenes"][0]
    assert updated["asset_type"] == "ken_burns"
    assert updated["kb_motion"] == "zoom_in"
    assert updated["image_url"] == "/imgs/fallback.png"


@pytest.mark.agent
async def test_video_fallback_to_ken_burns_then_image_also_fails():
    """Video fails twice, fallback Ken Burns also fails → scene failure."""
    jid = uuid4()
    scenes = [_scene(1, "video_clip")]

    gen_video_tool = _make_gen_video_tool(side_effect=Exception("API down"))
    gen_image_tool = _make_gen_image_tool(
        return_value=_make_gen_image_result(
            success=False, image_bytes=None, failure_reason="FLUX error"
        )
    )

    agent = _make_agent(
        gen_video_tool=gen_video_tool,
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=gen_image_tool,
        upload_image_tool=_make_upload_image_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert result.metadata["failed_scenes"] == 1
    assert result.metadata["failures"][0]["reason"] == "FLUX error"


@pytest.mark.agent
async def test_video_succeeds_on_second_attempt():
    """First video generation fails, retry succeeds."""
    jid = uuid4()
    scenes = [_scene(1, "video_clip")]

    # First call fails, second succeeds
    gen_video_tool = _make_gen_video_tool(
        side_effect=[
            Exception("Timeout"),
            _make_gen_video_result("vid-123"),
        ]
    )

    agent = _make_agent(
        gen_video_tool=gen_video_tool,
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(return_url="/vids/retry.mp4"),
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=_make_upload_image_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.metadata["successful_scenes"] == 1
    assert gen_video_tool.callable.call_count == 2
    updated = result.payload["updated_format_payload"]["scenes"][0]
    assert updated["video_url"] == "/vids/retry.mp4"
    assert updated["asset_type"] == "video_clip"


# ───────────────────────────
# Error / edge cases
# ───────────────────────────


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
async def test_returns_error_when_no_scenes():
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "No scenes" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_missing_generate_video_tool():
    agent = _make_agent(
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=_make_upload_image_tool(),
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "generate_video" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_missing_generate_image_tool():
    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        upload_image_tool=_make_upload_image_tool(),
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(
                [_scene(1, "ken_burns", kb_motion="zoom_in")]
            ),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "generate_image" in result.reasoning


@pytest.mark.agent
async def test_partial_failure_still_succeeds():
    jid = uuid4()
    scenes = [
        _scene(1, "ken_burns", kb_motion="zoom_in"),
        _scene(2, "ken_burns", kb_motion="pan_right"),
    ]

    gen_image_tool = _make_gen_image_tool(
        side_effect=[
            _make_gen_image_result(success=True, image_bytes=b"ok"),
            _make_gen_image_result(
                success=False, image_bytes=None, failure_reason="Rate limit"
            ),
        ]
    )
    upload_image_tool = _make_upload_image_tool(return_url="/imgs/ok.png")

    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=gen_image_tool,
        upload_image_tool=upload_image_tool,
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.metadata["successful_scenes"] == 1
    assert result.metadata["failed_scenes"] == 1
    updated = result.payload["updated_format_payload"]["scenes"]
    assert updated[0]["image_url"] is not None
    assert updated[1].get("image_url") is None


@pytest.mark.agent
async def test_all_scenes_fail_returns_error():
    jid = uuid4()
    scenes = [_scene(1, "ken_burns", kb_motion="zoom_in")]

    gen_image_tool = _make_gen_image_tool(
        return_value=_make_gen_image_result(
            success=False, image_bytes=None, failure_reason="API down"
        )
    )

    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=gen_image_tool,
        upload_image_tool=_make_upload_image_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert result.metadata["failed_scenes"] == 1
    assert result.metadata["successful_scenes"] == 0


@pytest.mark.agent
async def test_empty_visual_prompt_skips_scene():
    jid = uuid4()
    scenes = [_scene(1, "video_clip", visual_prompt="")]

    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=_make_upload_image_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert result.metadata["failed_scenes"] == 1
    assert result.metadata["failures"][0]["reason"] == "Empty visual_prompt"


@pytest.mark.agent
async def test_poll_failure_then_retry_success():
    """Poll fails on first attempt, second gen+poll succeeds."""
    jid = uuid4()
    scenes = [_scene(1, "video_clip")]

    # First: gen succeeds but poll fails (times out / returns failed)
    # Second: gen succeeds and poll succeeds
    gen_video_tool = _make_gen_video_tool(
        side_effect=[
            _make_gen_video_result("job-1"),
            _make_gen_video_result("job-2"),
        ]
    )
    poll_video_tool = _make_poll_video_tool(
        side_effect=[
            # First attempt polls — fail after a few processing responses
            _make_poll_result(status="processing"),
            _make_poll_result(status="failed", failure_reason="Model overloaded"),
            # Second attempt polls — succeed immediately
            _make_poll_result(
                status="completed", download_url="https://example.com/v2.mp4"
            ),
        ]
    )

    agent = _make_agent(
        gen_video_tool=gen_video_tool,
        poll_video_tool=poll_video_tool,
        upload_video_tool=_make_upload_video_tool(return_url="/vids/final.mp4"),
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=_make_upload_image_tool(),
    )

    with patch(
        "app.workers.short_visual_asset_agent.settings.video_gen_max_poll_retries",
        2,
    ):
        with patch(
            "app.workers.short_visual_asset_agent.settings.video_gen_poll_interval_seconds",
            0.001,
        ):
            result = await agent.run(
                {
                    "job_id": jid,
                    "format_payload": _format_payload(scenes),
                    "platform": "tiktok",
                }
            )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.metadata["successful_scenes"] == 1


@pytest.mark.agent
async def test_metadata_tracks_failures():
    jid = uuid4()
    scenes = [
        _scene(1, "ken_burns", kb_motion="zoom_in"),
        _scene(2, "ken_burns", kb_motion="pan_right"),
    ]

    gen_image_tool = _make_gen_image_tool(
        side_effect=[
            _make_gen_image_result(success=True, image_bytes=b"a"),
            _make_gen_image_result(
                success=False, image_bytes=None, failure_reason="Rate limited"
            ),
        ]
    )
    upload_image_tool = _make_upload_image_tool()

    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=gen_image_tool,
        upload_image_tool=upload_image_tool,
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
        }
    )

    assert len(result.metadata["failures"]) == 1
    assert result.metadata["failures"][0]["scene_number"] == 2
    assert result.metadata["failures"][0]["reason"] == "Rate limited"


@pytest.mark.agent
async def test_filename_and_folder_include_ids():
    jid = uuid4()
    scenes = [_scene(1, "video_clip")]

    upload_video_tool = _make_upload_video_tool()
    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=upload_video_tool,
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=_make_upload_image_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "tiktok",
            "device_id": "test-device",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    args, kwargs = upload_video_tool.callable.call_args
    filename = args[1]
    folder = kwargs.get("folder", args[2] if len(args) > 2 else "")
    assert filename == "scene_01.mp4"
    assert str(jid) in folder
    assert "test-device" in folder


@pytest.mark.agent
async def test_ken_burns_filename_format():
    jid = uuid4()
    scenes = [_scene(1, "ken_burns", kb_motion="zoom_in")]

    upload_image_tool = _make_upload_image_tool()
    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=_make_gen_image_tool(),
        upload_image_tool=upload_image_tool,
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "platform": "instagram",
            "device_id": "dev-42",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    args, kwargs = upload_image_tool.callable.call_args
    filename = args[1]
    folder = kwargs.get("folder", args[2] if len(args) > 2 else "")
    assert filename == "scene_01.png"
    assert str(jid) in folder
    assert "dev-42" in folder


@pytest.mark.agent
async def test_defaults_platform_to_tiktok():
    jid = uuid4()
    scenes = [_scene(1, "ken_burns", kb_motion="zoom_in")]

    gen_image_tool = _make_gen_image_tool()
    agent = _make_agent(
        gen_video_tool=_make_gen_video_tool(),
        poll_video_tool=_make_poll_video_tool(),
        upload_video_tool=_make_upload_video_tool(),
        gen_image_tool=gen_image_tool,
        upload_image_tool=_make_upload_image_tool(),
    )

    await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
        }
    )

    gen_image_tool.callable.assert_called_once_with(
        "A dramatic cityscape at sunset", "tiktok"
    )
