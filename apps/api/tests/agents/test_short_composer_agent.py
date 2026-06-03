import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.workers.agents import AgentActionStatus
from app.workers.short_composer_agent import ShortComposerAgent


# ── Helpers ─────────────────────────────────────────────────────────


def _make_agent():
    agent = ShortComposerAgent.__new__(ShortComposerAgent)
    agent.di_tools = {}
    return agent


def _scene(
    n: int = 1,
    asset_type: str = "video_clip",
    video_url: str = "http://s3/vid.mp4",
    image_url: str = "http://s3/img.png",
    narration: str = "Test narration",
    kb_motion: str = "zoom_in",
) -> dict:
    scene = {
        "scene_number": n,
        "narration_text": narration,
        "visual_prompt": "test visual",
        "asset_type": asset_type,
        "target_duration_seconds": 5.0,
    }
    if asset_type == "video_clip":
        scene["video_url"] = video_url
    else:
        scene["image_url"] = image_url
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


def _alignment(narration: str) -> list[dict]:
    words = narration.split()
    return [
        {"word": w, "start": i * 0.5, "end": (i + 1) * 0.5} for i, w in enumerate(words)
    ]


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_ffmpeg():
    async def _mock_exec(*cmd, **kwargs):
        # Create the output file that FFmpeg would produce
        for arg in cmd:
            if isinstance(arg, str) and arg.endswith(".mp4"):
                Path(arg).parent.mkdir(parents=True, exist_ok=True)
                Path(arg).write_bytes(b"fake-final-video")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec) as m:
        yield m


@pytest.fixture(autouse=True)
def _patch_storage():
    default_alignment = _alignment("Hello world")

    def _download_side_effect(url: str) -> bytes:
        if url.endswith(".json"):
            return json.dumps(default_alignment).encode()
        return b"fake-media-bytes"

    mock_storage = MagicMock()
    mock_storage.upload_video = MagicMock(return_value="/api/proxy/videos/final.mp4")
    mock_storage.download_file = MagicMock(side_effect=_download_side_effect)

    with patch(
        "app.workers.short_composer_agent.get_storage",
        return_value=mock_storage,
    ):
        yield mock_storage


@pytest.fixture(autouse=True)
def _patch_to_thread():
    async def _to_thread(f, *a, **k):
        return f(*a, **k)

    with patch(
        "app.workers.short_composer_agent.asyncio.to_thread",
        side_effect=_to_thread,
    ):
        yield


# ── Happy paths ─────────────────────────────────────────────────────


@pytest.mark.agent
async def test_happy_path_composes_video(_patch_storage):
    jid = uuid4()
    agent = _make_agent()

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "platform": "tiktok",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["final_video_url"] == "/api/proxy/videos/final.mp4"
    assert result.metadata["total_scenes"] == 1
    assert result.metadata["platform"] == "tiktok"


@pytest.mark.agent
async def test_ken_burns_scene_includes_zoompan(_patch_storage, _patch_ffmpeg):
    agent = _make_agent()

    await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(
                [_scene(1, "ken_burns", kb_motion="zoom_in")]
            ),
            "platform": "tiktok",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )

    args, _kwargs = _patch_ffmpeg.call_args
    cmd = " ".join(args)
    assert "zoompan" in cmd


@pytest.mark.agent
async def test_multiple_scenes_concatenated(_patch_storage, _patch_ffmpeg):
    def _custom_download(url: str) -> bytes:
        if url.endswith(".json"):
            return json.dumps(_alignment("Scene one here Scene two now")).encode()
        return b"fake-media-bytes"

    _patch_storage.download_file.side_effect = _custom_download
    agent = _make_agent()

    await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(
                [
                    _scene(1, "video_clip", narration="Scene one here"),
                    _scene(2, "ken_burns", narration="Scene two now"),
                ]
            ),
            "platform": "tiktok",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )

    args, _kwargs = _patch_ffmpeg.call_args
    cmd = " ".join(args)
    assert "concat=n=2:v=1:a=0" in cmd
    assert "zoompan" in cmd
    assert "trim=duration=" in cmd


@pytest.mark.agent
async def test_subtitle_burn_in_filter(_patch_storage, _patch_ffmpeg):
    agent = _make_agent()

    await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "platform": "tiktok",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )

    args, _kwargs = _patch_ffmpeg.call_args
    cmd = " ".join(args)
    assert "ass=" in cmd
    assert "subtitles.ass" in cmd


@pytest.mark.agent
async def test_platform_resolution_in_ffmpeg(_patch_storage, _patch_ffmpeg):
    agent = _make_agent()

    await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "platform": "instagram",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )

    args, _kwargs = _patch_ffmpeg.call_args
    cmd = " ".join(args)
    assert "scale=1080:1350" in cmd


@pytest.mark.agent
async def test_uploads_final_video(_patch_storage):
    agent = _make_agent()

    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "platform": "tiktok",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
            "device_id": "dev-123",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    _patch_storage.upload_video.assert_called_once()
    args, kwargs = _patch_storage.upload_video.call_args
    assert "final_output_" in args[1]
    assert "dev-123" in kwargs.get("folder", "")


# ── Error / edge cases ──────────────────────────────────────────────


@pytest.mark.agent
async def test_returns_error_when_format_payload_not_dict():
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
async def test_returns_error_when_video_url_missing():
    agent = _make_agent()
    scene = _scene(1, "video_clip")
    del scene["video_url"]
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([scene]),
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "video_url" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_image_url_missing():
    agent = _make_agent()
    scene = _scene(1, "ken_burns")
    del scene["image_url"]
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([scene]),
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "image_url" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_voiceover_url_missing():
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "vocal_alignment_url": "http://s3/al.json",
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "voiceover_url" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_alignment_url_missing():
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "voiceover_url": "http://s3/vo.mp3",
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "vocal_alignment_url" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_alignment_empty(_patch_storage):
    _patch_storage.download_file.side_effect = lambda url: (
        json.dumps([]).encode() if url.endswith(".json") else b"fake-media-bytes"
    )
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "Empty or missing vocal_alignment_data" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_ffmpeg_fails(_patch_storage):
    agent = _make_agent()

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await agent.run(
            {
                "job_id": uuid4(),
                "format_payload": _format_payload([_scene(1, "video_clip")]),
                "platform": "tiktok",
                "voiceover_url": "http://s3/vo.mp3",
                "vocal_alignment_url": "http://s3/al.json",
            }
        )

    assert result.status == AgentActionStatus.ERROR
    assert "FFmpeg composition failed" in result.reasoning


@pytest.mark.agent
async def test_returns_error_on_ffmpeg_not_found(_patch_storage, _patch_ffmpeg):
    _patch_ffmpeg.side_effect = FileNotFoundError("ffmpeg not found")
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "platform": "tiktok",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "FFmpeg composition failed" in result.reasoning


@pytest.mark.agent
async def test_returns_error_on_download_failure(_patch_storage):
    _patch_storage.download_file.side_effect = Exception("download failed")
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "platform": "tiktok",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "download failed" in result.reasoning


@pytest.mark.agent
async def test_temp_directory_cleaned_up(_patch_storage, _patch_ffmpeg):
    jid = uuid4()
    agent = _make_agent()

    await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload([_scene(1, "video_clip")]),
            "platform": "tiktok",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )

    import os

    assert not os.path.exists(f"/tmp/job_{jid}")


@pytest.mark.agent
async def test_mixed_scenes_both_asset_types(_patch_storage, _patch_ffmpeg):
    def _custom_download(url: str) -> bytes:
        if url.endswith(".json"):
            return json.dumps(_alignment("First scene Second scene")).encode()
        return b"fake-media-bytes"

    _patch_storage.download_file.side_effect = _custom_download
    agent = _make_agent()

    await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(
                [
                    _scene(1, "video_clip", narration="First scene"),
                    _scene(2, "ken_burns", narration="Second scene"),
                ]
            ),
            "platform": "youtube",
            "voiceover_url": "http://s3/vo.mp3",
            "vocal_alignment_url": "http://s3/al.json",
        }
    )

    args, _kwargs = _patch_ffmpeg.call_args
    cmd = " ".join(args)
    assert "zoompan" in cmd
    assert "trim=duration=" in cmd
    assert "concat=n=2:v=1:a=0" in cmd
