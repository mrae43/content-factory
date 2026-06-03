from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.workers.agents import AgentActionStatus
from app.workers.short_voiceover_agent import ShortVoiceoverAgent
from app.services.tools import Tool


def _make_gen_voiceover_tool(
    audio_bytes: bytes = b"fake-audio",
    vocal_alignment: list | None = None,
    duration: float = 12.5,
    side_effect=None,
):
    if vocal_alignment is None:
        vocal_alignment = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
    mock_gen = MagicMock()
    if side_effect is not None:
        mock_gen.callable = AsyncMock(side_effect=side_effect)
    else:
        mock_gen.callable = AsyncMock(
            return_value={
                "audio_bytes": audio_bytes,
                "vocal_alignment_data": vocal_alignment,
                "duration_seconds": duration,
            }
        )
    return Tool(
        name="generate_voiceover",
        description="Test generate voiceover",
        callable=mock_gen.callable,
        permissions={"*"},
    )


def _make_upload_voiceover_tool(return_url: str = "/api/proxy/voiceovers/vo.mp3"):
    mock_up = MagicMock()
    mock_up.callable = AsyncMock(return_value=return_url)
    return Tool(
        name="upload_voiceover",
        description="Test upload voiceover",
        callable=mock_up.callable,
        permissions={"*"},
    )


def _make_get_alignment_tool():
    mock_al = MagicMock()
    mock_al.callable = AsyncMock(
        return_value={
            "message": "Alignment data is inline.",
            "vocal_alignment_data": [],
        }
    )
    return Tool(
        name="get_alignment",
        description="Test get alignment",
        callable=mock_al.callable,
        permissions={"*"},
    )


def _make_agent(
    gen_tool=None,
    upload_tool=None,
    alignment_tool=None,
):
    agent = ShortVoiceoverAgent.__new__(ShortVoiceoverAgent)
    agent.di_tools = {}
    if gen_tool:
        agent.di_tools["generate_voiceover"] = gen_tool
    if upload_tool:
        agent.di_tools["upload_voiceover"] = upload_tool
    if alignment_tool:
        agent.di_tools["get_alignment"] = alignment_tool
    return agent


def _scene(n: int = 1, narration: str = "The city never sleeps.") -> dict:
    return {
        "scene_number": n,
        "narration_text": narration,
        "visual_prompt": "A dramatic cityscape at sunset",
        "asset_type": "video_clip",
        "target_duration_seconds": 5.0,
    }


def _format_payload(
    scenes: list[dict],
    voice_id: str = "test-voice",
) -> dict:
    return {
        "format": "short",
        "version": 1,
        "scenes": scenes,
        "target_total_duration": 30.0,
        "visual_style": "cinematic",
        "audio_direction": "upbeat",
        "music_mood": "synthwave_hype",
        "voice_id": voice_id,
        "subtitle_preset": "CENTER_POP_YELLOW",
    }


# ───────────────────────────
# Happy paths
# ───────────────────────────


@pytest.mark.agent
async def test_happy_path_generates_voiceover_and_uploads():
    jid = uuid4()
    scenes = [_scene(1, "Hello world"), _scene(2, "This is a test")]
    gen_tool = _make_gen_voiceover_tool()
    upload_tool = _make_upload_voiceover_tool()
    agent = _make_agent(
        gen_tool=gen_tool,
        upload_tool=upload_tool,
        alignment_tool=_make_get_alignment_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
            "device_id": "dev-123",
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["voiceover_url"] == "/api/proxy/voiceovers/vo.mp3"
    assert result.payload["duration_seconds"] == 12.5
    assert result.metadata["total_scenes"] == 2
    assert result.metadata["text_length"] > 0
    gen_tool.callable.assert_awaited_once()
    assert upload_tool.callable.call_count == 2  # audio + alignment


@pytest.mark.agent
async def test_avoids_double_punctuation_when_narration_ends_with_period():
    jid = uuid4()
    scenes = [
        _scene(1, "First sentence."),
        _scene(2, "Second sentence."),
    ]
    gen_tool = _make_gen_voiceover_tool()
    agent = _make_agent(
        gen_tool=gen_tool,
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )

    await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
        }
    )

    args, kwargs = gen_tool.callable.call_args
    assert kwargs["text"] == "First sentence. Second sentence."


@pytest.mark.agent
async def test_preserves_question_mark_when_narration_ends_with_question():
    jid = uuid4()
    scenes = [
        _scene(1, "How are you?"),
        _scene(2, "I am fine."),
    ]
    gen_tool = _make_gen_voiceover_tool()
    agent = _make_agent(
        gen_tool=gen_tool,
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )

    await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
        }
    )

    args, kwargs = gen_tool.callable.call_args
    assert kwargs["text"] == "How are you? I am fine."


@pytest.mark.agent
async def test_adds_period_when_narration_lacks_terminal_punctuation():
    jid = uuid4()
    scenes = [
        _scene(1, "First sentence"),
        _scene(2, "Second sentence"),
    ]
    gen_tool = _make_gen_voiceover_tool()
    agent = _make_agent(
        gen_tool=gen_tool,
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )

    await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
        }
    )

    args, kwargs = gen_tool.callable.call_args
    assert kwargs["text"] == "First sentence. Second sentence"


@pytest.mark.agent
async def test_concatenates_narration_text_with_separator():
    jid = uuid4()
    scenes = [
        _scene(1, "First sentence"),
        _scene(2, "Second sentence"),
    ]
    gen_tool = _make_gen_voiceover_tool()
    agent = _make_agent(
        gen_tool=gen_tool,
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )

    await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
        }
    )

    args, kwargs = gen_tool.callable.call_args
    assert kwargs["text"] == "First sentence. Second sentence"


@pytest.mark.agent
async def test_uploads_vocal_alignment_json():
    jid = uuid4()
    scenes = [_scene(1, "Only scene")]
    upload_tool = _make_upload_voiceover_tool()
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=upload_tool,
        alignment_tool=_make_get_alignment_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["vocal_alignment_url"] is not None
    calls = upload_tool.callable.call_args_list
    assert len(calls) == 2
    # Second call should be alignment JSON
    _, kwargs = calls[1]
    assert kwargs.get("content_type") == "application/json"


@pytest.mark.agent
async def test_ignores_empty_narration_text():
    jid = uuid4()
    scenes = [
        _scene(1, "Valid narration"),
        _scene(2, ""),
        _scene(3, "Another valid"),
    ]
    gen_tool = _make_gen_voiceover_tool()
    agent = _make_agent(
        gen_tool=gen_tool,
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )

    await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload(scenes),
        }
    )

    _, kwargs = gen_tool.callable.call_args
    assert kwargs["text"] == "Valid narration. Another valid"


# ───────────────────────────
# Error / edge cases
# ───────────────────────────


@pytest.mark.agent
async def test_returns_error_when_format_payload_not_dict():
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )
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
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "No scenes" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_voice_id_missing():
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )
    payload = _format_payload([_scene(1)])
    del payload["voice_id"]
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": payload,
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "voice_id" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_generate_voiceover_tool_missing():
    agent = _make_agent(
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1)]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "generate_voiceover" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_upload_voiceover_tool_missing():
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1)]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "upload_voiceover" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_get_alignment_tool_missing():
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=_make_upload_voiceover_tool(),
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1)]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "get_alignment" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_generate_voiceover_raises():
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(side_effect=Exception("TTS API down")),
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1)]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "TTS API down" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_generate_voiceover_returns_no_audio():
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(audio_bytes=b""),
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1)]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "no audio_bytes" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_upload_voiceover_raises():
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=_make_upload_voiceover_tool(
            return_url="/api/proxy/voiceovers/vo.mp3"
        ),
        alignment_tool=_make_get_alignment_tool(),
    )
    # Make the upload tool raise on first call
    agent.di_tools["upload_voiceover"].callable = AsyncMock(
        side_effect=Exception("S3 error")
    )
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload([_scene(1)]),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "S3 error" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_all_scenes_have_empty_narration():
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )
    scenes = [
        _scene(1, ""),
        _scene(2, ""),
    ]
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": _format_payload(scenes),
        }
    )
    assert result.status == AgentActionStatus.ERROR
    assert "No narration_text" in result.reasoning


@pytest.mark.agent
async def test_filename_and_folder_include_ids():
    jid = uuid4()
    upload_tool = _make_upload_voiceover_tool()
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=upload_tool,
        alignment_tool=_make_get_alignment_tool(),
    )

    await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload([_scene(1)]),
            "device_id": "test-device",
        }
    )

    calls = upload_tool.callable.call_args_list
    # First call: audio
    args, kwargs = calls[0]
    filename = args[1]
    folder = kwargs.get("folder", args[2] if len(args) > 2 else "")
    assert filename == f"voiceover_{jid}.mp3"
    assert str(jid) in folder
    assert "test-device" in folder
    # Second call: alignment
    args2, kwargs2 = calls[1]
    filename2 = args2[1]
    assert filename2 == f"vocal_alignment_{jid}.json"


@pytest.mark.agent
async def test_duration_seconds_in_payload():
    jid = uuid4()
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(duration=8.3),
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload([_scene(1)]),
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.payload["duration_seconds"] == 8.3
    assert result.metadata["duration_seconds"] == 8.3


@pytest.mark.agent
async def test_single_scene_success():
    jid = uuid4()
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=_make_upload_voiceover_tool(),
        alignment_tool=_make_get_alignment_tool(),
    )

    result = await agent.run(
        {
            "job_id": jid,
            "format_payload": _format_payload([_scene(1, "Only one scene here.")]),
        }
    )

    assert result.status == AgentActionStatus.SUCCESS
    assert result.metadata["total_scenes"] == 1
    _, kwargs = agent.di_tools["generate_voiceover"].callable.call_args
    assert kwargs["text"] == "Only one scene here."


@pytest.mark.agent
async def test_defaults_device_and_job_id_in_folder():
    upload_tool = _make_upload_voiceover_tool()
    agent = _make_agent(
        gen_tool=_make_gen_voiceover_tool(),
        upload_tool=upload_tool,
        alignment_tool=_make_get_alignment_tool(),
    )

    await agent.run(
        {
            "format_payload": _format_payload([_scene(1)]),
        }
    )

    args, kwargs = upload_tool.callable.call_args_list[0]
    folder = kwargs.get("folder", args[2] if len(args) > 2 else "")
    assert "__anonymous__" in folder
    assert "standalone" in folder
