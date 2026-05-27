from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.workers.agents import AgentActionStatus
from app.workers.carousel_image_agent import CarouselImageAgent
from app.services.tools import Tool


def _make_image_result(
    success: bool = True,
    image_bytes: bytes | None = b"img",
    failure_reason: str | None = None,
) -> dict:
    return {
        "success": success,
        "image_bytes": image_bytes,
        "width": 1088,
        "height": 1344,
        "failure_reason": failure_reason,
        "prompt_used": "test prompt",
    }


def _make_gen_tool(side_effect=None):
    mock_gen = MagicMock()
    if side_effect is not None:
        mock_gen.callable = AsyncMock(side_effect=side_effect)
    else:
        mock_gen.callable = AsyncMock(return_value=_make_image_result())
    return Tool(
        name="generate_image",
        description="Test gen",
        callable=mock_gen.callable,
        permissions={"*"},
    )


def _make_upload_tool(return_url="/api/proxy/images/slide_01.png"):
    mock_up = MagicMock()
    mock_up.callable = AsyncMock(return_value=return_url)
    return Tool(
        name="upload_image",
        description="Test upload",
        callable=mock_up.callable,
        permissions={"*"},
    )


def _make_agent(gen_tool=None, upload_tool=None):
    agent = CarouselImageAgent.__new__(CarouselImageAgent)
    agent.di_tools = {}
    if gen_tool:
        agent.di_tools["generate_image"] = gen_tool
    if upload_tool:
        agent.di_tools["upload_image"] = upload_tool
    return agent


def _slide(n: int, visual: str = "Chart", text: str = "Text") -> dict:
    return {"slide_number": n, "visual_description": visual, "text": text}


@pytest.mark.agent
async def test_generates_images_for_all_slides():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {
            "slides": [_slide(1), _slide(2)],
            "thread_title": "Test",
        },
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.SUCCESS
    slides = result.payload["format_payload"]["slides"]
    assert slides[0]["image_url"] == "/api/proxy/images/slide_01.png"
    assert slides[1]["image_url"] == "/api/proxy/images/slide_01.png"
    assert gen_tool.callable.call_count == 2
    assert upload_tool.callable.call_count == 2
    assert result.metadata["successful_slides"] == 2


@pytest.mark.agent
async def test_honours_platform_arg():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1)]},
        "platform": "linkedin",
    }

    await agent.run(context)

    gen_tool.callable.assert_called_once_with("Chart", "linkedin")


@pytest.mark.agent
async def test_returns_error_when_no_slides():
    agent = _make_agent()
    result = await agent.run(
        {
            "job_id": uuid4(),
            "format_payload": {"slides": []},
        }
    )

    assert result.status == AgentActionStatus.ERROR
    assert "No slides" in result.reasoning


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
async def test_partial_failure_still_succeeds():
    jid = uuid4()
    gen_tool = _make_gen_tool(
        side_effect=[
            _make_image_result(success=True, image_bytes=b"ok"),
            _make_image_result(success=False, image_bytes=None, failure_reason="API error"),
        ]
    )
    upload_tool = _make_upload_tool(return_url="/static/ok.png")

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1), _slide(2)]},
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.SUCCESS
    slides = result.payload["format_payload"]["slides"]
    assert slides[0]["image_url"] is not None
    assert slides[1]["image_url"] is None
    assert result.metadata["failed_slides"] == 1
    assert result.metadata["successful_slides"] == 1


@pytest.mark.agent
async def test_returns_error_when_all_slides_fail():
    jid = uuid4()
    gen_tool = _make_gen_tool(
        side_effect=[
            _make_image_result(success=False, image_bytes=None, failure_reason="API error"),
        ]
    )
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1)]},
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.ERROR
    assert result.metadata["failed_slides"] == 1


@pytest.mark.agent
async def test_skips_slide_without_visual_description():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1, visual="")]},
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert gen_tool.callable.call_count == 0
    assert upload_tool.callable.call_count == 0
    assert result.payload["format_payload"]["slides"][0]["image_url"] is None


@pytest.mark.agent
async def test_filename_and_folder_include_ids():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1)]},
        "platform": "instagram",
        "device_id": "test-device",
    }

    await agent.run(context)

    args, kwargs = upload_tool.callable.call_args
    filename = args[1]
    folder = kwargs.get("folder", args[2] if len(args) > 2 else "")
    assert filename == "slide_01.png"
    assert str(jid) in folder
    assert "test-device" in folder


@pytest.mark.agent
async def test_metadata_tracks_failures():
    jid = uuid4()
    gen_tool = _make_gen_tool(
        side_effect=[
            _make_image_result(success=True, image_bytes=b"a"),
            _make_image_result(success=False, image_bytes=None, failure_reason="Rate limited"),
        ]
    )
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1), _slide(2)]},
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert len(result.metadata["failures"]) == 1
    assert result.metadata["failures"][0]["slide_number"] == 2
    assert result.metadata["failures"][0]["reason"] == "Rate limited"


@pytest.mark.agent
async def test_missing_job_id_still_runs():
    gen_tool = _make_gen_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "format_payload": {"slides": [_slide(1)]},
        "platform": "instagram",
        "device_id": "anon-device",
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert gen_tool.callable.call_count == 1


@pytest.mark.agent
async def test_defaults_platform_to_instagram():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1)]},
    }

    await agent.run(context)

    gen_tool.callable.assert_called_once_with("Chart", "instagram")


@pytest.mark.agent
async def test_exposes_success_count_in_metadata():
    jid = uuid4()
    gen_tool = _make_gen_tool()
    upload_tool = _make_upload_tool()

    agent = _make_agent(gen_tool=gen_tool, upload_tool=upload_tool)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1), _slide(2), _slide(3)]},
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert result.metadata["total_slides"] == 3
    assert result.metadata["successful_slides"] == 3
    assert result.metadata["failed_slides"] == 0


@pytest.mark.agent
async def test_returns_error_when_gen_tool_missing():
    agent = _make_agent(upload_tool=_make_upload_tool())
    context = {
        "job_id": uuid4(),
        "format_payload": {"slides": [_slide(1)]},
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.ERROR
    assert "generate_image" in result.reasoning


@pytest.mark.agent
async def test_returns_error_when_upload_tool_missing():
    agent = _make_agent(gen_tool=_make_gen_tool())
    context = {
        "job_id": uuid4(),
        "format_payload": {"slides": [_slide(1)]},
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.ERROR
    assert "upload_image" in result.reasoning
