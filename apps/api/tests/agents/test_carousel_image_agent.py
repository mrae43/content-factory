from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.workers.agents import AgentActionStatus
from app.workers.carousel_image_agent import CarouselImageAgent


def _make_agent(mock_image_service=None, mock_storage=None):
    with patch("app.workers.carousel_image_agent.get_storage") as mock_get_storage:
        mock_get_storage.return_value = mock_storage or MagicMock()
        agent = CarouselImageAgent.__new__(CarouselImageAgent)
        agent.image_service = mock_image_service or MagicMock()
        agent.storage = mock_get_storage.return_value
        return agent


def _slide(n: int, visual: str = "Chart", text: str = "Text") -> dict:
    return {"slide_number": n, "visual_description": visual, "text": text}


@pytest.mark.agent
async def test_generates_images_for_all_slides():
    jid = uuid4()
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        return_value=MagicMock(success=True, image_bytes=b"img", failure_reason=None)
    )
    mock_store = MagicMock()
    mock_store.upload_image.return_value = "/static/carousel_images/slide_01.png"

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
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
    assert slides[0]["image_url"] == "/static/carousel_images/slide_01.png"
    assert slides[1]["image_url"] == "/static/carousel_images/slide_01.png"
    assert mock_gen.generate.call_count == 2
    assert mock_store.upload_image.call_count == 2
    assert result.metadata["successful_slides"] == 2


@pytest.mark.agent
async def test_honours_platform_arg():
    jid = uuid4()
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        return_value=MagicMock(success=True, image_bytes=b"img", failure_reason=None)
    )
    mock_store = MagicMock()
    mock_store.upload_image.return_value = "/static/carousel_images/slide.png"

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1)]},
        "platform": "linkedin",
    }

    await agent.run(context)

    mock_gen.generate.assert_called_once_with("Chart", "linkedin")


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
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        side_effect=[
            MagicMock(success=True, image_bytes=b"ok", failure_reason=None),
            MagicMock(success=False, image_bytes=None, failure_reason="API error"),
        ]
    )
    mock_store = MagicMock()
    mock_store.upload_image.return_value = "/static/ok.png"

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
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
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        return_value=MagicMock(
            success=False, image_bytes=None, failure_reason="API error"
        )
    )

    agent = _make_agent(mock_image_service=mock_gen)
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
    mock_gen = MagicMock()
    mock_store = MagicMock()

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1, visual="")]},
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert mock_gen.generate.call_count == 0
    assert mock_store.upload_image.call_count == 0
    assert result.payload["format_payload"]["slides"][0]["image_url"] is None


@pytest.mark.agent
async def test_filename_includes_job_id():
    jid = uuid4()
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        return_value=MagicMock(success=True, image_bytes=b"data", failure_reason=None)
    )
    mock_store = MagicMock()

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1)]},
        "platform": "instagram",
    }

    await agent.run(context)

    filename = mock_store.upload_image.call_args[0][1]
    assert str(jid) in filename
    assert filename.endswith(".png")


@pytest.mark.agent
async def test_metadata_tracks_failures():
    jid = uuid4()
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        side_effect=[
            MagicMock(success=True, image_bytes=b"a", failure_reason=None),
            MagicMock(success=False, image_bytes=None, failure_reason="Rate limited"),
        ]
    )
    mock_store = MagicMock()

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
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
    """job_id is only used for the filename, so it can be absent."""
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        return_value=MagicMock(success=True, image_bytes=b"data", failure_reason=None)
    )
    mock_store = MagicMock()

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
    context = {
        "format_payload": {"slides": [_slide(1)]},
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert result.status == AgentActionStatus.SUCCESS
    assert mock_gen.generate.call_count == 1


@pytest.mark.agent
async def test_defaults_platform_to_instagram():
    jid = uuid4()
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        return_value=MagicMock(success=True, image_bytes=b"data", failure_reason=None)
    )
    mock_store = MagicMock()

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1)]},
    }

    await agent.run(context)

    mock_gen.generate.assert_called_once_with("Chart", "instagram")


@pytest.mark.agent
async def test_exposes_success_count_in_metadata():
    jid = uuid4()
    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(
        return_value=MagicMock(success=True, image_bytes=b"img", failure_reason=None)
    )
    mock_store = MagicMock()

    agent = _make_agent(mock_image_service=mock_gen, mock_storage=mock_store)
    context = {
        "job_id": jid,
        "format_payload": {"slides": [_slide(1), _slide(2), _slide(3)]},
        "platform": "instagram",
    }

    result = await agent.run(context)

    assert result.metadata["total_slides"] == 3
    assert result.metadata["successful_slides"] == 3
    assert result.metadata["failed_slides"] == 0
