import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.db.models import ResearchChunk


def _make_mock_job(job_id=None):
    job = MagicMock()
    job.id = job_id or uuid4()
    job.title = "BRICS De-dollarization"
    job.status = "COMPLETED"
    return job


def _make_mock_chunk(content="Test fact.", scope="LOCAL"):
    chunk = MagicMock(spec=ResearchChunk)
    chunk.content = content
    chunk.meta = {"scope": scope, "version": "1.0"}
    return chunk


@pytest.mark.unit
async def test_promote_to_global_skips_when_no_local_chunks():
    from app.workers.orchestrator import _promote_to_global

    db = AsyncMock()
    job = _make_mock_job()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    db.execute.return_value = mock_result

    with patch("app.workers.orchestrator.get_llm") as mock_get_llm:
        await _promote_to_global(db, job)

    mock_get_llm.assert_not_called()


@pytest.mark.unit
async def test_promote_to_global_ingests_compressed_facts():
    from app.workers.orchestrator import _promote_to_global

    job = _make_mock_job()
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        _make_mock_chunk("BRICS GDP grew 3.2% in 2024."),
        _make_mock_chunk("China GDP 5.2% in Q3 2024."),
    ]
    db.execute.return_value = mock_result

    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = (
        "- BRICS GDP grew 3.2% in 2024.\n- China GDP was 5.2% in Q3 2024."
    )
    mock_llm.ainvoke.return_value = mock_response

    with (
        patch("app.workers.orchestrator.get_llm", return_value=mock_llm),
        patch("app.workers.orchestrator._get_vector_store") as mock_get_vs,
    ):
        mock_vs = AsyncMock()
        mock_vs.ingest_chunks = AsyncMock(return_value=2)
        mock_get_vs.return_value = mock_vs

        await _promote_to_global(db, job)

    mock_llm.ainvoke.assert_awaited_once()
    mock_vs.ingest_chunks.assert_awaited_once_with(
        job_id=None,
        chunks=["BRICS GDP grew 3.2% in 2024.", "China GDP was 5.2% in Q3 2024."],
        scope="GLOBAL",
        meta={"source_job_id": str(job.id), "source_title": job.title},
    )


@pytest.mark.unit
async def test_promote_to_global_logs_and_continues_on_error():
    from app.workers.orchestrator import _promote_to_global

    job = _make_mock_job()
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        _make_mock_chunk("Test fact."),
    ]
    db.execute.return_value = mock_result

    with (
        patch("app.workers.orchestrator.get_llm", side_effect=Exception("LLM failed")),
        patch("app.workers.orchestrator.logger") as mock_logger,
    ):
        await _promote_to_global(db, job)

    mock_logger.exception.assert_called_once()
