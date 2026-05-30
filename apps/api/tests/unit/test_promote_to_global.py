import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


def _make_mock_job(job_id=None):
    job = MagicMock()
    job.id = job_id or uuid4()
    job.title = "BRICS De-dollarization"
    job.status = "COMPLETED"
    return job


def _make_mock_script(content="Test script content.", script_id=None):
    script = MagicMock()
    script.id = script_id or uuid4()
    script.content = content
    script.version = 1
    return script


def _make_mock_claim(claim_text, verdict="SUPPORTED"):
    return {
        "claim_text": claim_text,
        "verdict": verdict,
        "evidence_text": "Source: test",
        "evidence_references": [],
        "hedge_required": False,
    }


@pytest.mark.unit
async def test_promote_to_global_skips_when_no_script():
    from app.workers.orchestrator import _promote_to_global

    db = AsyncMock()
    job = _make_mock_job()

    with (
        patch("app.workers.orchestrator.get_latest_script", return_value=None),
        patch("app.workers.orchestrator.get_llm") as mock_get_llm,
    ):
        await _promote_to_global(db, job)

    mock_get_llm.assert_not_called()


@pytest.mark.unit
async def test_promote_to_global_skips_when_no_supported_claims():
    from app.workers.orchestrator import _promote_to_global

    db = AsyncMock()
    job = _make_mock_job()
    script = _make_mock_script()

    with (
        patch("app.workers.orchestrator.get_latest_script", return_value=script),
        patch("app.workers.orchestrator.get_script_claims", return_value=[]),
        patch("app.workers.orchestrator.get_llm") as mock_get_llm,
    ):
        await _promote_to_global(db, job)

    mock_get_llm.assert_not_called()


@pytest.mark.unit
async def test_promote_to_global_ingests_compressed_facts():
    from app.workers.orchestrator import _promote_to_global

    job = _make_mock_job()
    db = AsyncMock()
    script = _make_mock_script(content="BRICS economic outlook.")
    claims = [
        _make_mock_claim("BRICS GDP grew 3.2% in 2024."),
        _make_mock_claim("China GDP 5.2% in Q3 2024."),
    ]

    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = (
        "- BRICS GDP grew 3.2% in 2024.\n- China GDP was 5.2% in Q3 2024."
    )
    mock_llm.ainvoke.return_value = mock_response

    with (
        patch("app.workers.orchestrator.get_latest_script", return_value=script),
        patch("app.workers.orchestrator.get_script_claims", return_value=claims),
        patch("app.workers.orchestrator.get_llm", return_value=mock_llm),
        patch("app.workers.orchestrator._get_vector_store") as mock_get_vs,
    ):
        mock_vs = AsyncMock()
        mock_vs.ingest_chunks = AsyncMock(return_value=2)
        mock_get_vs.return_value = mock_vs

        await _promote_to_global(db, job)

    mock_llm.ainvoke.assert_awaited_once()
    args, kwargs = mock_vs.ingest_chunks.call_args
    assert kwargs["job_id"] is None
    assert kwargs["chunks"] == [
        "BRICS GDP grew 3.2% in 2024.",
        "China GDP was 5.2% in Q3 2024.",
    ]
    assert kwargs["scope"] == "GLOBAL"
    assert kwargs["meta"]["source_job_id"] == str(job.id)
    assert kwargs["meta"]["source_title"] == job.title
    assert kwargs["meta"]["source_type"] == "COMPRESSED_FACT"
    assert "ingested_at" in kwargs["meta"]


@pytest.mark.unit
async def test_promote_to_global_logs_and_continues_on_error():
    from app.workers.orchestrator import _promote_to_global

    job = _make_mock_job()
    db = AsyncMock()
    script = _make_mock_script(content="Test script.")
    claims = [_make_mock_claim("Test supported claim.")]

    with (
        patch("app.workers.orchestrator.get_latest_script", return_value=script),
        patch("app.workers.orchestrator.get_script_claims", return_value=claims),
        patch("app.workers.orchestrator.get_llm", side_effect=Exception("LLM failed")),
        patch("app.workers.orchestrator.logger") as mock_logger,
    ):
        await _promote_to_global(db, job)

    mock_logger.exception.assert_called_once()
