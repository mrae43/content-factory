import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.db.crud import (
    get_latest_script,
    get_render_job,
    update_job_status,
    log_error,
    save_script,
    append_script_feedback,
    save_fact_check_claims,
    claim_next_job,
    release_job_lock,
    recover_stuck_jobs,
)
from app.schemas.shorts import JobStatusEnum


@pytest.fixture
def job_id():
    return uuid4()


@pytest.fixture
def mock_scalar_result():
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    return result


@pytest.fixture
def mock_db(mock_scalar_result):
    db = AsyncMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(return_value=mock_scalar_result)
    db.add = MagicMock()
    db.add_all = MagicMock()
    return db


@pytest.fixture
def mock_script():
    script = MagicMock()
    script.feedback_history = []
    script.version = 1
    script.content = "test script content"
    return script


@pytest.fixture
def mock_render_job():
    job = MagicMock()
    job.error_log = None
    job.locked_at = None
    job.locked_by = None
    return job


@pytest.mark.unit
class TestGetLatestScript:
    async def test_should_return_script_when_one_exists(
        self, mock_db, mock_scalar_result, mock_script, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = mock_script

        result = await get_latest_script(mock_db, job_id)

        assert result == mock_script

    async def test_should_return_none_when_no_scripts_found(
        self, mock_db, mock_scalar_result, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = None

        result = await get_latest_script(mock_db, job_id)

        assert result is None


@pytest.mark.unit
class TestGetRenderJob:
    async def test_should_return_job_when_found(
        self, mock_db, mock_scalar_result, mock_render_job, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = mock_render_job

        result = await get_render_job(mock_db, job_id)

        assert result == mock_render_job

    async def test_should_return_none_when_job_not_found(
        self, mock_db, mock_scalar_result, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = None

        result = await get_render_job(mock_db, job_id)

        assert result is None


@pytest.mark.unit
class TestUpdateJobStatus:
    async def test_should_execute_update_and_commit(self, mock_db, job_id):
        await update_job_status(mock_db, job_id, JobStatusEnum.SCRIPTING)

        mock_db.execute.assert_awaited_once()
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestLogError:
    async def test_should_append_error_to_existing_error_log(
        self, mock_db, mock_scalar_result, mock_render_job, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = mock_render_job
        mock_render_job.error_log = {
            "phase1": {"message": "old error", "timestamp": "t"}
        }

        await log_error(mock_db, job_id, "boom", "research")

        assert "research" in mock_render_job.error_log
        assert mock_render_job.error_log["research"]["message"] == "boom"
        mock_db.commit.assert_awaited_once()

    async def test_should_initialize_error_log_when_none(
        self, mock_db, mock_scalar_result, mock_render_job, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = mock_render_job
        mock_render_job.error_log = None

        await log_error(mock_db, job_id, "boom", "research")

        assert mock_render_job.error_log is not None
        assert "research" in mock_render_job.error_log
        assert mock_render_job.error_log["research"]["message"] == "boom"
        mock_db.commit.assert_awaited_once()

    async def test_should_do_nothing_when_job_not_found(
        self, mock_db, mock_scalar_result, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = None

        await log_error(mock_db, job_id, "boom", "research")

        mock_db.commit.assert_not_awaited()


@pytest.mark.unit
class TestSaveScript:
    async def test_should_create_script_and_commit(self, mock_db, job_id):
        await save_script(mock_db, job_id, "script content", 1)

        mock_db.add.assert_called_once()
        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.content == "script content"
        assert added_obj.version == 1
        assert added_obj.is_approved is False
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestAppendScriptFeedback:
    async def test_should_append_feedback_to_existing_history(
        self, mock_db, mock_scalar_result, mock_script, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = mock_script
        mock_script.feedback_history = ["old feedback"]

        await append_script_feedback(mock_db, job_id, "new feedback")

        assert mock_script.feedback_history == ["old feedback", "new feedback"]
        mock_db.commit.assert_awaited_once()

    async def test_should_initialize_list_when_feedback_history_is_none(
        self, mock_db, mock_scalar_result, mock_script, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = mock_script
        mock_script.feedback_history = None

        await append_script_feedback(mock_db, job_id, "first feedback")

        assert mock_script.feedback_history == ["first feedback"]
        mock_db.commit.assert_awaited_once()

    async def test_should_do_nothing_when_no_script_exists(
        self, mock_db, mock_scalar_result, job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = None

        await append_script_feedback(mock_db, job_id, "feedback")

        mock_db.commit.assert_not_awaited()


@pytest.mark.unit
class TestSaveFactCheckClaims:
    async def test_should_create_claim_rows_and_flush(self, mock_db, job_id):
        script_id = uuid4()
        claims = [
            {
                "claim_text": "GDP grew 15%",
                "verdict": "UNSUPPORTED",
                "confidence": 0.3,
                "evidence_references": ["ref1", "ref2"],
            }
        ]

        await save_fact_check_claims(mock_db, script_id, claims)

        mock_db.add_all.assert_called_once()
        added_rows = mock_db.add_all.call_args[0][0]
        assert len(added_rows) == 1
        assert added_rows[0].claim_text == "GDP grew 15%"
        assert added_rows[0].script_id == script_id
        mock_db.flush.assert_awaited_once()
        mock_db.commit.assert_not_awaited()

    async def test_should_handle_empty_claims_list(self, mock_db, job_id):
        script_id = uuid4()

        await save_fact_check_claims(mock_db, script_id, [])

        mock_db.add_all.assert_called_once_with([])
        mock_db.flush.assert_awaited_once()

    async def test_should_default_evidence_references_to_empty_list(
        self, mock_db, job_id
    ):
        script_id = uuid4()
        claims = [
            {
                "claim_text": "some claim",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
            }
        ]

        await save_fact_check_claims(mock_db, script_id, claims)

        added_rows = mock_db.add_all.call_args[0][0]
        assert added_rows[0].evidence_references == []


@pytest.mark.unit
class TestClaimNextJob:
    async def test_should_lock_and_return_available_job(
        self, mock_db, mock_scalar_result, mock_render_job
    ):
        mock_scalar_result.scalar_one_or_none.return_value = mock_render_job

        result = await claim_next_job(mock_db, "worker-1")

        assert result == mock_render_job
        assert mock_render_job.locked_at is not None
        assert mock_render_job.locked_by == "worker-1"
        mock_db.commit.assert_awaited_once()

    async def test_should_return_none_when_no_jobs_available(
        self, mock_db, mock_scalar_result
    ):
        mock_scalar_result.scalar_one_or_none.return_value = None

        result = await claim_next_job(mock_db, "worker-1")

        assert result is None
        mock_db.commit.assert_not_awaited()


@pytest.mark.unit
class TestReleaseJobLock:
    async def test_should_clear_lock_fields_and_commit(self, mock_db, job_id):
        await release_job_lock(mock_db, job_id)

        mock_db.execute.assert_awaited_once()
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestRecoverStuckJobs:
    async def test_should_release_locks_on_stale_jobs(self, mock_db):
        await recover_stuck_jobs(mock_db, timeout_minutes=15)

        mock_db.execute.assert_awaited_once()
        mock_db.commit.assert_awaited_once()
