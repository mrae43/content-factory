import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.db.format_crud import (
    create_format_job,
    claim_next_format_job,
    release_format_job_lock,
    recover_stuck_format_jobs,
    get_format_job,
    update_format_job_status,
    log_format_job_error,
    update_format_job_format_payload,
    update_format_job_video_url,
    update_format_job_working_memory,
    get_format_jobs_for_watcher,
    get_format_jobs_missed_terminal,
    reset_format_job_to_composition,
)
from app.schemas.shorts import FormatJobStatusEnum


@pytest.fixture
def mock_scalar_result():
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    result.scalars.return_value.all.return_value = []
    return result


@pytest.fixture
def mock_db(mock_scalar_result):
    db = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.execute = AsyncMock(return_value=mock_scalar_result)
    db.add = MagicMock()
    return db


@pytest.fixture
def source_job_id():
    return uuid4()


@pytest.fixture
def format_job_id():
    return uuid4()


@pytest.fixture
def sample_snapshot():
    return {
        "title": "Test Script",
        "script_content": "This is a script",
        "claims": [{"claim_text": "test", "verdict": "SUPPORTED"}],
        "refined_context": "refined",
        "story_directives": {"tone": "analytical"},
        "hedge_index": [],
        "epistemic_ledger": None,
    }


@pytest.mark.unit
class TestCreateFormatJob:
    async def test_creates_and_returns_job(
        self, mock_db, mock_scalar_result, source_job_id, sample_snapshot
    ):
        mock_db.execute.return_value = mock_scalar_result

        await create_format_job(
            mock_db,
            source_job_id=source_job_id,
            platform="instagram",
            format_type="carousel",
            snapshot_data=sample_snapshot,
        )

        mock_db.add.assert_called_once()
        added = mock_db.add.call_args[0][0]
        assert added.title == "Test Script"
        assert added.source_job_id == source_job_id
        assert added.platform == "instagram"
        assert added.format_type == "carousel"
        assert added.script_content == "This is a script"
        mock_db.commit.assert_awaited_once()

    async def test_returns_existing_on_duplicate(
        self, mock_db, mock_scalar_result, source_job_id, sample_snapshot
    ):
        existing = MagicMock()
        existing.id = uuid4()

        from sqlalchemy.exc import IntegrityError

        mock_db.commit.side_effect = [
            IntegrityError(
                "mock", "mock", Exception("uq_format_jobs_source_platform_type")
            ),
            None,
        ]
        mock_db.refresh = AsyncMock()
        mock_scalar_result.scalar_one.return_value = existing

        result = await create_format_job(
            mock_db,
            source_job_id=source_job_id,
            platform="instagram",
            format_type="carousel",
            snapshot_data=sample_snapshot,
        )

        assert result == existing
        assert result.status == FormatJobStatusEnum.PENDING
        assert result.error_log is None
        assert result.format_payload is None
        assert result.locked_at is None
        assert result.locked_by is None
        mock_db.refresh.assert_awaited_once_with(existing)

    async def test_empty_claims_in_snapshot(
        self, mock_db, mock_scalar_result, source_job_id
    ):
        mock_db.execute.return_value = mock_scalar_result

        await create_format_job(
            mock_db,
            source_job_id=source_job_id,
            platform="instagram",
            format_type="carousel",
            snapshot_data={"script_content": "content"},
        )

        added = mock_db.add.call_args[0][0]
        assert added.script_content == "content"
        assert added.claims is None

    async def test_reraises_unexpected_integrity_error(
        self, mock_db, mock_scalar_result, source_job_id, sample_snapshot
    ):
        from sqlalchemy.exc import IntegrityError

        mock_db.commit.side_effect = IntegrityError(
            "mock", "mock", Exception("not_null_violation")
        )
        mock_scalar_result.scalar_one.side_effect = Exception("should not reach")

        with pytest.raises(IntegrityError):
            await create_format_job(
                mock_db,
                source_job_id=source_job_id,
                platform="instagram",
                format_type="carousel",
                snapshot_data=sample_snapshot,
            )


@pytest.mark.unit
class TestClaimNextFormatJob:
    async def test_locks_and_returns_available_job(self, mock_db, mock_scalar_result):
        mock_job = MagicMock()
        mock_job.locked_at = None
        mock_job.locked_by = None
        mock_scalar_result.scalar_one_or_none.return_value = mock_job

        result = await claim_next_format_job(mock_db, "worker-1")

        assert result == mock_job
        assert mock_job.locked_at is not None
        assert mock_job.locked_by == "worker-1"
        mock_db.commit.assert_awaited_once()

    async def test_returns_none_when_no_jobs(self, mock_db, mock_scalar_result):
        mock_scalar_result.scalar_one_or_none.return_value = None

        result = await claim_next_format_job(mock_db, "worker-1")

        assert result is None
        mock_db.commit.assert_not_awaited()

    async def test_uses_for_update_skip_locked(self, mock_db, mock_scalar_result):
        mock_scalar_result.scalar_one_or_none.return_value = None

        await claim_next_format_job(mock_db, "worker-1")

        stmt = mock_db.execute.call_args[0][0]
        stmt_str = str(stmt)
        assert "FOR UPDATE" in stmt_str.upper()
        assert hasattr(stmt, "_for_update_arg")
        assert stmt._for_update_arg.skip_locked is True


@pytest.mark.unit
class TestReleaseFormatJobLock:
    async def test_clears_lock_fields(self, mock_db, format_job_id):
        await release_format_job_lock(mock_db, format_job_id)

        mock_db.execute.assert_awaited_once()
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestRecoverStuckFormatJobs:
    async def test_releases_stale_locks(self, mock_db):
        await recover_stuck_format_jobs(mock_db, timeout_minutes=15)

        mock_db.execute.assert_awaited_once()
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestGetFormatJob:
    async def test_returns_job_when_found(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_job = MagicMock()
        mock_scalar_result.scalar_one_or_none.return_value = mock_job

        result = await get_format_job(mock_db, format_job_id)

        assert result == mock_job

    async def test_returns_none_when_not_found(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = None

        result = await get_format_job(mock_db, format_job_id)

        assert result is None


@pytest.mark.unit
class TestUpdateFormatJobStatus:
    async def test_executes_update_and_commits(self, mock_db, format_job_id):
        await update_format_job_status(
            mock_db, format_job_id, FormatJobStatusEnum.FORMATTING
        )

        mock_db.execute.assert_awaited_once()
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestLogFormatJobError:
    async def test_appends_error_to_existing_log(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_job = MagicMock()
        mock_job.error_log = {"phase1": {"message": "old", "timestamp": "t"}}
        mock_scalar_result.scalar_one_or_none.return_value = mock_job

        await log_format_job_error(mock_db, format_job_id, "boom", "formatting")

        assert "formatting" in mock_job.error_log
        assert mock_job.error_log["formatting"]["message"] == "boom"
        mock_db.commit.assert_awaited_once()

    async def test_initializes_error_log_when_none(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_job = MagicMock()
        mock_job.error_log = None
        mock_scalar_result.scalar_one_or_none.return_value = mock_job

        await log_format_job_error(mock_db, format_job_id, "boom", "formatting")

        assert mock_job.error_log is not None
        assert "formatting" in mock_job.error_log

    async def test_does_nothing_when_job_not_found(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = None

        await log_format_job_error(mock_db, format_job_id, "boom", "formatting")

        mock_db.commit.assert_not_awaited()


@pytest.mark.unit
class TestUpdateFormatJobFormatPayload:
    async def test_saves_payload_and_commits(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_job = MagicMock()
        mock_scalar_result.scalar_one_or_none.return_value = mock_job
        payload = {"BLOG": {"status": "SUCCESS"}}

        await update_format_job_format_payload(mock_db, format_job_id, payload)

        assert mock_job.format_payload == payload
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestUpdateFormatJobVideoUrl:
    async def test_saves_url_and_commits(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_job = MagicMock()
        mock_scalar_result.scalar_one_or_none.return_value = mock_job

        await update_format_job_video_url(
            mock_db, format_job_id, "https://example.com/video.mp4"
        )

        assert mock_job.final_video_url == "https://example.com/video.mp4"
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestUpdateFormatJobWorkingMemory:
    async def test_updates_working_memory_and_commits(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_job = MagicMock()
        mock_job.working_memory = {}
        mock_scalar_result.scalar_one_or_none.return_value = mock_job
        new_wm = {"discord_thread_id": "123", "discord_message_id": "456"}

        result = await update_format_job_working_memory(mock_db, format_job_id, new_wm)

        assert result == mock_job
        assert mock_job.working_memory == new_wm
        mock_db.commit.assert_awaited_once()

    async def test_returns_none_when_job_not_found(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_scalar_result.scalar_one_or_none.return_value = None

        result = await update_format_job_working_memory(
            mock_db, format_job_id, {"key": "val"}
        )

        assert result is None
        mock_db.commit.assert_not_awaited()


@pytest.mark.unit
class TestGetFormatJobsForWatcher:
    async def test_returns_jobs_with_discord_thread_id(
        self, mock_db, mock_scalar_result
    ):
        mock_job = MagicMock()
        mock_scalar_result.scalars.return_value.all.return_value = [mock_job]

        result = await get_format_jobs_for_watcher(mock_db)

        assert result == [mock_job]
        mock_db.execute.assert_awaited_once()

    async def test_returns_empty_list_when_no_matching_jobs(
        self, mock_db, mock_scalar_result
    ):
        mock_scalar_result.scalars.return_value.all.return_value = []

        result = await get_format_jobs_for_watcher(mock_db)

        assert result == []


@pytest.mark.unit
class TestGetFormatJobsMissedTerminal:
    async def test_returns_completed_failed_without_final_embed_updated(
        self, mock_db, mock_scalar_result
    ):
        mock_job = MagicMock()
        mock_scalar_result.scalars.return_value.all.return_value = [mock_job]

        result = await get_format_jobs_missed_terminal(mock_db)

        assert result == [mock_job]
        mock_db.execute.assert_awaited_once()

    async def test_excludes_jobs_with_final_embed_updated(
        self, mock_db, mock_scalar_result
    ):
        mock_scalar_result.scalars.return_value.all.return_value = []

        result = await get_format_jobs_missed_terminal(mock_db)

        assert result == []


@pytest.mark.unit
class TestResetFormatJobToComposition:
    async def test_resets_status_and_clears_lock(
        self, mock_db, mock_scalar_result, format_job_id
    ):
        mock_job = MagicMock()
        mock_job.status = FormatJobStatusEnum.FAILED
        mock_job.locked_at = MagicMock()
        mock_job.locked_by = "worker-1"
        mock_job.working_memory = {"final_embed_updated": True, "other_key": "val"}
        mock_scalar_result.scalar_one_or_none.return_value = mock_job

        result = await reset_format_job_to_composition(mock_db, format_job_id)

        assert result == mock_job
        assert mock_job.status == FormatJobStatusEnum.COMPOSITION
        assert mock_job.locked_at is None
        assert mock_job.locked_by is None
        assert "final_embed_updated" not in mock_job.working_memory
        assert mock_job.working_memory["other_key"] == "val"
        mock_db.commit.assert_awaited_once()
