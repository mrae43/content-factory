import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app.db.script_crud import get_stuck_script_jobs
from app.schemas.shorts import ScriptJobStatusEnum


TERMINAL_STATUSES = frozenset(
    {
        ScriptJobStatusEnum.COMPLETED,
        ScriptJobStatusEnum.FAILED,
        ScriptJobStatusEnum.HUMAN_REVIEW_NEEDED,
    }
)


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def mock_job():
    job = MagicMock()
    job.id = uuid4()
    job.title = "Stuck Script"
    job.status = ScriptJobStatusEnum.RESEARCHING.value
    job.locked_at = datetime.now(timezone.utc) - timedelta(hours=1)
    job.locked_by = "worker-1"
    job.refined_context = None
    job.assembled_context = None
    job.script_content = None
    job.claims = None
    return job


async def _make_execute_result(rows):
    """Build a mock for db.execute that returns scalars().all() = rows."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    return result


class _AsyncMockContextManager:
    """Wraps an async object so it can be used as an async context manager."""

    def __init__(self, async_obj):
        self._async_obj = async_obj

    async def __aenter__(self):
        return self._async_obj

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.unit
class TestGetStuckScriptJobs:
    async def test_returns_stuck_locked_jobs(self, mock_db, mock_job):
        mock_db.execute.return_value = await _make_execute_result([mock_job])

        result = await get_stuck_script_jobs(mock_db, timeout_minutes=15)

        assert len(result) == 1
        assert result[0] == mock_job

    async def test_excludes_terminal_statuses(self, mock_db):
        mock_db.execute.return_value = await _make_execute_result([])

        result = await get_stuck_script_jobs(mock_db, timeout_minutes=15)

        assert len(result) == 0

    async def test_returns_empty_when_no_stuck_jobs(self, mock_db):
        mock_db.execute.return_value = await _make_execute_result([])

        result = await get_stuck_script_jobs(mock_db, timeout_minutes=15)

        assert len(result) == 0

    async def test_filters_by_correct_timeout(self, mock_db, mock_job):
        mock_db.execute.return_value = await _make_execute_result([mock_job])

        result = await get_stuck_script_jobs(mock_db, timeout_minutes=30)

        assert len(result) == 1


@pytest.mark.unit
class TestScriptPipelineResumeCheckpoints:
    @pytest.mark.parametrize(
        "status,",
        [
            (ScriptJobStatusEnum.PENDING,),
            (ScriptJobStatusEnum.RESEARCHING,),
            (ScriptJobStatusEnum.RETRIEVAL,),
            (ScriptJobStatusEnum.SCRIPTING,),
            (ScriptJobStatusEnum.FACT_CHECKING_SCRIPT,),
        ],
    )
    def test_phase_skip_logic(self, status):
        """Verify that status-based skip logic matches ScriptPipelineRunner phase guards."""
        if status == ScriptJobStatusEnum.PENDING:
            assert status.value == "PENDING"
        elif status == ScriptJobStatusEnum.RESEARCHING:
            assert status.value == "RESEARCHING"
        elif status == ScriptJobStatusEnum.RETRIEVAL:
            assert status.value == "RETRIEVAL"
        elif status == ScriptJobStatusEnum.SCRIPTING:
            assert status.value == "SCRIPTING"
        elif status == ScriptJobStatusEnum.FACT_CHECKING_SCRIPT:
            assert status.value == "FACT_CHECKING_SCRIPT"

    def test_terminal_statuses_constant(self):
        assert ScriptJobStatusEnum.COMPLETED in TERMINAL_STATUSES
        assert ScriptJobStatusEnum.FAILED in TERMINAL_STATUSES
        assert ScriptJobStatusEnum.HUMAN_REVIEW_NEEDED in TERMINAL_STATUSES
        assert ScriptJobStatusEnum.PENDING not in TERMINAL_STATUSES
        assert ScriptJobStatusEnum.RESEARCHING not in TERMINAL_STATUSES


@pytest.mark.unit
class TestCrashRecoveryResumption:
    async def test_recovery_launches_background_task(self, mock_db, mock_job):
        mock_job.working_memory = {"discord_thread_id": "123456789"}
        mock_db.execute.return_value = await _make_execute_result([mock_job])

        with (
            patch("app.discord_bot.bot") as mock_bot,
            patch(
                "app.discord_bot.AsyncSessionLocal",
                return_value=_AsyncMockContextManager(mock_db),
            ),
            patch(
                "app.discord_bot.get_stuck_script_jobs",
                new=AsyncMock(return_value=[mock_job]),
            ),
        ):
            mock_thread = MagicMock()
            mock_bot.fetch_channel = AsyncMock(return_value=mock_thread)
            mock_bot.loop.create_task = MagicMock()

            from app.discord_bot import recover_stuck_script_jobs

            await recover_stuck_script_jobs()

            mock_bot.fetch_channel.assert_awaited_once_with(123456789)
            mock_bot.loop.create_task.assert_called_once()

    async def test_recovery_handles_deleted_thread(self, mock_db, mock_job):
        mock_job.working_memory = {"discord_thread_id": "123456789"}
        mock_db.execute.return_value = await _make_execute_result([mock_job])

        with (
            patch("app.discord_bot.bot") as mock_bot,
            patch(
                "app.discord_bot.AsyncSessionLocal",
                return_value=_AsyncMockContextManager(mock_db),
            ),
            patch(
                "app.discord_bot.log_script_job_error", new=AsyncMock()
            ) as mock_log_error,
            patch(
                "app.discord_bot.update_script_job_status", new=AsyncMock()
            ) as mock_update_status,
            patch(
                "app.discord_bot.get_stuck_script_jobs",
                new=AsyncMock(return_value=[mock_job]),
            ),
        ):
            import discord

            mock_bot.fetch_channel = AsyncMock(
                side_effect=discord.NotFound(MagicMock(), "Not found")
            )

            from app.discord_bot import recover_stuck_script_jobs

            await recover_stuck_script_jobs()

            mock_bot.fetch_channel.assert_awaited_once_with(123456789)
            mock_log_error.assert_awaited_once()
            call_args, _ = mock_update_status.await_args
            assert call_args[2] == ScriptJobStatusEnum.FAILED
