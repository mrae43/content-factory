import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

from app.workers.queue_worker import QueueWorker
from app.schemas.shorts import FormatJobStatusEnum


def _make_session_ctx(mock_db):
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_db)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_claim_factory(mock_job, worker, *, error=False):
    call_count = 0

    async def fake_claim(db, worker_id):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_job
        worker._running = False
        return None

    return fake_claim


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
def mock_render_job():
    job = MagicMock()
    job.id = uuid4()
    return job


@pytest.fixture
def mock_format_job():
    job = MagicMock()
    job.id = uuid4()
    return job


@pytest.fixture
def worker():
    return QueueWorker()


POLL_INTERVAL = 0.01


@pytest.mark.unit
class TestQueueWorkerDualPollLoop:
    @patch(
        "app.workers.queue_worker.settings",
        worker_poll_interval_seconds=POLL_INTERVAL,
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.asyncio.sleep")
    async def test_poll_processes_both_render_and_format_jobs(
        self,
        mock_sleep,
        mock_session_factory,
        mock_settings,
        mock_db,
        mock_render_job,
        mock_format_job,
        worker,
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)
        worker._running = True
        last_render = None
        last_format = None

        async def fake_render_process(jid):
            nonlocal last_render
            last_render = jid

        async def fake_format_process(jid):
            nonlocal last_format
            last_format = jid
            worker._running = False

        with (
            patch(
                "app.workers.queue_worker.claim_next_job",
                return_value=mock_render_job,
            ),
            patch(
                "app.workers.queue_worker.claim_next_format_job",
                return_value=mock_format_job,
            ),
            patch.object(
                worker,
                "_process_one_transition",
                new=AsyncMock(side_effect=fake_render_process),
            ),
            patch.object(
                worker,
                "_process_one_format_transition",
                new=AsyncMock(side_effect=fake_format_process),
            ),
        ):
            await worker._poll_loop()

        assert last_render == mock_render_job.id
        assert last_format == mock_format_job.id

    @patch(
        "app.workers.queue_worker.settings",
        worker_poll_interval_seconds=POLL_INTERVAL,
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.asyncio.sleep")
    async def test_poll_skips_both_when_none_available(
        self,
        mock_sleep,
        mock_session_factory,
        mock_settings,
        mock_db,
        worker,
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)

        def stop_after_first_sleep(*args, **kwargs):
            worker._running = False

        mock_sleep.side_effect = stop_after_first_sleep
        worker._running = True

        with (
            patch("app.workers.queue_worker.claim_next_job", return_value=None),
            patch(
                "app.workers.queue_worker.claim_next_format_job", return_value=None
            ),
        ):
            await worker._poll_loop()

        mock_sleep.assert_called_once()

    @patch(
        "app.workers.queue_worker.settings",
        worker_poll_interval_seconds=POLL_INTERVAL,
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.asyncio.sleep")
    async def test_poll_releases_format_job_lock_on_success(
        self,
        mock_sleep,
        mock_session_factory,
        mock_settings,
        mock_db,
        mock_format_job,
        worker,
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)
        fake_claim = _make_claim_factory(mock_format_job, worker)
        worker._running = True

        with (
            patch(
                "app.workers.queue_worker.claim_next_job",
                return_value=None,
            ),
            patch(
                "app.workers.queue_worker.claim_next_format_job",
                side_effect=fake_claim,
            ),
            patch(
                "app.workers.queue_worker.release_format_job_lock"
            ) as mock_release,
            patch.object(
                worker,
                "_process_one_format_transition",
                new=AsyncMock(
                    side_effect=lambda jid: setattr(worker, "_running", False)
                ),
            ),
        ):
            await worker._poll_loop()

        mock_release.assert_awaited_with(mock_db, mock_format_job.id)

    @patch(
        "app.workers.queue_worker.settings",
        worker_poll_interval_seconds=POLL_INTERVAL,
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.asyncio.sleep")
    async def test_poll_marks_format_job_failed_on_exception(
        self,
        mock_sleep,
        mock_session_factory,
        mock_settings,
        mock_db,
        mock_format_job,
        worker,
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)
        fake_claim = _make_claim_factory(mock_format_job, worker)
        worker._running = True

        with (
            patch(
                "app.workers.queue_worker.claim_next_job",
                return_value=None,
            ),
            patch(
                "app.workers.queue_worker.claim_next_format_job",
                side_effect=fake_claim,
            ),
            patch(
                "app.workers.queue_worker.log_format_job_error"
            ) as mock_log_error,
            patch(
                "app.workers.queue_worker.update_format_job_status"
            ) as mock_update,
            patch("app.workers.queue_worker.release_format_job_lock"),
            patch.object(
                worker,
                "_process_one_format_transition",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            await worker._poll_loop()

        mock_log_error.assert_awaited_with(
            mock_db, mock_format_job.id, "boom", phase="queue_worker"
        )
        mock_update.assert_awaited_with(
            mock_db, mock_format_job.id, FormatJobStatusEnum.FAILED
        )

    @patch(
        "app.workers.queue_worker.settings",
        worker_lock_timeout_minutes=15,
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.recover_stuck_jobs")
    @patch("app.workers.queue_worker.recover_stuck_format_jobs")
    async def test_start_recovers_both_job_types(
        self,
        mock_recover_fmt,
        mock_recover_render,
        mock_session_factory,
        mock_settings,
        mock_db,
        worker,
    ):
        mock_session_factory.return_value = _make_session_ctx(mock_db)

        await worker.start()

        mock_recover_render.assert_awaited_once_with(mock_db, 15)
        mock_recover_fmt.assert_awaited_once_with(mock_db, 15)
        await worker.stop()
