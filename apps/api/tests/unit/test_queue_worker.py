import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

from app.workers.queue_worker import QueueWorker
from app.schemas.shorts import JobStatusEnum


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
def mock_job():
    job = MagicMock()
    job.id = uuid4()
    job.status = JobStatusEnum.PENDING
    return job


@pytest.fixture
def worker():
    return QueueWorker()


POLL_INTERVAL = 0.01
LOCK_TIMEOUT = 10


@pytest.mark.unit
class TestQueueWorkerInit:
    def test_init_generates_uuid_worker_id(self, worker):
        assert UUID(worker._worker_id)

    def test_init_running_is_false(self, worker):
        assert worker._running is False

    def test_init_current_task_is_none(self, worker):
        assert worker._current_task is None


@pytest.mark.unit
class TestQueueWorkerStart:
    @patch("app.workers.queue_worker.settings")
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.recover_stuck_jobs")
    async def test_start_calls_recover_stuck_jobs(
        self, mock_recover, mock_session_factory, mock_settings, mock_db, worker
    ):
        mock_session_factory.return_value = _make_session_ctx(mock_db)
        mock_settings.worker_lock_timeout_minutes = LOCK_TIMEOUT

        await worker.start()

        mock_recover.assert_awaited_once_with(mock_db, LOCK_TIMEOUT)
        await worker.stop()

    @patch("app.workers.queue_worker.settings")
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.recover_stuck_jobs")
    async def test_start_sets_running_true(
        self, mock_recover, mock_session_factory, mock_settings, mock_db, worker
    ):
        mock_session_factory.return_value = _make_session_ctx(mock_db)

        await worker.start()

        assert worker._running is True
        await worker.stop()

    @patch("app.workers.queue_worker.settings")
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.recover_stuck_jobs")
    async def test_start_creates_poll_loop_task(
        self, mock_recover, mock_session_factory, mock_settings, mock_db, worker
    ):
        mock_session_factory.return_value = _make_session_ctx(mock_db)

        await worker.start()

        assert worker._current_task is not None
        assert not worker._current_task.done()
        await worker.stop()


@pytest.mark.unit
class TestQueueWorkerStop:
    @patch("app.workers.queue_worker.settings")
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.recover_stuck_jobs")
    async def test_stop_sets_running_false(
        self, mock_recover, mock_session_factory, mock_settings, mock_db, worker
    ):
        mock_session_factory.return_value = _make_session_ctx(mock_db)
        await worker.start()

        await worker.stop()

        assert worker._running is False

    @patch("app.workers.queue_worker.settings")
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.recover_stuck_jobs")
    async def test_stop_cancels_active_task(
        self, mock_recover, mock_session_factory, mock_settings, mock_db, worker
    ):
        mock_session_factory.return_value = _make_session_ctx(mock_db)
        await worker.start()
        task = worker._current_task

        await worker.stop()

        assert task.cancelled()

    @patch("app.workers.queue_worker.settings")
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.recover_stuck_jobs")
    async def test_stop_handles_cancelled_error_gracefully(
        self, mock_recover, mock_session_factory, mock_settings, mock_db, worker
    ):
        mock_session_factory.return_value = _make_session_ctx(mock_db)
        await worker.start()

        await worker.stop()

        assert worker._running is False
        assert worker._current_task is not None

    async def test_stop_noop_when_no_task(self, worker):
        await worker.stop()

        assert worker._running is False
        assert worker._current_task is None


@pytest.mark.unit
class TestQueueWorkerPollLoop:
    @patch(
        "app.workers.queue_worker.settings", worker_poll_interval_seconds=POLL_INTERVAL
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    @patch("app.workers.queue_worker.asyncio.sleep")
    async def test_poll_sleeps_when_no_job_available(
        self, mock_sleep, mock_session_factory, mock_settings, mock_db, worker
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)

        def stop_after_first_sleep(*args, **kwargs):
            worker._running = False

        mock_sleep.side_effect = stop_after_first_sleep

        worker._running = True

        with patch("app.workers.queue_worker.claim_next_job", return_value=None):
            await worker._poll_loop()

        mock_sleep.assert_called_with(POLL_INTERVAL)

    @patch(
        "app.workers.queue_worker.settings", worker_poll_interval_seconds=POLL_INTERVAL
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    async def test_poll_processes_job_when_available(
        self, mock_session_factory, mock_settings, mock_db, mock_job, worker
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)
        fake_claim = _make_claim_factory(mock_job, worker)
        worker._running = True

        with patch("app.workers.queue_worker.claim_next_job", side_effect=fake_claim):
            with patch("app.workers.queue_worker.release_job_lock") as mock_release:
                with patch.object(
                    worker,
                    "_process_one_transition",
                    new=AsyncMock(
                        side_effect=lambda jid: setattr(worker, "_running", False)
                    ),
                ):
                    await worker._poll_loop()

        mock_release.assert_awaited_with(mock_db, mock_job.id)


@pytest.mark.unit
class TestQueueWorkerPollLoopErrorHandling:
    @patch(
        "app.workers.queue_worker.settings", worker_poll_interval_seconds=POLL_INTERVAL
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    async def test_poll_marks_job_failed_on_exception(
        self,
        mock_session_factory,
        mock_settings,
        mock_db,
        mock_job,
        worker,
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)
        fake_claim = _make_claim_factory(mock_job, worker)
        worker._running = True

        with patch("app.workers.queue_worker.claim_next_job", side_effect=fake_claim):
            with patch("app.workers.queue_worker.log_error") as mock_log_error:
                with patch("app.workers.queue_worker.update_job_status") as mock_update:
                    with patch("app.workers.queue_worker.release_job_lock"):
                        with patch.object(
                            worker,
                            "_process_one_transition",
                            new=AsyncMock(side_effect=RuntimeError("boom")),
                        ):
                            await worker._poll_loop()

        mock_log_error.assert_awaited_with(
            mock_db, mock_job.id, "boom", phase="queue_worker"
        )
        mock_update.assert_awaited_with(mock_db, mock_job.id, JobStatusEnum.FAILED)

    @patch(
        "app.workers.queue_worker.settings", worker_poll_interval_seconds=POLL_INTERVAL
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    async def test_poll_releases_lock_on_error(
        self,
        mock_session_factory,
        mock_settings,
        mock_db,
        mock_job,
        worker,
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)
        fake_claim = _make_claim_factory(mock_job, worker)
        worker._running = True

        with patch("app.workers.queue_worker.claim_next_job", side_effect=fake_claim):
            with patch("app.workers.queue_worker.release_job_lock") as mock_release:
                with patch("app.workers.queue_worker.log_error"):
                    with patch("app.workers.queue_worker.update_job_status"):
                        with patch.object(
                            worker,
                            "_process_one_transition",
                            new=AsyncMock(side_effect=RuntimeError("boom")),
                        ):
                            await worker._poll_loop()

        mock_release.assert_awaited_with(mock_db, mock_job.id)

    @patch(
        "app.workers.queue_worker.settings", worker_poll_interval_seconds=POLL_INTERVAL
    )
    @patch("app.workers.queue_worker.AsyncSessionLocal")
    async def test_poll_releases_lock_on_success(
        self,
        mock_session_factory,
        mock_settings,
        mock_db,
        mock_job,
        worker,
    ):
        mock_session_factory.side_effect = lambda: _make_session_ctx(mock_db)
        fake_claim = _make_claim_factory(mock_job, worker)
        worker._running = True

        with patch("app.workers.queue_worker.claim_next_job", side_effect=fake_claim):
            with patch("app.workers.queue_worker.release_job_lock") as mock_release:
                with patch.object(
                    worker,
                    "_process_one_transition",
                    new=AsyncMock(
                        side_effect=lambda jid: setattr(worker, "_running", False)
                    ),
                ):
                    await worker._poll_loop()

        mock_release.assert_awaited_with(mock_db, mock_job.id)
