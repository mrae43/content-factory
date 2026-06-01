import asyncio
import logging
from uuid import uuid4
from typing import Optional

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.crud import (
    claim_next_job,
    release_job_lock,
    recover_stuck_jobs,
    get_render_job,
    log_error,
    update_job_status,
)
from app.db.format_crud import (
    claim_next_format_job,
    release_format_job_lock,
    recover_stuck_format_jobs,
    get_format_job,
    log_format_job_error,
    update_format_job_status,
)
from app.schemas.shorts import JobStatusEnum, FormatJobStatusEnum
from app.workers.orchestrator import execute_state_transition
from app.workers.format_orchestrator import execute_format_state_transition

logger = logging.getLogger("factory.queue_worker")


class QueueWorker:
    def __init__(self):
        self._worker_id = str(uuid4())
        self._running = False
        self._current_task: Optional[asyncio.Task] = None

    async def start(self):
        logger.info(f"QueueWorker {self._worker_id} starting...")
        async with AsyncSessionLocal() as db:
            await recover_stuck_jobs(db, settings.worker_lock_timeout_minutes)
            await recover_stuck_format_jobs(db, settings.worker_lock_timeout_minutes)
        self._running = True
        self._current_task = asyncio.create_task(self._poll_loop())
        logger.info(f"QueueWorker {self._worker_id} started.")

    async def stop(self):
        logger.info(f"QueueWorker {self._worker_id} stopping...")
        self._running = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        logger.info(f"QueueWorker {self._worker_id} stopped.")

    async def _poll_loop(self):
        while self._running:
            render_job = None
            async with AsyncSessionLocal() as db:
                render_job = await claim_next_job(db, self._worker_id)

            if render_job:
                try:
                    await self._process_one_transition(render_job.id)
                except Exception as e:
                    logger.exception(
                        f"Unhandled error processing job {render_job.id}: {e}"
                    )
                    async with AsyncSessionLocal() as db:
                        await log_error(db, render_job.id, str(e), phase="queue_worker")
                        await update_job_status(db, render_job.id, JobStatusEnum.FAILED)
                finally:
                    async with AsyncSessionLocal() as db:
                        await release_job_lock(db, render_job.id)

            fmt_job = None
            async with AsyncSessionLocal() as db:
                fmt_job = await claim_next_format_job(db, self._worker_id)

            if fmt_job:
                try:
                    await self._process_one_format_transition(fmt_job.id)
                except Exception as e:
                    logger.exception(
                        f"Unhandled error processing format job {fmt_job.id}: {e}"
                    )
                    async with AsyncSessionLocal() as db:
                        await log_format_job_error(
                            db, fmt_job.id, str(e), phase="queue_worker"
                        )
                        await update_format_job_status(
                            db, fmt_job.id, FormatJobStatusEnum.FAILED
                        )
                finally:
                    async with AsyncSessionLocal() as db:
                        await release_format_job_lock(db, fmt_job.id)

            if render_job is None and fmt_job is None:
                await asyncio.sleep(settings.worker_poll_interval_seconds)

    async def _process_one_transition(self, job_id):
        async with AsyncSessionLocal() as db:
            job = await get_render_job(db, job_id)
            if not job:
                logger.error(f"Job {job_id} not found during processing.")
                return
            await execute_state_transition(db, job)

    async def _process_one_format_transition(self, job_id):
        async with AsyncSessionLocal() as db:
            job = await get_format_job(db, job_id)
            if not job:
                logger.error(f"Format job {job_id} not found during processing.")
                return
            await execute_format_state_transition(db, job)
