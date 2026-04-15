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
from app.schemas.shorts import JobStatusEnum
from app.workers.orchestrator import execute_state_transition

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
            job = None
            async with AsyncSessionLocal() as db:
                job = await claim_next_job(db, self._worker_id)

            if job is None:
                await asyncio.sleep(settings.worker_poll_interval_seconds)
                continue

            try:
                await self._process_one_transition(job.id)
            except Exception as e:
                logger.exception(f"Unhandled error processing job {job.id}: {e}")
                async with AsyncSessionLocal() as db:
                    await log_error(db, job.id, str(e), phase="queue_worker")
                    await update_job_status(db, job.id, JobStatusEnum.FAILED)
            finally:
                async with AsyncSessionLocal() as db:
                    await release_job_lock(db, job.id)

    async def _process_one_transition(self, job_id):
        async with AsyncSessionLocal() as db:
            job = await get_render_job(db, job_id)
            if not job:
                logger.error(f"Job {job_id} not found during processing.")
                return
            await execute_state_transition(db, job)
