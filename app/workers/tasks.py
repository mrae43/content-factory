import logging
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

from app.db.models import ResearchChunk, RenderJob, JobStatusEnum

logger = logging.getLogger(__name__)


async def cleanup_local_research_chunks(job_id: UUID, db: AsyncSession) -> dict:
    """
    Garbage Collection Step (End of Pipeline).
    Deletes all ephemeral (LOCAL) vector chunks for a completed job,
    but retains anything flagged as 'GLOBAL' for future AEO/RAG memory.
    """
    logger.info(f"Initiating RAG cleanup for completed Job: {job_id}")

    job = await db.get(RenderJob, job_id)
    if not job or job.status not in (JobStatusEnum.COMPLETED, JobStatusEnum.FAILED):
        logger.warning(f"Skipping cleanup: Job {job_id} is not in a terminal state.")
        return {"status": "skipped", "reason": "Job not found in terminal state"}

    # Delete chunks where meta->>'scope' == 'LOCAL' for this specific job
    stmt = delete(ResearchChunk).where(
        ResearchChunk.job_id == job_id, ResearchChunk.meta["scope"].astext == "LOCAL"
    )

    result = await db.execute(stmt)
    await db.commit()

    deleted_count = result.rowcount
    logger.info(
        f"Successfully purged {deleted_count} ephemeral RAG chunks for Job {job_id}."
    )

    return {"status": "success", "deleted_chunks": deleted_count, "job_id": str(job_id)}
