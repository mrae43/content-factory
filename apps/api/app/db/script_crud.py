from uuid import UUID
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.discord_models import ScriptJob
from app.schemas.shorts import ScriptJobStatusEnum


async def create_script_job(
    db: AsyncSession,
    title: str,
    user_reference: str = "",
    source_urls: Optional[list[str]] = None,
    story_directives: Optional[dict] = None,
) -> ScriptJob:
    job = ScriptJob(
        title=title,
        user_reference=user_reference,
        source_urls=source_urls or [],
        story_directives=story_directives or {},
        status=ScriptJobStatusEnum.PENDING,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def get_script_job(db: AsyncSession, job_id: UUID) -> Optional[ScriptJob]:
    stmt = select(ScriptJob).filter(ScriptJob.id == job_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_script_job_status(
    db: AsyncSession, job_id: UUID, status: ScriptJobStatusEnum
) -> None:
    stmt = update(ScriptJob).where(ScriptJob.id == job_id).values(status=status)
    await db.execute(stmt)
    await db.commit()


async def log_script_job_error(
    db: AsyncSession, job_id: UUID, error_message: str, phase: str
) -> None:
    stmt = select(ScriptJob).filter(ScriptJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    if job:
        error_log = job.error_log or {}
        error_log[phase] = {
            "message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        job.error_log = error_log
        await db.commit()
