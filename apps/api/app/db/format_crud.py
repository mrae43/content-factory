from uuid import UUID
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.attributes import flag_modified

from app.db.discord_models import FormatJob
from app.schemas.shorts import FormatJobStatusEnum


async def create_format_job(
    db: AsyncSession,
    source_job_id: UUID,
    platform: str,
    format_type: str,
    snapshot_data: dict,
) -> FormatJob:
    try:
        job = FormatJob(
            source_job_id=source_job_id,
            platform=platform,
            format_type=format_type,
            title=snapshot_data.get("title", ""),
            script_content=snapshot_data.get("script_content"),
            claims=snapshot_data.get("claims"),
            refined_context=snapshot_data.get("refined_context"),
            story_directives=snapshot_data.get("story_directives"),
            hedge_index=snapshot_data.get("hedge_index"),
            epistemic_ledger=snapshot_data.get("epistemic_ledger"),
        )
        db.add(job)
        await db.commit()
        return job
    except IntegrityError:
        await db.rollback()
        stmt = select(FormatJob).where(
            FormatJob.source_job_id == source_job_id,
            FormatJob.platform == platform,
            FormatJob.format_type == format_type,
        )
        result = await db.execute(stmt)
        return result.scalar_one()


async def claim_next_format_job(
    db: AsyncSession, worker_id: str
) -> Optional[FormatJob]:
    stmt = (
        select(FormatJob)
        .where(
            FormatJob.status.notin_(
                [
                    FormatJobStatusEnum.COMPLETED,
                    FormatJobStatusEnum.FAILED,
                    FormatJobStatusEnum.HUMAN_REVIEW_NEEDED,
                ]
            ),
            FormatJob.locked_at.is_(None),
        )
        .order_by(FormatJob.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if job:
        job.locked_at = datetime.now(timezone.utc)
        job.locked_by = worker_id
        await db.commit()

    return job


async def release_format_job_lock(db: AsyncSession, job_id: UUID) -> None:
    stmt = (
        update(FormatJob)
        .where(FormatJob.id == job_id)
        .values(locked_at=None, locked_by=None)
    )
    await db.execute(stmt)
    await db.commit()


async def recover_stuck_format_jobs(db: AsyncSession, timeout_minutes: int) -> None:
    cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(
        minutes=timeout_minutes
    )
    stmt = (
        update(FormatJob)
        .where(FormatJob.locked_at.isnot(None), FormatJob.locked_at < cutoff)
        .values(locked_at=None, locked_by=None)
    )
    await db.execute(stmt)
    await db.commit()


async def get_format_job(db: AsyncSession, job_id: UUID) -> Optional[FormatJob]:
    stmt = select(FormatJob).filter(FormatJob.id == job_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_format_job_status(
    db: AsyncSession, job_id: UUID, status: FormatJobStatusEnum
) -> None:
    stmt = update(FormatJob).where(FormatJob.id == job_id).values(status=status)
    await db.execute(stmt)
    await db.commit()


async def log_format_job_error(
    db: AsyncSession, job_id: UUID, error_message: str, phase: str
) -> None:
    stmt = select(FormatJob).filter(FormatJob.id == job_id)
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


async def update_format_job_format_payload(
    db: AsyncSession, job_id: UUID, payload: dict
) -> None:
    stmt = select(FormatJob).filter(FormatJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if job:
        job.format_payload = payload
        flag_modified(job, "format_payload")
        await db.commit()


async def update_format_job_video_url(db: AsyncSession, job_id: UUID, url: str) -> None:
    stmt = select(FormatJob).filter(FormatJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if job:
        job.final_video_url = url
        await db.commit()
