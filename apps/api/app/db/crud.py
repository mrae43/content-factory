from uuid import UUID
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import FactCheckClaim, RenderJob, Script
from app.schemas.shorts import JobStatusEnum


async def get_latest_script(db: AsyncSession, job_id: UUID) -> Optional[Script]:
    stmt = (
        select(Script)
        .where(Script.job_id == job_id)
        .order_by(Script.version.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_render_job(db: AsyncSession, job_id: UUID) -> Optional[RenderJob]:
    """
    Fetches a RenderJob with its associated scripts and assets eagerly loaded.
    """
    stmt = (
        select(RenderJob)
        .options(selectinload(RenderJob.scripts), selectinload(RenderJob.assets))
        .filter(RenderJob.id == job_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_job_status(db: AsyncSession, job_id: UUID, status: JobStatusEnum):
    """
    Updates the status of a RenderJob.
    """
    stmt = update(RenderJob).where(RenderJob.id == job_id).values(status=status)
    await db.execute(stmt)
    await db.commit()


async def log_error(db: AsyncSession, job_id: UUID, error_message: str, phase: str):
    """
    Logs an error into the RenderJob's error_log JSONB column.
    """
    stmt = select(RenderJob).filter(RenderJob.id == job_id)
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


async def save_script(db: AsyncSession, job_id: UUID, content: str, version: int):
    new_script = Script(
        job_id=job_id, content=content, version=version, is_approved=False
    )
    db.add(new_script)
    await db.commit()


async def append_script_feedback(
    db: AsyncSession,
    job_id: UUID,
    feedback: str = "",
    structured_claims: Optional[List[dict]] = None,
    overall_reasoning: str = "",
    revision_number: int = 0,
):
    latest = await get_latest_script(db, job_id)
    if latest:
        history = latest.feedback_history or []

        if structured_claims:
            entry = {
                "feedback_type": "structured_claims",
                "failed_claims": structured_claims,
                "overall_reasoning": overall_reasoning,
                "revision_number": revision_number,
            }
        else:
            entry = feedback

        history.append(entry)
        latest.feedback_history = history
        await db.commit()


async def save_fact_check_claims(
    db: AsyncSession, script_id: UUID, claims: list[dict]
) -> None:
    rows = [
        FactCheckClaim(
            script_id=script_id,
            claim_text=c["claim_text"],
            verdict=c["verdict"],
            confidence=c["confidence"],
            evidence_references=c.get("evidence_references", []),
        )
        for c in claims
    ]
    db.add_all(rows)
    await db.flush()


async def claim_next_job(db: AsyncSession, worker_id: str) -> Optional[RenderJob]:
    stmt = (
        select(RenderJob)
        .where(
            RenderJob.status.notin_(
                [
                    JobStatusEnum.COMPLETED,
                    JobStatusEnum.FAILED,
                    JobStatusEnum.HUMAN_REVIEW_NEEDED,
                ]
            ),
            RenderJob.locked_at.is_(None),
        )
        .order_by(RenderJob.created_at.asc())
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


async def release_job_lock(db: AsyncSession, job_id: UUID) -> None:
    stmt = (
        update(RenderJob)
        .where(RenderJob.id == job_id)
        .values(locked_at=None, locked_by=None)
    )
    await db.execute(stmt)
    await db.commit()


async def recover_stuck_jobs(db: AsyncSession, timeout_minutes: int) -> None:
    cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(
        minutes=timeout_minutes
    )
    stmt = (
        update(RenderJob)
        .where(RenderJob.locked_at.isnot(None), RenderJob.locked_at < cutoff)
        .values(locked_at=None, locked_by=None)
    )
    await db.execute(stmt)
    await db.commit()
