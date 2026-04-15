from uuid import UUID
from typing import Optional
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
            "timestamp": "now()",  # Simplified for now, model handles updated_at
        }
        job.error_log = error_log
        await db.commit()


async def save_script(db: AsyncSession, job_id: UUID, content: str, version: int):
    new_script = Script(
        job_id=job_id, content=content, version=version, is_approved=False
    )
    db.add(new_script)
    await db.commit()


async def append_script_feedback(db: AsyncSession, job_id: UUID, feedback: str):
    latest = await get_latest_script(db, job_id)
    if latest:
        history = latest.feedback_history or []
        history.append(feedback)
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
