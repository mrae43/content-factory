from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload
from uuid import UUID
import logging

from app.schemas.shorts import (
    JobCreateRequest,
    RenderJobResponse,
    ScriptApprovalRequest,
    JobStatusEnum,
)
from app.db.models import RenderJob, Script
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["Content Factory"])


@router.post(
    "/", response_model=RenderJobResponse, status_code=status.HTTP_202_ACCEPTED
)
async def create_render_job(
    request: JobCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        new_job = RenderJob(
            topic=request.topic,
            pre_context=request.pre_context.model_dump(mode="json"),
            strict_compliance_mode=request.strict_compliance_mode,
            status=JobStatusEnum.PENDING,
        )
        db.add(new_job)
        await db.commit()

        stmt = (
            select(RenderJob)
            .options(selectinload(RenderJob.scripts), selectinload(RenderJob.assets))
            .filter(RenderJob.id == new_job.id)
        )
        result = await db.execute(stmt)
        job_to_return = result.scalar_one()

        logger.info(
            f"Created RenderJob {job_to_return.id} for topic: {job_to_return.topic}"
        )
        return job_to_return

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database transaction failed: {str(e)}"
        )


@router.get(
    "/{job_id}", response_model=RenderJobResponse, status_code=status.HTTP_200_OK
)
async def get_render_job(job_id: UUID, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(RenderJob)
        .options(
            selectinload(RenderJob.scripts).selectinload(Script.claims),
            selectinload(RenderJob.assets),
        )
        .filter(RenderJob.id == job_id)
    )

    result = await db.execute(stmt)
    job = result.unique().scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="RenderJob not found")

    return job


@router.post(
    "/{job_id}/approve-script",
    response_model=RenderJobResponse,
    status_code=status.HTTP_200_OK,
)
async def approve_script(
    job_id: UUID,
    request: ScriptApprovalRequest,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(RenderJob)
        .options(joinedload(RenderJob.scripts))
        .filter(RenderJob.id == job_id)
    )
    result = await db.execute(stmt)
    job = result.unique().scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="RenderJob not found")

    if job.status not in [
        JobStatusEnum.HUMAN_REVIEW_NEEDED,
        JobStatusEnum.FACT_CHECKING_SCRIPT,
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Job is in '{job.status}', cannot approve script at this stage.",
        )

    if not job.scripts:
        raise HTTPException(status_code=400, detail="No script found to approve")

    latest_script = max(job.scripts, key=lambda s: s.version)

    if request.is_approved:
        latest_script.is_approved = True
        job.status = JobStatusEnum.ASSET_GENERATION
        await db.commit()

        stmt = (
            select(RenderJob)
            .options(selectinload(RenderJob.scripts))
            .filter(RenderJob.id == job.id)
        )
        result = await db.execute(stmt)
        job = result.scalar_one()

    else:
        job.status = JobStatusEnum.SCRIPTING

        if request.human_feedback:
            feedback_history = latest_script.feedback_history or []
            feedback_history.append(
                {"source": "human_editor", "comment": request.human_feedback}
            )
            latest_script.feedback_history = feedback_history

        await db.commit()

        stmt = (
            select(RenderJob)
            .options(selectinload(RenderJob.scripts))
            .filter(RenderJob.id == job.id)
        )
        result = await db.execute(stmt)
        job = result.scalar_one()

    return job
