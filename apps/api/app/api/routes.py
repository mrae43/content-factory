from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm.attributes import flag_modified
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
    next_status_after_fact_check,
)
from app.db.models import RenderJob, Script
from app.db.session import get_db
from app.db.crud import list_render_jobs as crud_list_jobs, get_latest_format_script
from app.workers.carousel_image_agent import CarouselImageAgent, merge_image_urls
from app.workers.agents import AgentActionStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["Content Factory"])


@router.post(
    "",
    response_model=RenderJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
@router.post(
    "/", response_model=RenderJobResponse, status_code=status.HTTP_202_ACCEPTED
)
async def create_render_job(
    request: JobCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        new_job = RenderJob(
            title=request.title,
            user_reference=request.user_reference,
            source_urls=[str(u) for u in request.research_inputs.source_urls],
            story_directives=request.story_directives.model_dump(mode="json"),
            format_type=request.format_type.value,
            platform=request.platform.value,
            status=JobStatusEnum.PENDING,
            device_id=request.device_id,
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
            f"Created RenderJob {job_to_return.id} for title: {job_to_return.title}"
        )
        return job_to_return

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database transaction failed: {str(e)}"
        )


@router.get(
    "",
    response_model=list[RenderJobResponse],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
@router.get(
    "/",
    response_model=list[RenderJobResponse],
    status_code=status.HTTP_200_OK,
)
async def list_jobs(
    status: Optional[JobStatusEnum] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    jobs, _total = await crud_list_jobs(db, status=status, limit=limit, offset=offset)
    return jobs


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

    master_scripts = [s for s in job.scripts if s.role == "master"]
    if not master_scripts:
        raise HTTPException(status_code=400, detail="No master script found to approve")
    latest_script = max(master_scripts, key=lambda s: s.version)

    if request.is_approved:
        latest_script.is_approved = True
        job.status = next_status_after_fact_check(job.format_type)
        await db.commit()

        stmt = (
            select(RenderJob)
            .options(
                selectinload(RenderJob.scripts).selectinload(Script.claims),
                selectinload(RenderJob.assets),
            )
            .filter(RenderJob.id == job.id)
        )
        result = await db.execute(stmt)
        job = result.unique().scalar_one()

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
            .options(
                selectinload(RenderJob.scripts).selectinload(Script.claims),
                selectinload(RenderJob.assets),
            )
            .filter(RenderJob.id == job.id)
        )
        result = await db.execute(stmt)
        job = result.unique().scalar_one()

    return job


@router.post(
    "/{job_id}/regenerate-assets",
    status_code=status.HTTP_200_OK,
)
async def regenerate_assets(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(RenderJob)
        .options(selectinload(RenderJob.scripts))
        .filter(RenderJob.id == job_id)
    )
    result = await db.execute(stmt)
    job = result.unique().scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="RenderJob not found")

    carousel_script = await get_latest_format_script(db, job_id, "CAROUSEL")

    if not carousel_script or not carousel_script.format_payload:
        raise HTTPException(
            status_code=400,
            detail="No carousel format script found to regenerate assets for",
        )

    agent = CarouselImageAgent()
    context = {
        "job_id": job.id,
        "format_payload": carousel_script.format_payload,
        "platform": job.platform or "instagram",
        "device_id": job.device_id,
    }
    carousel_result = await agent.run(context)

    if carousel_result.status == AgentActionStatus.SUCCESS:
        carousel_script.format_payload = merge_image_urls(
            carousel_script.format_payload,
            carousel_result.payload["format_payload"],
        )
        flag_modified(carousel_script, "format_payload")
        await db.commit()
        return {"status": "ok", **carousel_script.format_payload}

    raise HTTPException(
        status_code=500,
        detail=f"Asset regeneration failed: {carousel_result.reasoning}",
    )
