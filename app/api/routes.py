from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from uuid import UUID
import logging

# Import schemas and models
from app.schemas.shorts import JobCreateRequest, RenderJobResponse, ScriptApprovalRequest, JobStatusEnum
from app.db.models import RenderJob, Script
from app.db.session import get_db

# Import agentic orchestrator
from app.workers.orchestrator import run_content_factory_pipeline, resume_pipeline_after_approval

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["Content Factory"])

# ==========================================
# 1. CREATE JOB & TRIGGER PIPELINE
# ==========================================
@router.post(
    "/", 
    response_model=RenderJobResponse, 
    status_code=status.HTTP_202_ACCEPTED
)
def create_render_job(
  request: JobCreateRequest,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_db)
):
  """
  Step 1: User submits a topic and context.
  Creates a new RenderJob and triggers the Agentic background pipeline.
  """
  try:
    # Create DB record using Pydantic validation
    new_job = RenderJob(
      topic=request.topic,
      pre_context=request.pre_context.model_dump(mode='json'),
      strict_compliance_mode=request.strict_compliance_mode,
      status=JobStatusEnum.PENDING
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Trigger the Agentic Workflow in the background
    # (In production, you'd push this to a Celery/ARQ queue. 
    # Using FastAPI BackgroundTasks here for architecture flow.)
    background_tasks.add_task(run_content_factory_pipeline, job_id=new_job.id)

    logger.info(f"Created RenderJob {new_job.id} for topic: {new_job.topic}")
    return new_job

  except Exception as e:
    db.rollback()
    logger.error(f"Failed to create job: {str(e)}")
    raise HTTPException(status_code=500, detail="Database transaction failed")

# ==========================================
# 2. POLL JOB STATUS
# ==========================================
@router.get(
    "/{job_id}", 
    response_model=RenderJobResponse, 
    status_code=status.HTTP_200_OK
)
def get_render_job(job_id: UUID, db: Session = Depends(get_db)):
  """
  Step 8: Check status. 
  Eager loads scripts, claims, and assets to prevent N+1 query performance hits.
  """
  job = (
        db.query(RenderJob)
        .options(
            joinedload(RenderJob.scripts).joinedload(Script.claims),
            joinedload(RenderJob.assets)
        )
        .filter(RenderJob.id == job_id)
        .first()
    )

  if not job:
    raise HTTPException(status_code=404, detail="RenderJob not found")

  return job

# ==========================================
# 3. HUMAN-IN-THE-LOOP APPROVAL (Red Team Override)
# ==========================================
@router.post(
    "/{job_id}/approve-script", 
    response_model=RenderJobResponse,
    status_code=status.HTTP_200_OK
)
def approve_script(
    job_id: UUID, 
    request: ScriptApprovalRequest, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
  """
  Step 6 Override: If the Red Team agent flags the script (HUMAN_REVIEW_NEEDED),
  an editor uses this endpoint to force approval, inject feedback, or reject.
  """
  # Fetch job with active scripts
  job = (
      db.query(RenderJob)
      .options(joinedload(RenderJob.scripts))
      .filter(RenderJob.id == job_id)
      .first()
  )

  if not job:
    raise HTTPException(status_code=404, detail="RenderJob not found")
  
  if job.status not in [JobStatusEnum.HUMAN_REVIEW_NEEDED, JobStatusEnum.FACT_CHECKING_SCRIPT]:
    raise HTTPException(
        status_code=400, 
        detail=f"Job is in '{job.status}', cannot approve script at this stage."
    )

  # Find the latest script that needs review
  if not job.scripts:
    raise HTTPException(status_code=400, detail="No script found to approve")

  latest_script = max(job.scripts, key=lambda s: s.version)

  if request.is_approved:
    # Move forward
    latest_script.is_approved = True
    job.status = JobStatusEnum.ASSET_GENERATION
    
    db.commit()
    db.refresh(job)
    
    # Resume the pipeline (triggering Veo, Lyria, Python Charts)
    background_tasks.add_task(resume_pipeline_after_approval, job_id=job.id)
        
  else:
    # Send back to Scripting phase with human feedback
    job.status = JobStatusEnum.SCRIPTING
    
    # Append human feedback to the Agentic Loop history
    if request.human_feedback:
      feedback_history = latest_script.feedback_history or []
      feedback_history.append({"source": "human_editor", "comment": request.human_feedback})
      latest_script.feedback_history = feedback_history
      
    db.commit()
        db.refresh(job)
        
        # Trigger the pipeline to re-run the copywriter agent
        background_tasks.add_task(run_content_factory_pipeline, job_id=job.id)

  return job