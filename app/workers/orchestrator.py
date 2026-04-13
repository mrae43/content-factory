import asyncio
import logging
import traceback
from uuid import UUID

from app.db.session import AsyncSessionLocal
from app.db.crud import get_render_job, update_job_status, log_error, save_script
from app.services.vector_store import ContentFactoryVectorStore
from app.services.chunking import process_extraction_job
from app.workers.tasks import cleanup_local_research_chunks
from app.workers.agents import (
    ResearchAgent,
    CopywriterAgent,
    RedTeamAgent,
    AssetStudioAgent,
    AgentActionStatus,
)
from app.schemas.shorts import JobStatusEnum

logger = logging.getLogger("factory.orchestrator")

# Maximum times the Red Team can reject a script before human intervention
MAX_RED_TEAM_REVISIONS = 3


async def run_content_factory_pipeline(job_id: UUID):
    """
    The Agentic State Machine.
    Manages its own DB session to ensure persistence even in long-running background tasks.
    """
    async with AsyncSessionLocal() as db:
        vector_store = ContentFactoryVectorStore(db)
        
        while True:
            # 1. Fetch latest state from DB
            job = await get_render_job(db, job_id)

            if not job:
                logger.error(f"Job {job_id} not found in DB.")
                break

            logger.info(f"Job {job_id} current status: {job.status}")

            try:
                # State Machine Transitions
                if job.status == JobStatusEnum.PENDING:
                    logger.info(f"Job {job_id}: Running Extraction (Text Chunking)")
                    raw_chunks = await process_extraction_job(str(job_id), job.pre_context)

                    if raw_chunks:
                        await vector_store.ingest_chunks(
                            job_id=job_id,
                            chunks=raw_chunks,
                            scope="RAW-CONTEXT"
                        )
                    else:
                        logger.warning(f"No raw chunks found for job {job_id}")

                    await update_job_status(db, job_id, JobStatusEnum.RESEARCHING)

                elif job.status == JobStatusEnum.RESEARCHING:
                    researcher = ResearchAgent(model_name="gemini-3.1-flash")
                    agent_context = {
                        "job_id": str(job.id),
                        "topic": job.topic,
                        "vector_store": vector_store,
                    }
                    result = await researcher.run(
                        context=agent_context
                    )

                    if result.status == AgentActionStatus.SUCCESS:
                        # Success! Move to Fact-Checking the research
                        await update_job_status(
                            db, job_id, JobStatusEnum.FACT_CHECKING_RESEARCH
                        )
                    else:
                        raise Exception(f"Research failed: {result.reasoning}")

                elif job.status == JobStatusEnum.FACT_CHECKING_RESEARCH:
                    # Step 4: Rapid baseline check
                    # MVP: Assume research is verified for now
                    await update_job_status(db, job_id, JobStatusEnum.SCRIPTING)

                elif job.status == JobStatusEnum.SCRIPTING:
                    copywriter = CopywriterAgent(
                        model_name="gemini-3.1-pro", temperature=0.7
                    )
                    agent_context = {
                        "job_id": str(job.id),
                        "topic": job.topic,
                        "vector_store": vector_store,
                    }
                    result = await copywriter.run(
                        context=agent_context
                    )

                    if result.status == AgentActionStatus.SUCCESS:
                        # Save script version to DB
                        version = len(job.scripts) + 1
                        await save_script(
                            db, job_id, result.payload["script_content"], version
                        )
                        await update_job_status(
                            db, job_id, JobStatusEnum.FACT_CHECKING_SCRIPT
                        )
                    else:
                        raise Exception(f"Scripting failed: {result.reasoning}")

                elif job.status == JobStatusEnum.FACT_CHECKING_SCRIPT:
                    # THE RED TEAM EVALUATOR
                    red_team = RedTeamAgent(
                        model_name="gemini-3.1-pro", temperature=0.0
                    )
                    
                    # Sort scripts to get the latest one
                    latest_script = sorted(job.scripts, key=lambda s: s.version)[-1].content if job.scripts else ""
                    
                    agent_context = {
                        "job_id": str(job.id),
                        "script_content": latest_script,
                        "vector_store": vector_store,
                    }
                    result = await red_team.run(context=agent_context)

                    if result.status == AgentActionStatus.SUCCESS:
                        logger.info(
                            f"Red Team Approved for Job {job_id}. Proceeding to Asset Gen."
                        )
                        await update_job_status(
                            db, job_id, JobStatusEnum.ASSET_GENERATION
                        )

                    elif result.status == AgentActionStatus.REVISION_NEEDED:
                        # CRASH-RESILIENT REVISION TRACKING:
                        # We use the actual number of scripts in the DB to determine revisions
                        current_revision = len(job.scripts)
                        logger.warning(
                            f"Red Team Rejected Job {job_id}. Revision {current_revision}/{MAX_RED_TEAM_REVISIONS}"
                        )

                        if current_revision >= MAX_RED_TEAM_REVISIONS:
                            logger.error(
                                f"Max revisions reached for Job {job_id}. Escalating."
                            )
                            await update_job_status(
                                db, job_id, JobStatusEnum.HUMAN_REVIEW_NEEDED
                            )
                            break
                        else:
                            await update_job_status(db, job_id, JobStatusEnum.SCRIPTING)

                elif job.status == JobStatusEnum.ASSET_GENERATION:
                    studio = AssetStudioAgent(model_name="gemini-3.1-multimodal")
                    result = await studio.run(context={"job_id": job_id})

                    if result.status == AgentActionStatus.SUCCESS:
                        # Update with final video URL (mocked)
                        job.final_video_url = result.payload["video_url"]
                        await db.commit()
                        await update_job_status(db, job_id, JobStatusEnum.COMPLETED)

                elif job.status == JobStatusEnum.COMPLETED:
                    logger.info(f"Pipeline finished successfully for Job {job_id}")
                    await cleanup_local_research_chunks(job_id, db)
                    break

                elif job.status in [
                    JobStatusEnum.HUMAN_REVIEW_NEEDED,
                    JobStatusEnum.FAILED,
                ]:
                    logger.warning(
                        f"Pipeline paused/stopped for Job {job_id} at {job.status}"
                    )
                    break

                else:
                    logger.error(f"Unrecognized status '{job.status}' for Job {job_id}")
                    break

            except Exception as e:
                logger.exception(f"Fatal error in orchestrator for Job {job_id}")
                await log_error(
                    db,
                    job_id,
                    f"{str(e)}\n{traceback.format_exc()}",
                    phase=str(job.status),
                )
                await update_job_status(db, job_id, JobStatusEnum.FAILED)
                break


async def resume_pipeline_after_approval(job_id: UUID):
    """
    Entry point for human intervention to restart the loop.
    """
    async with AsyncSessionLocal() as db:
        job = await get_render_job(db, job_id)
        if job and job.status == JobStatusEnum.HUMAN_REVIEW_NEEDED:
            logger.info(
                f"Manual override for Job {job_id}. Restarting at Asset Generation."
            )
            await update_job_status(db, job_id, JobStatusEnum.ASSET_GENERATION)
            # Fire and forget the background task
            asyncio.create_task(run_content_factory_pipeline(job_id))
