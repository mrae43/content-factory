import asyncio
import logging
import traceback
from uuid import UUID

from app.db.session import AsyncSessionLocal
from app.db.crud import (
    get_render_job,
    update_job_status,
    log_error,
    save_script,
    get_latest_script,
    append_script_feedback,
    save_fact_check_claims,
)
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
from app.core.config import settings

logger = logging.getLogger("factory.orchestrator")


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
                    raw_text = (
                        job.pre_context.get("raw_text", "")
                        if isinstance(job.pre_context, dict)
                        else str(job.pre_context)
                    )
                    raw_chunks = await process_extraction_job(str(job_id), raw_text)

                    if raw_chunks:
                        await vector_store.ingest_chunks(
                            job_id=job_id, chunks=raw_chunks, scope="RAW-CONTEXT"
                        )
                    else:
                        logger.warning(f"No raw chunks found for job {job_id}")

                    await update_job_status(db, job_id, JobStatusEnum.RESEARCHING)

                elif job.status == JobStatusEnum.RESEARCHING:
                    researcher = ResearchAgent(model_name="gemini-3.1-flash")
                    agent_context = {
                        "job_id": job.id,
                        "topic": job.topic,
                        "vector_store": vector_store,
                    }
                    result = await researcher.run(context=agent_context)

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
                    latest_script_for_feedback = await get_latest_script(db, job_id)
                    revision_feedback = ""
                    if (
                        latest_script_for_feedback
                        and latest_script_for_feedback.feedback_history
                    ):
                        last_entry = latest_script_for_feedback.feedback_history[-1]
                        revision_feedback = (
                            last_entry
                            if isinstance(last_entry, str)
                            else last_entry.get("feedback", "")
                        )
                    agent_context = {
                        "job_id": job.id,
                        "topic": job.topic,
                        "vector_store": vector_store,
                        "feedback": revision_feedback,
                    }
                    result = await copywriter.run(context=agent_context)

                    if result.status == AgentActionStatus.SUCCESS:
                        latest_script_for_version = await get_latest_script(db, job_id)
                        version = (
                            (latest_script_for_version.version + 1)
                            if latest_script_for_version
                            else 1
                        )
                        await save_script(
                            db, job_id, result.payload["script_content"], version
                        )
                        await update_job_status(
                            db, job_id, JobStatusEnum.FACT_CHECKING_SCRIPT
                        )
                    else:
                        raise Exception(f"Scripting failed: {result.reasoning}")

                elif job.status == JobStatusEnum.FACT_CHECKING_SCRIPT:
                    red_team = RedTeamAgent(
                        model_name="gemini-3.1-pro", temperature=0.0
                    )

                    latest_script_obj = await get_latest_script(db, job_id)
                    latest_script = (
                        latest_script_obj.content if latest_script_obj else ""
                    )

                    agent_context = {
                        "job_id": job.id,
                        "script_content": latest_script,
                        "vector_store": vector_store,
                    }
                    result = await red_team.run(context=agent_context)

                    if result.status == AgentActionStatus.SUCCESS:
                        claims_data = result.payload.get("claims", [])
                        for claim in claims_data:
                            evidence_text = claim.get("evidence_text", "")
                            claim["evidence_references"] = []
                            if evidence_text:
                                matches = await vector_store.semantic_search(
                                    query=evidence_text,
                                    job_id=job_id,
                                    scopes=["RAW-CONTEXT", "LOCAL"],
                                    top_k=3,
                                )
                                claim["evidence_references"] = [
                                    str(m["id"])
                                    for m in matches
                                    if m.get("similarity_score", 0)
                                    >= settings.similarity_threshold
                                ]

                        if claims_data and latest_script_obj:
                            await save_fact_check_claims(
                                db, latest_script_obj.id, claims_data
                            )

                        if latest_script_obj:
                            latest_script_obj.is_approved = True
                            await db.commit()

                        logger.info(
                            f"Red Team Approved for Job {job_id}. "
                            f"{len(claims_data)} claims persisted. Proceeding to Asset Gen."
                        )
                        await update_job_status(
                            db, job_id, JobStatusEnum.ASSET_GENERATION
                        )

                    elif result.status == AgentActionStatus.REVISION_NEEDED:
                        claims_data = result.payload.get("claims", [])
                        for claim in claims_data:
                            evidence_text = claim.get("evidence_text", "")
                            claim["evidence_references"] = []
                            if evidence_text:
                                matches = await vector_store.semantic_search(
                                    query=evidence_text,
                                    job_id=job_id,
                                    scopes=["RAW-CONTEXT", "LOCAL"],
                                    top_k=3,
                                )
                                claim["evidence_references"] = [
                                    str(m["id"])
                                    for m in matches
                                    if m.get("similarity_score", 0)
                                    >= settings.similarity_threshold
                                ]

                        if claims_data and latest_script_obj:
                            await save_fact_check_claims(
                                db, latest_script_obj.id, claims_data
                            )
                            await db.commit()

                        current_revision = (
                            latest_script_obj.version if latest_script_obj else 0
                        )
                        logger.warning(
                            f"Red Team Rejected Job {job_id}. Revision {current_revision}/{settings.max_red_team_revisions}"
                        )

                        if current_revision >= settings.max_red_team_revisions:
                            logger.error(
                                f"Max revisions reached for Job {job_id}. Escalating."
                            )
                            await update_job_status(
                                db, job_id, JobStatusEnum.HUMAN_REVIEW_NEEDED
                            )
                            break
                        else:
                            await append_script_feedback(db, job_id, result.reasoning)
                            await update_job_status(db, job_id, JobStatusEnum.SCRIPTING)

                    elif result.status == AgentActionStatus.ESCALATE:
                        logger.error(
                            f"Red Team escalated Job {job_id}: {result.reasoning}"
                        )
                        await update_job_status(
                            db, job_id, JobStatusEnum.HUMAN_REVIEW_NEEDED
                        )
                        break

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
