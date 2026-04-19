import logging
import traceback
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import (
    update_job_status,
    log_error,
    save_script,
    get_latest_script,
    append_script_feedback,
    save_fact_check_claims,
)
from app.services.vector_store import ContentFactoryVectorStore
from app.services.web_search import TavilySearchService
from app.services.chunking import process_extraction_job
from app.workers.tasks import cleanup_local_research_chunks
from app.workers.agents import (
    ResearchAgent,
    CopywriterAgent,
    RedTeamAgent,
    AssetStudioAgent,
    AgentActionStatus,
)
from app.workers.optimizer import ScriptOptimizerAgent
from app.schemas.shorts import JobStatusEnum
from app.core.config import settings

logger = logging.getLogger("factory.orchestrator")

_web_search_service = TavilySearchService()


async def execute_state_transition(db: AsyncSession, job) -> None:
    """
    Execute ONE state transition for the given job.
    Called by QueueWorker per poll cycle. The session is managed externally.
    """
    logger.info(f"Job {job.id} current status: {job.status}")

    try:
        if job.status == JobStatusEnum.PENDING:
            await _transition_pending(db, job)

        elif job.status == JobStatusEnum.RESEARCHING:
            await _transition_researching(db, job)

        elif job.status == JobStatusEnum.FACT_CHECKING_RESEARCH:
            await update_job_status(db, job.id, JobStatusEnum.SCRIPTING)

        elif job.status == JobStatusEnum.SCRIPTING:
            await _transition_scripting(db, job)

        elif job.status == JobStatusEnum.FACT_CHECKING_SCRIPT:
            await _transition_fact_checking_script(db, job)

        elif job.status == JobStatusEnum.ASSET_GENERATION:
            await _transition_asset_generation(db, job)

        elif job.status == JobStatusEnum.COMPLETED:
            logger.info(f"Pipeline finished successfully for Job {job.id}")
            await cleanup_local_research_chunks(job.id, db)

        elif job.status in [
            JobStatusEnum.HUMAN_REVIEW_NEEDED,
            JobStatusEnum.FAILED,
        ]:
            logger.warning(f"Pipeline paused/stopped for Job {job.id} at {job.status}")

        else:
            logger.error(f"Unrecognized status '{job.status}' for Job {job.id}")

    except Exception as e:
        logger.exception(f"Fatal error in orchestrator for Job {job.id}")
        await log_error(
            db,
            job.id,
            f"{str(e)}\n{traceback.format_exc()}",
            phase=str(job.status),
        )
        await update_job_status(db, job.id, JobStatusEnum.FAILED)


async def _transition_pending(db: AsyncSession, job) -> None:
    logger.info(f"Job {job.id}: Running Extraction (Text Chunking)")
    raw_text = (
        job.pre_context.get("raw_text", "")
        if isinstance(job.pre_context, dict)
        else str(job.pre_context)
    )
    raw_chunks = await process_extraction_job(str(job.id), raw_text)

    if raw_chunks:
        vector_store = ContentFactoryVectorStore(db)
        await vector_store.ingest_chunks(
            job_id=job.id, chunks=raw_chunks, scope="RAW-CONTEXT"
        )
    else:
        logger.warning(f"No raw chunks found for job {job.id}")

    await update_job_status(db, job.id, JobStatusEnum.RESEARCHING)


async def _transition_researching(db: AsyncSession, job) -> None:
    vector_store = ContentFactoryVectorStore(db)

    web_service = _web_search_service
    web_results = await web_service.search(job.topic)

    if web_results:
        valid_results = [r for r in web_results if r.get("content")]
        web_texts = [r["content"] for r in valid_results]
        web_urls = [r.get("url", "") for r in valid_results]
        if web_texts:
            logger.info(
                f"Ingesting {len(web_texts)} web search results for Job {job.id}"
            )
            await vector_store.ingest_chunks(
                job_id=job.id,
                chunks=web_texts,
                scope="LOCAL",
                meta={
                    "source": "web_search",
                    "query": job.topic,
                    "urls": web_urls,
                    "search_depth": "basic",
                },
            )

    researcher = ResearchAgent(model_name="gemini-2.5-flash")
    agent_context = {
        "job_id": job.id,
        "topic": job.topic,
        "vector_store": vector_store,
    }
    result = await researcher.run(context=agent_context)

    if result.status == AgentActionStatus.SUCCESS:
        refined_context = result.payload.get("refined_context", "")
        if refined_context:
            job.refined_context = refined_context
            await db.commit()
        await update_job_status(db, job.id, JobStatusEnum.FACT_CHECKING_RESEARCH)
    else:
        raise Exception(f"Research failed: {result.reasoning}")


async def _transition_scripting(db: AsyncSession, job) -> None:
    latest_script = await get_latest_script(db, job.id)

    if latest_script and latest_script.feedback_history:
        last_feedback = latest_script.feedback_history[-1]

        if (
            isinstance(last_feedback, dict)
            and last_feedback.get("feedback_type") == "structured_claims"
        ):
            failed_claims = last_feedback.get("failed_claims", [])
            await _run_optimizer(db, job, latest_script, failed_claims)
            return
        else:
            revision_feedback = (
                last_feedback
                if isinstance(last_feedback, str)
                else last_feedback.get("feedback", "")
            )
            await _run_copywriter(db, job, feedback=revision_feedback)
            return

    await _run_copywriter(db, job)


async def _run_copywriter(db: AsyncSession, job, feedback: str = "") -> None:
    copywriter = CopywriterAgent(model_name="gemini-1.5-pro", temperature=0.7)
    agent_context = {
        "job_id": job.id,
        "topic": job.topic,
        "refined_context": job.refined_context or "",
        "feedback": feedback,
    }
    result = await copywriter.run(context=agent_context)

    if result.status == AgentActionStatus.SUCCESS:
        latest = await get_latest_script(db, job.id)
        version = (latest.version + 1) if latest else 1
        await save_script(db, job.id, result.payload["script_content"], version)
        await update_job_status(db, job.id, JobStatusEnum.FACT_CHECKING_SCRIPT)
    else:
        raise Exception(f"Copywriter failed: {result.reasoning}")


async def _run_optimizer(
    db: AsyncSession, job, latest_script, failed_claims: list
) -> None:
    optimizer = ScriptOptimizerAgent(
        model_name=settings.optimizer_model,
        temperature=settings.optimizer_temperature,
    )
    agent_context = {
        "job_id": job.id,
        "script_content": latest_script.content,
        "failed_claims": failed_claims,
        "refined_context": job.refined_context or "",
    }
    result = await optimizer.run(context=agent_context)

    if result.status == AgentActionStatus.SUCCESS:
        version = latest_script.version + 1
        await save_script(db, job.id, result.payload["script_content"], version)
        await update_job_status(db, job.id, JobStatusEnum.FACT_CHECKING_SCRIPT)
    else:
        raise Exception(f"Optimizer failed: {result.reasoning}")


async def _transition_fact_checking_script(db: AsyncSession, job) -> None:
    red_team = RedTeamAgent(model_name="gemini-1.5-pro", temperature=0.0)
    vector_store = ContentFactoryVectorStore(db)

    latest_script_obj = await get_latest_script(db, job.id)
    latest_script = latest_script_obj.content if latest_script_obj else ""

    agent_context = {
        "job_id": job.id,
        "script_content": latest_script,
        "vector_store": vector_store,
    }
    result = await red_team.run(context=agent_context)

    if result.status == AgentActionStatus.SUCCESS:
        claims_data = result.payload.get("claims", [])
        await _resolve_evidence_refs(db, vector_store, job.id, claims_data)

        if claims_data and latest_script_obj:
            await save_fact_check_claims(db, latest_script_obj.id, claims_data)

        if latest_script_obj:
            latest_script_obj.is_approved = True
            await db.commit()

        logger.info(
            f"Red Team Approved for Job {job.id}. "
            f"{len(claims_data)} claims persisted. Proceeding to Asset Gen."
        )
        await update_job_status(db, job.id, JobStatusEnum.ASSET_GENERATION)

    elif result.status == AgentActionStatus.REVISION_NEEDED:
        claims_data = result.payload.get("claims", [])
        await _resolve_evidence_refs(db, vector_store, job.id, claims_data)

        if claims_data and latest_script_obj:
            await save_fact_check_claims(db, latest_script_obj.id, claims_data)
            await db.commit()

        failed_claims = [
            c for c in claims_data if c.get("verdict") in ("UNSUPPORTED", "CONTESTED")
        ]

        current_revision = latest_script_obj.version if latest_script_obj else 0
        logger.warning(
            f"Red Team Rejected Job {job.id}. Revision {current_revision}/{settings.max_red_team_revisions}"
        )

        if current_revision >= settings.max_red_team_revisions:
            logger.error(f"Max revisions reached for Job {job.id}. Escalating.")
            await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)
        else:
            await append_script_feedback(
                db,
                job.id,
                feedback=result.reasoning,
                structured_claims=failed_claims,
                overall_reasoning=result.reasoning,
                revision_number=current_revision,
            )
            await update_job_status(db, job.id, JobStatusEnum.SCRIPTING)

    elif result.status == AgentActionStatus.ESCALATE:
        logger.error(f"Red Team escalated Job {job.id}: {result.reasoning}")
        await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)


async def _transition_asset_generation(db: AsyncSession, job) -> None:
    studio = AssetStudioAgent(model_name="gemini-2.5-flash")
    result = await studio.run(context={"job_id": job.id})

    if result.status == AgentActionStatus.SUCCESS:
        job.final_video_url = result.payload["video_url"]
        await db.commit()
        await update_job_status(db, job.id, JobStatusEnum.COMPLETED)


async def _resolve_evidence_refs(
    db: AsyncSession, vector_store, job_id: UUID, claims_data: list
) -> None:
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
                if m.get("similarity_score", 0) >= settings.similarity_threshold
            ]
