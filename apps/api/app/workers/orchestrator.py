import asyncio
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified
from app.db.crud import (
    update_job_status,
    log_error,
    save_script,
    save_format_script,
    get_latest_script,
    get_latest_format_script,
    get_script_claims,
    append_script_feedback,
    save_fact_check_claims,
)
from app.services.vector_store import (
    ContentFactoryVectorStore,
    make_ingest_chunks_tool,
    make_semantic_search_tool,
)
from app.services.tools import ToolRegistry
from app.services.web_search import get_tavily_service
from app.services.chunking import process_extraction_job
from app.services.claim_mapper import (
    init_ledger,
    map_claims,
    compute_verdict_delta,
    update_ledger,
)
from app.services.llm import get_embeddings, get_llm
from app.services.format_validator import (
    BlogValidator,
    CarouselValidator,
    VideoValidator,
)
from app.workers.tasks import cleanup_local_research_chunks
from app.workers.agents import (
    CopywriterAgent,
    RedTeamAgent,
    AssetStudioAgent,
    AgentActionStatus,
)
from app.workers.optimizer import ScriptOptimizerAgent
from app.workers.carousel_image_agent import CarouselImageAgent, merge_image_urls
from app.workers.formatters import (
    BlogFormatterAgent,
    CarouselFormatterAgent,
    VideoFormatterAgent,
)
from app.workers.harness import AgentHarness
from app.schemas.shorts import (
    JobStatusEnum,
    AssembledContext,
    next_status_after_fact_check,
    resolve_formats,
    FormatTypeEnum,
    PlatformEnum,
)
from app.services.context_builder import build as _build_context_from_service
from app.services.optimizer_tools import make_gated_search_tool
from app.core.config import settings
from app.core.guardrails import get_guardrail_config, GuardrailStrictness

logger = logging.getLogger("factory.orchestrator")

_VECTOR_STORE_ATTR = "_transition_vector_store"


def _get_vector_store(db: AsyncSession) -> ContentFactoryVectorStore:
    """Obtain a ``ContentFactoryVectorStore`` for the current transition.

    Creates a fresh instance.  Once the orchestrator gains access to a
    pre-configured vector store via the tool registry this helper can be
    replaced with a registry lookup.
    """
    return ContentFactoryVectorStore(db)


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

        elif job.status == JobStatusEnum.RETRIEVAL:
            await _transition_retrieval(db, job)

        elif job.status == JobStatusEnum.FACT_CHECKING_RESEARCH:
            logger.warning(
                f"Legacy state FACT_CHECKING_RESEARCH for Job {job.id} — forwarding to SCRIPTING"
            )
            await update_job_status(db, job.id, JobStatusEnum.SCRIPTING)

        elif job.status == JobStatusEnum.SCRIPTING:
            await _transition_scripting(db, job)

        elif job.status == JobStatusEnum.FACT_CHECKING_SCRIPT:
            await _transition_fact_checking_script(db, job)

        elif job.status == JobStatusEnum.FORMATTING:
            await _transition_formatting(db, job)

        elif job.status == JobStatusEnum.ASSET_GENERATION:
            await _transition_asset_generation(db, job)

        elif job.status == JobStatusEnum.COMPLETED:
            logger.info(f"Pipeline finished successfully for Job {job.id}")
            await _promote_to_global(db, job)
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
    raw_text = job.user_reference or ""
    raw_chunks = await process_extraction_job(str(job.id), raw_text)

    if raw_chunks:
        vs = _get_vector_store(db)
        ingest_tool = make_ingest_chunks_tool(vs)
        await ingest_tool.callable(
            job_id=job.id,
            chunks=raw_chunks,
            scope="RAW-CONTEXT",
            meta={"source_type": "USER_PROVIDED"},
        )
    else:
        logger.warning(f"No raw chunks found for job {job.id}")

    await update_job_status(db, job.id, JobStatusEnum.RESEARCHING)


async def _transition_researching(db: AsyncSession, job) -> None:
    vs = _get_vector_store(db)
    ingest_tool = make_ingest_chunks_tool(vs)

    web_service = get_tavily_service()

    # 1. Tavily search by title
    web_results = await web_service.search(job.title)

    if web_results:
        valid_results = [r for r in web_results if r.get("content")]
        web_texts = [r["content"] for r in valid_results]
        web_urls = [r.get("url", "") for r in valid_results]
        if web_texts:
            logger.info(
                f"Ingesting {len(web_texts)} web search results for Job {job.id}"
            )
            await ingest_tool.callable(
                job_id=job.id,
                chunks=web_texts,
                scope="LOCAL",
                meta={
                    "source_type": "WEB_SEARCH",
                    "query": job.title,
                    "urls": web_urls,
                    "search_depth": "basic",
                },
            )

    # 2. Tavily extract from user-provided source URLs
    source_urls = job.source_urls or []
    if source_urls:
        extracted = await web_service.extract(source_urls)
        if extracted:
            valid_extracted = [r for r in extracted if r.get("content")]
            ext_texts = [r["content"] for r in valid_extracted]
            ext_urls = [r.get("url", "") for r in valid_extracted]
            if ext_texts:
                logger.info(
                    f"Ingesting {len(ext_texts)} URL extraction results for Job {job.id}"
                )
                await ingest_tool.callable(
                    job_id=job.id,
                    chunks=ext_texts,
                    scope="LOCAL",
                    meta={
                        "source_type": "URL_EXTRACT",
                        "urls": ext_urls,
                    },
                )

    await update_job_status(db, job.id, JobStatusEnum.RETRIEVAL)


def _format_narrative(user_reference: str, story_directives: dict) -> str:
    """Format user reference + story directives into the refined_context narrative."""
    truncated_ref = (user_reference or "")[:2000]
    parts = [truncated_ref] if truncated_ref else []
    entries = []
    for key in ("target_audience", "tone", "angle"):
        val = story_directives.get(key, "")
        if val:
            entries.append(f"{key}: {val}")
    if entries:
        parts.append("Editorial Directives: " + "; ".join(entries))
    return "\n\n".join(parts) if parts else ""


async def _transition_retrieval(db: AsyncSession, job) -> None:
    # Build narrative directly from user_reference + story_directives (no ResearchAgent)
    job.refined_context = _format_narrative(
        job.user_reference, job.story_directives or {}
    )
    if not job.refined_context:
        raise Exception(
            "No refined_context could be built — user_reference is empty. "
            "Cannot proceed to scripting without a narrative foundation."
        )

    await db.commit()

    assembled = await _build_script_context(db, job)
    job.assembled_context = assembled.model_dump()
    await db.commit()
    await update_job_status(db, job.id, JobStatusEnum.SCRIPTING)


async def _build_script_context(db: AsyncSession, job) -> AssembledContext:
    try:
        vector_store = _get_vector_store(db)
        story_directives = job.story_directives or {}
        return await _build_context_from_service(
            title=job.title,
            story_directives=story_directives,
            refined_context=job.refined_context or "",
            vector_store=vector_store,
            job_id=job.id,
            top_k=settings.context_builder_top_k,
            user_reference=job.user_reference or "",
        )
    except Exception:
        logger.exception(
            f"ContextBuilder failed for Job {job.id} — continuing with empty evidence"
        )
        return AssembledContext(
            narrative_summary=job.refined_context or "",
            evidence_sections="",
            raw_chunks=[],
        )


async def _transition_scripting(db: AsyncSession, job) -> None:
    if not job.assembled_context:
        retry_count = (job.retrieval_retry_count or 0) + 1
        if retry_count > settings.retrieval_retry_max:
            raise Exception(
                f"assembled_context still None after {retry_count} "
                f"RETRIEVAL attempts for Job {job.id}"
            )
        job.retrieval_retry_count = retry_count
        await db.commit()
        await update_job_status(db, job.id, JobStatusEnum.RETRIEVAL)
        return

    assembled = AssembledContext(**job.assembled_context)
    evidence_sections = assembled.evidence_sections

    latest_script = await get_latest_script(db, job.id)

    if latest_script and latest_script.feedback_history:
        last_feedback = latest_script.feedback_history[-1]

        if (
            isinstance(last_feedback, dict)
            and last_feedback.get("feedback_type") == "structured_claims"
        ):
            all_claims = await get_script_claims(db, latest_script.id)
            red_team_evidence = {
                c["claim_text"]: {
                    "evidence_text": c["evidence_text"],
                    "evidence_references": c["evidence_references"],
                    "confidence": c["confidence"],
                    "verdict": c["verdict"],
                }
                for c in all_claims
                if c["verdict"] in ("UNSUPPORTED", "CONTESTED", "UNCERTAIN")
            }
            await _run_optimizer(
                db,
                job,
                latest_script,
                evidence_sections=evidence_sections,
                red_team_evidence=red_team_evidence,
            )
            return
        else:
            revision_feedback = (
                last_feedback
                if isinstance(last_feedback, str)
                else last_feedback.get("feedback", "")
            )
            await _run_copywriter(
                db,
                job,
                feedback=revision_feedback,
                evidence_sections=evidence_sections,
            )
            return

    await _run_copywriter(db, job, evidence_sections=evidence_sections)


async def _run_copywriter(
    db: AsyncSession,
    job,
    feedback: str = "",
    evidence_sections: str = "",
) -> None:
    copywriter = CopywriterAgent(
        model_name=settings.copywriter_model,
        temperature=settings.copywriter_temperature,
    )
    harness = AgentHarness(agent=copywriter)
    story_directives = job.story_directives or {}
    working_memory = job.working_memory or {}
    copywriter_rationale = working_memory.get("copywriter_rationale")
    agent_context = {
        "job_id": job.id,
        "topic": job.title,
        "refined_context": job.refined_context or "",
        "evidence_sections": evidence_sections,
        "story_directives": {
            "target_audience": story_directives.get("target_audience", "General"),
            "tone": story_directives.get("tone", ""),
            "angle": story_directives.get("angle", ""),
        },
        "feedback": feedback,
    }
    if copywriter_rationale:
        agent_context["copywriter_rationale"] = copywriter_rationale
    result = await harness.run_with_harness(agent_context)

    if result.success:
        working_memory = dict(job.working_memory or {})
        rationale = result.payload.get("copywriter_rationale")
        if rationale:
            working_memory["copywriter_rationale"] = rationale
            job.working_memory = working_memory
        latest = await get_latest_script(db, job.id)
        version = (latest.version + 1) if latest else 1
        opt_history = latest.optimization_history if latest else None
        await save_script(
            db,
            job.id,
            result.payload["script_content"],
            version,
            optimization_history=opt_history,
        )
        await update_job_status(db, job.id, JobStatusEnum.FACT_CHECKING_SCRIPT)
    else:
        error_msg = result.error_log[-1] if result.error_log else "Unknown error"
        logger.warning(
            f"Copywriter {result.error_log[0] if result.error_log else 'FAILED'} for Job {job.id}: {error_msg}"
        )
        await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)


async def _run_optimizer(
    db: AsyncSession,
    job,
    latest_script,
    evidence_sections: str = "",
    red_team_evidence: dict | None = None,
) -> None:
    optimizer = ScriptOptimizerAgent(
        model_name=settings.optimizer_model,
        temperature=settings.optimizer_temperature,
    )
    story_directives = job.story_directives or {}
    ledger = latest_script.optimization_history or {}
    active_failures = [
        c
        for c in ledger.get("active_claims", [])
        if c.get("latest_verdict") in ("UNSUPPORTED", "CONTESTED", "UNCERTAIN")
    ]
    optimization_history = ledger.get("historical_iterations", [])
    working_memory = job.working_memory or {}
    optimizer_phase = working_memory.get("optimizer_phase", {})

    red_team_evidence = red_team_evidence or {}
    vector_store = _get_vector_store(db)
    gated_tool = make_gated_search_tool(
        vector_store=vector_store,
        red_team_evidence=red_team_evidence,
        job_id=job.id,
        top_k=3,
    )
    registry = ToolRegistry()
    registry.register(gated_tool, replace=True)
    harness = AgentHarness(agent=optimizer)
    registry.unregister("retrieve_evidence_for_claim")

    agent_context = {
        "job_id": job.id,
        "script_content": latest_script.content,
        "active_failures": active_failures,
        "optimization_history": optimization_history,
        "refined_context": job.refined_context or "",
        "evidence_sections": evidence_sections,
        "red_team_evidence": red_team_evidence,
        "story_directives": {
            "target_audience": story_directives.get("target_audience", "General"),
            "tone": story_directives.get("tone", ""),
            "angle": story_directives.get("angle", ""),
        },
    }
    if optimizer_phase:
        agent_context["optimizer_history_phases"] = optimizer_phase
    result = await harness.run_with_harness(agent_context)

    fallback_count = getattr(gated_tool.callable, "fallback_count", [0])[0]
    total_failed = len(active_failures) if active_failures else 0
    fallback_rate = fallback_count / total_failed if total_failed > 0 else 0.0
    logger.info(
        f"Optimizer fallback rate for Job {job.id}: "
        f"{fallback_count}/{total_failed} = {fallback_rate:.2%}"
    )

    if result.success:
        working_memory = dict(job.working_memory or {})
        per_claim_patches = result.payload.get("per_claim_patches", [])

        if per_claim_patches:
            optimizer_phase = working_memory.setdefault("optimizer_phase", {})
            iteration = len(optimizer_phase) + 1

            resolved_claims: List[dict] = []
            ledger = latest_script.optimization_history or {}
            active_claims = ledger.get("active_claims", [])
            text_to_uuid = {c["claim_text"]: c["claim_uuid"] for c in active_claims}
            for patch in per_claim_patches:
                claim_uuid = text_to_uuid.get(patch["original_claim_text"])
                resolved_claims.append(
                    {
                        "claim_uuid": str(claim_uuid)
                        if claim_uuid
                        else patch["original_claim_text"],
                        "patch_intent": patch["patch_intent"],
                        "is_completely_resolved": patch["is_completely_resolved"],
                    }
                )

            optimizer_phase[f"iteration_{iteration}"] = {
                "patch_summary": result.payload.get("patch_summary", ""),
                "resolved_claims": resolved_claims,
                "fallback_rate": fallback_rate,
            }
            job.working_memory = working_memory
        version = latest_script.version + 1
        opt_history = dict(latest_script.optimization_history or {})
        patch_summary = result.payload.get("patch_summary", "")
        if patch_summary:
            working_memory["_pending_patch_summary"] = patch_summary
            job.working_memory = working_memory

        if fallback_rate > 0.2:
            logger.warning(
                f"High optimizer fallback rate ({fallback_rate:.2%}) for Job {job.id}"
            )
        elif fallback_rate < 0.05:
            logger.info(
                f"Low optimizer fallback rate ({fallback_rate:.2%}) for Job {job.id}"
            )

        await save_script(
            db,
            job.id,
            result.payload["script_content"],
            version,
            optimization_history=opt_history,
        )
        await update_job_status(db, job.id, JobStatusEnum.FACT_CHECKING_SCRIPT)
    elif result.escalated:
        error_msg = result.error_log[0] if result.error_log else "Unknown escalation"
        logger.warning(f"Optimizer ESCALATE for Job {job.id}: {error_msg}")
        await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)
    else:
        error_msg = result.error_log[-1] if result.error_log else "Unknown error"
        raise Exception(f"Optimizer failed: {error_msg}")


async def _update_optimization_ledger(
    latest_script_obj,
    claims_data: list,
    pending_patch_summary: str | None = None,
) -> None:
    if not claims_data or not latest_script_obj:
        return
    try:
        embedder = get_embeddings()
    except Exception:
        logger.exception("Failed to initialise embedder for optimization ledger")
        return
    raw_ledger = latest_script_obj.optimization_history or {}
    patches_applied = [pending_patch_summary] if pending_patch_summary else None
    if raw_ledger.get("active_claims"):
        prev_active = raw_ledger["active_claims"]
        try:
            mapping = await map_claims(prev_active, claims_data, embedder)
        except Exception:
            logger.exception("Failed to map claims for optimization ledger")
            return
        delta = compute_verdict_delta(prev_active, claims_data, mapping)
        updated = update_ledger(
            raw_ledger,
            claims_data,
            mapping,
            delta,
            patches_applied=patches_applied,
        )
    else:
        updated = init_ledger(claims_data)
    latest_script_obj.optimization_history = updated


async def _update_epistemic_ledger(job, claims_data: list) -> None:
    if not claims_data:
        return
    weak_passes = [
        {
            "claim_text": c["claim_text"],
            "verdict": c["verdict"],
            "confidence": c.get("confidence", 0.0),
            "weakness_reason": c.get("evidence_text", ""),
        }
        for c in claims_data
        if c.get("verdict") in ("UNCERTAIN", "CONTESTED")
        or (c.get("verdict") == "SUPPORTED" and c.get("confidence", 1.0) < 0.7)
    ]
    if weak_passes:
        working_memory = dict(job.working_memory or {})
        working_memory["epistemic_ledger"] = {"weak_passes": weak_passes}
        job.working_memory = working_memory


async def _transition_fact_checking_script(db: AsyncSession, job) -> None:
    red_team = RedTeamAgent(
        model_name=settings.evaluator_model,
        temperature=settings.evaluator_temperature,
    )
    vector_store = _get_vector_store(db)
    registry = ToolRegistry()
    registry.register(make_semantic_search_tool(vector_store), replace=True)
    harness = AgentHarness(agent=red_team)

    latest_script_obj = await get_latest_script(db, job.id)
    latest_script = latest_script_obj.content if latest_script_obj else ""

    story_directives = job.story_directives or {}
    try:
        strictness = GuardrailStrictness(
            story_directives.get("guardrail_strictness", "High")
        )
    except ValueError:
        logger.warning(
            f"Invalid guardrail_strictness '{story_directives.get('guardrail_strictness')}' "
            f"for Job {job.id} — falling back to High"
        )
        strictness = GuardrailStrictness.High
    guardrail_cfg = get_guardrail_config(
        strictness=strictness,
        uncertain_pass_through=story_directives.get("uncertain_pass_through", False),
    )

    agent_context = {
        "job_id": job.id,
        "script_content": latest_script,
        "guardrail_config": guardrail_cfg,
    }
    working_memory = job.working_memory or {}
    if "copywriter_rationale" in working_memory:
        agent_context["copywriter_rationale"] = working_memory["copywriter_rationale"]
    if "optimizer_phase" in working_memory:
        agent_context["optimizer_phase"] = working_memory["optimizer_phase"]
    result = await harness.run_with_harness(agent_context)

    if result.escalated:
        error_msg = result.error_log[0] if result.error_log else "Unknown escalation"
        logger.error(f"Red Team escalated Job {job.id}: {error_msg}")
        await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)
    elif result.success:
        claims_data = result.payload.get("claims", [])
        await _resolve_evidence_refs(db, vector_store, job.id, claims_data)

        if not claims_data:
            logger.info(
                f"Red Team found 0 claims for Job {job.id}. "
                f"Script contains no verifiable factual assertions. "
                f"Proceeding without fact-check audit."
            )

        if claims_data and latest_script_obj:
            await save_fact_check_claims(db, latest_script_obj.id, claims_data)
            pending_patch_summary = (job.working_memory or {}).get(
                "_pending_patch_summary"
            )
            await _update_optimization_ledger(
                latest_script_obj,
                claims_data,
                pending_patch_summary=pending_patch_summary,
            )
            await _update_epistemic_ledger(job, claims_data)

        if latest_script_obj:
            if guardrail_cfg.requires_human_review:
                latest_script_obj.is_approved = False
                await db.commit()
                logger.info(
                    f"Red Team Approved for Job {job.id}. "
                    f"{len(claims_data)} claims persisted. "
                    f"High profile: awaiting human review."
                )
                await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)
            else:
                latest_script_obj.is_approved = True
                await db.commit()
                logger.info(
                    f"Red Team Approved for Job {job.id}. "
                    f"{len(claims_data)} claims persisted. Proceeding to {next_status_after_fact_check(job.format_type).value}."
                )
                await update_job_status(
                    db, job.id, next_status_after_fact_check(job.format_type)
                )
    else:
        if (
            result.agent_status == AgentActionStatus.REVISION_NEEDED
            and result.payload is not None
        ):
            claims_data = result.payload.get("claims", [])
            await _resolve_evidence_refs(db, vector_store, job.id, claims_data)

            if claims_data and latest_script_obj:
                await save_fact_check_claims(db, latest_script_obj.id, claims_data)
                pending_patch_summary = (job.working_memory or {}).get(
                    "_pending_patch_summary"
                )
                await _update_optimization_ledger(
                    latest_script_obj,
                    claims_data,
                    pending_patch_summary=pending_patch_summary,
                )
                await _update_epistemic_ledger(job, claims_data)
                await db.commit()

            failed_claims = [
                c
                for c in claims_data
                if c.get("verdict") in ("UNSUPPORTED", "CONTESTED")
            ]

            current_revision = latest_script_obj.version if latest_script_obj else 0
            logger.warning(
                f"Red Team Rejected Job {job.id}. Revision {current_revision}/{settings.max_red_team_revisions}"
            )

            if current_revision >= settings.max_red_team_revisions:
                logger.error(f"Max revisions reached for Job {job.id}. Escalating.")
                await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)
            else:
                reasoning = (
                    result.reasoning
                    or (result.error_log[-1] if result.error_log else None)
                    or "Claims require revision"
                )
                await append_script_feedback(
                    db,
                    job.id,
                    feedback=reasoning,
                    structured_claims=failed_claims,
                    overall_reasoning=reasoning,
                    revision_number=current_revision,
                )
                await update_job_status(db, job.id, JobStatusEnum.SCRIPTING)
        else:
            error_msg = (
                result.error_log[-1]
                if result.error_log
                else "Unknown error after retries"
            )
            logger.error(f"Red Team failed Job {job.id}: {error_msg}")
            await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)


async def _transition_asset_generation(db: AsyncSession, job) -> None:
    video_script = await get_latest_format_script(db, job.id, "VIDEO")
    carousel_script = await get_latest_format_script(db, job.id, "CAROUSEL")

    if not video_script and not carousel_script:
        raise Exception(
            f"Cannot proceed to ASSET_GENERATION for Job {job.id}: "
            f"no approved format script with payload found (checked VIDEO, CAROUSEL)."
        )

    any_success = False

    # --- Video asset generation ---
    if video_script and video_script.format_payload:
        studio_context: Dict[str, Any] = {"job_id": job.id}
        fmt = video_script.format_payload
        studio_context["scenes"] = fmt.get("scenes", [])
        studio_context["visual_style"] = fmt.get("visual_style", "")
        studio_context["script_content"] = video_script.content

        studio = AssetStudioAgent(
            model_name=settings.asset_model,
            temperature=settings.asset_temperature,
        )
        result = await studio.run(context=studio_context)

        if result.status == AgentActionStatus.SUCCESS:
            job.final_video_url = result.payload["video_url"]
            any_success = True
        elif result.status == AgentActionStatus.ERROR:
            await log_error(
                db, job.id, result.reasoning, phase="VIDEO_ASSET_GENERATION"
            )

    # --- Carousel image generation ---
    if carousel_script and carousel_script.format_payload:
        carousel_harness = AgentHarness(agent=CarouselImageAgent())
        context: Dict[str, Any] = {
            "format_type": "carousel",
            "job_id": job.id,
            "format_payload": carousel_script.format_payload,
            "platform": job.platform or "instagram",
            "device_id": job.device_id,
        }
        carousel_result = await carousel_harness.run_with_harness(context)

        if carousel_result.success:
            carousel_script.format_payload = merge_image_urls(
                carousel_script.format_payload,
                carousel_result.payload["format_payload"],
            )
            flag_modified(carousel_script, "format_payload")
            any_success = True
        else:
            error_msg = (
                carousel_result.error_log[0]
                if carousel_result.error_log
                else "Unknown error"
            )
            await log_error(
                db,
                job.id,
                error_msg,
                phase="CAROUSEL_IMAGE_GENERATION",
            )

    if any_success:
        await db.commit()
        await update_job_status(db, job.id, JobStatusEnum.COMPLETED)
    else:
        await db.commit()
        await update_job_status(db, job.id, JobStatusEnum.FAILED)


async def _promote_to_global(db: AsyncSession, job) -> None:
    try:
        script = await get_latest_script(db, job.id)
        if not script:
            logger.info(f"No script to promote for Job {job.id}")
            return

        claims = await get_script_claims(db, script.id)
        supported_claims = [c for c in claims if c["verdict"] == "SUPPORTED"]

        if not supported_claims:
            logger.info(f"No SUPPORTED claims to promote for Job {job.id}")
            return

        llm = get_llm(
            model_name=settings.promotion_model,
            temperature=settings.promotion_temperature,
        )
        vs = _get_vector_store(db)
        now_iso = datetime.now(timezone.utc).isoformat()
        total_facts = 0

        for claim in supported_claims:
            prompt = (
                "You are a knowledge compression specialist. Below is a script excerpt "
                "and a verified fact-check claim from a content pipeline run. Extract the "
                "key factual statement — numbers, events, causal relationships, attributions, "
                "and timelines — that is broadly useful as long-term context. Omit "
                "editorial framing, speculative content, and run-specific details. Output "
                "the compressed fact as a single-line text.\n\n"
                f"Script context:\n{script.content[:2000]}\n\n"
                f"Verified claim:\n{claim['claim_text']}"
            )
            response = await llm.ainvoke(prompt)
            compressed = (
                response.content if hasattr(response, "content") else str(response)
            )
            fact = compressed.strip().lstrip("- ").strip()
            if not fact:
                continue

            await vs.ingest_chunks(
                job_id=None,
                chunks=[fact],
                scope="GLOBAL",
                meta={
                    "source_job_id": str(job.id),
                    "source_title": job.title,
                    "source_type": "COMPRESSED_FACT",
                    "claim_verdict": claim["verdict"],
                    "confidence": claim.get("confidence"),
                    "ingested_at": now_iso,
                },
            )
            total_facts += 1

        if total_facts:
            logger.info(
                f"Promoted {total_facts} GLOBAL facts for Job {job.id} "
                f"(from {len(supported_claims)} SUPPORTED claims)"
            )
        else:
            logger.warning(f"GLOBAL promotion produced 0 facts for Job {job.id}")
    except Exception:
        logger.exception(f"GLOBAL promotion failed for Job {job.id} — continuing")


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


def _build_format_content(format_type: str, payload: dict) -> str:
    if format_type == "BLOG":
        title = payload.get("title", "")
        sections = payload.get("sections", [])
        parts = [f"# {title}"] if title else []
        for sec in sections:
            heading = sec.get("heading", "")
            body = sec.get("body", "")
            parts.append(f"## {heading}\n\n{body}" if heading else body)
        return "\n\n".join(parts)

    if format_type == "CAROUSEL":
        thread_title = payload.get("thread_title", "")
        slides = payload.get("slides", [])
        parts = [f"# {thread_title}"] if thread_title else []
        for slide in slides:
            num = slide.get("slide_number", "")
            text = slide.get("text", "")
            parts.append(f"**Slide {num}**\n\n{text}")
        return "\n\n---\n\n".join(parts)

    if format_type == "VIDEO":
        title = payload.get("title", "")
        scenes = payload.get("scenes", [])
        parts = [f"# {title}"] if title else []
        for scene in scenes:
            num = scene.get("scene_number", "")
            narration = scene.get("narration_text", "")
            visual = scene.get("visual_prompt", "")
            audio = scene.get("audio_cue", "")
            parts.append(
                f"### Scene {num}\n\n"
                f"**Narration:** {narration}\n\n"
                f"**Visual:** {visual}\n\n"
                f"**Audio:** {audio}"
            )
        return "\n\n".join(parts)

    return payload.get("title", payload.get("thread_title", ""))


def _next_status_after_formatting(
    resolved_formats: list[FormatTypeEnum],
) -> JobStatusEnum:
    if (
        FormatTypeEnum.VIDEO in resolved_formats
        or FormatTypeEnum.CAROUSEL in resolved_formats
    ):
        return JobStatusEnum.ASSET_GENERATION
    return JobStatusEnum.COMPLETED


async def _transition_formatting(db: AsyncSession, job) -> None:
    latest_script = await get_latest_script(db, job.id)
    if not latest_script:
        raise Exception(f"No script found for Job {job.id} at FORMATTING stage")

    verified_claims = await get_script_claims(db, latest_script.id)

    hedge_index = [
        {"claim_text": c["claim_text"], "verdict": c["verdict"]}
        for c in verified_claims
        if c.get("hedge_required")
    ]
    if hedge_index:
        job.hedge_index = hedge_index
        await db.commit()

    working_memory = job.working_memory or {}
    epistemic_ledger = working_memory.get("epistemic_ledger", {})

    base_context = {
        "script_content": latest_script.content,
        "refined_context": job.refined_context or "",
        "verified_claims": verified_claims,
        "hedge_index": hedge_index,
        "epistemic_ledger": epistemic_ledger,
        "platform": job.platform or "",
        "story_directives": job.story_directives or {},
    }

    db_format_type = (job.format_type or "all").lower()
    db_platform = job.platform if job.platform else "twitter"

    format_type_enum = FormatTypeEnum(db_format_type)
    platform_enum = PlatformEnum(db_platform)
    target_formats = resolve_formats(platform_enum, format_type_enum)
    target_format_names = [f.value.upper() for f in target_formats]

    formatter_specs: list[tuple[str, AgentHarness, dict]] = []

    if "BLOG" in target_format_names:
        blog_ctx = {**base_context, "format_type": "blog"}
        blog_harness = AgentHarness(
            agent=BlogFormatterAgent(
                model_name=settings.formatter_model,
                temperature=settings.formatter_temperature,
            ),
            validator=BlogValidator(),
            max_retries=2,
        )
        formatter_specs.append(("BLOG", blog_harness, blog_ctx))

    if "CAROUSEL" in target_format_names:
        carousel_ctx = {
            **base_context,
            "format_type": "carousel",
            "platform": job.platform or "default",
        }
        carousel_harness = AgentHarness(
            agent=CarouselFormatterAgent(
                model_name=settings.formatter_model,
                temperature=settings.formatter_temperature,
            ),
            validator=CarouselValidator(platform=job.platform or "default"),
            max_retries=2,
        )
        formatter_specs.append(("CAROUSEL", carousel_harness, carousel_ctx))

    if "VIDEO" in target_format_names:
        video_ctx = {**base_context, "format_type": "video"}
        video_harness = AgentHarness(
            agent=VideoFormatterAgent(
                model_name=settings.formatter_model,
                temperature=settings.formatter_temperature,
            ),
            validator=VideoValidator(),
            max_retries=2,
        )
        formatter_specs.append(("VIDEO", video_harness, video_ctx))

    if not formatter_specs:
        logger.warning(
            f"No formatters configured for format_type='{db_format_type}' on Job {job.id}"
        )
        await update_job_status(
            db, job.id, _next_status_after_formatting(target_formats)
        )
        return

    logger.info(
        f"Running {len(formatter_specs)} formatter(s) in parallel for Job {job.id} "
        f"(format_type={db_format_type})"
    )

    harness_results = await asyncio.gather(
        *[h.run_with_harness(ctx) for _, h, ctx in formatter_specs],
        return_exceptions=True,
    )

    next_version = latest_script.version + 1
    any_success = False
    any_escalated = False

    for i, raw_result in enumerate(harness_results):
        fmt_name, harness, ctx = formatter_specs[i]

        if isinstance(raw_result, Exception):
            logger.error(
                f"Formatter {fmt_name} threw exception for Job {job.id}: {raw_result}"
            )
            await save_format_script(
                db,
                job_id=job.id,
                content=f"[{fmt_name} FORMATTING FAILED: {raw_result}]",
                version=next_version,
                format_type=fmt_name,
                format_payload=None,
                is_approved=False,
            )
            next_version += 1
            continue

        if raw_result.success:
            content = _build_format_content(fmt_name, raw_result.payload)
            await save_format_script(
                db,
                job_id=job.id,
                content=content,
                version=next_version,
                format_type=fmt_name,
                format_payload=raw_result.payload,
                is_approved=True,
            )
            any_success = True
            logger.info(
                f"Formatter {fmt_name} succeeded for Job {job.id} "
                f"(attempts={raw_result.attempts})"
            )
        else:
            if getattr(raw_result, "escalated", False):
                any_escalated = True
                logger.warning(
                    f"Formatter {fmt_name} ESCALATED for Job {job.id}: "
                    f"{raw_result.error_log}"
                )
            else:
                logger.error(
                    f"Formatter {fmt_name} failed for Job {job.id} after "
                    f"{raw_result.attempts} attempts: {raw_result.error_log}"
                )
            await save_format_script(
                db,
                job_id=job.id,
                content=f"[{fmt_name} FORMATTING FAILED]",
                version=next_version,
                format_type=fmt_name,
                format_payload=None,
                is_approved=False,
            )

        next_version += 1

    if any_escalated:
        logger.warning(
            f"Formatter escalation for Job {job.id}; routing to HUMAN_REVIEW_NEEDED"
        )
        await update_job_status(db, job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED)
        return

    if not any_success:
        raise Exception(
            f"All formatters failed for Job {job.id}. Failed script rows recorded."
        )

    await update_job_status(db, job.id, _next_status_after_formatting(target_formats))
