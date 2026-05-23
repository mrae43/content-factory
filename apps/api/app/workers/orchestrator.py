import asyncio
import logging
import traceback
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

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
from app.services.vector_store import ContentFactoryVectorStore
from app.services.web_search import TavilySearchService
from app.services.chunking import process_extraction_job
from app.services.format_validator import (
    BlogValidator,
    CarouselValidator,
    VideoValidator,
)
from app.workers.tasks import cleanup_local_research_chunks
from app.workers.agents import (
    ResearchAgent,
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
from app.workers.harness import FormatterHarness
from app.schemas.shorts import (
    JobStatusEnum,
    AssembledContext,
    next_status_after_fact_check,
    resolve_formats,
    FormatTypeEnum,
    PlatformEnum,
)
from app.services.context_builder import build as _build_context_from_service
from app.core.config import settings
from app.core.guardrails import get_guardrail_config, GuardrailStrictness

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
            job_id=job.id,
            chunks=raw_chunks,
            scope="RAW-CONTEXT",
            meta={"source_type": "USER_PROVIDED"},
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
                    "source_type": "WEB_SEARCH",
                    "query": job.topic,
                    "urls": web_urls,
                    "search_depth": "basic",
                },
            )

    await update_job_status(db, job.id, JobStatusEnum.RETRIEVAL)


async def _transition_retrieval(db: AsyncSession, job) -> None:
    vector_store = ContentFactoryVectorStore(db)

    researcher = ResearchAgent(
        model_name=settings.research_model,
        temperature=settings.research_temperature,
    )
    agent_context = {
        "job_id": job.id,
        "topic": job.topic,
        "vector_store": vector_store,
    }
    result = await researcher.run(context=agent_context)

    if result.status == AgentActionStatus.SUCCESS:
        refined_context = result.payload.get("refined_context", "")
        if not refined_context:
            raise Exception(
                "Research agent succeeded but produced no refined_context. "
                "Cannot proceed to scripting without a research summary."
            )
        job.refined_context = refined_context

        confidence = result.confidence_score
        if confidence is not None:
            job.research_confidence = confidence

        citation_index = result.payload.get("citation_index", [])
        if citation_index:
            job.citation_index = citation_index

        await db.commit()
    else:
        raise Exception(f"Research failed: {result.reasoning}")

    assembled = await _build_script_context(db, job)
    job.assembled_context = assembled.model_dump()
    await db.commit()
    await update_job_status(db, job.id, JobStatusEnum.SCRIPTING)


async def _build_script_context(db: AsyncSession, job) -> AssembledContext:
    try:
        vector_store = ContentFactoryVectorStore(db)
        pre_context = job.pre_context or {}
        story_directives = {
            "target_audience": pre_context.get("target_audience", "General"),
            "tone": pre_context.get("tone", ""),
            "angle": pre_context.get("angle", ""),
        }
        return await _build_context_from_service(
            topic=job.topic,
            story_directives=story_directives,
            refined_context=job.refined_context or "",
            vector_store=vector_store,
            job_id=job.id,
            top_k=settings.context_builder_top_k,
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
            failed_claims = last_feedback.get("failed_claims", [])
            await _run_optimizer(
                db,
                job,
                latest_script,
                failed_claims,
                evidence_sections=evidence_sections,
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
    pre_context = job.pre_context or {}
    agent_context = {
        "job_id": job.id,
        "topic": job.topic,
        "refined_context": job.refined_context or "",
        "evidence_sections": evidence_sections,
        "story_directives": {
            "target_audience": pre_context.get("target_audience", "General"),
            "tone": pre_context.get("tone", ""),
            "angle": pre_context.get("angle", ""),
        },
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
    db: AsyncSession,
    job,
    latest_script,
    failed_claims: list,
    evidence_sections: str = "",
) -> None:
    optimizer = ScriptOptimizerAgent(
        model_name=settings.optimizer_model,
        temperature=settings.optimizer_temperature,
    )
    pre_context = job.pre_context or {}
    agent_context = {
        "job_id": job.id,
        "script_content": latest_script.content,
        "failed_claims": failed_claims,
        "refined_context": job.refined_context or "",
        "evidence_sections": evidence_sections,
        "story_directives": {
            "target_audience": pre_context.get("target_audience", "General"),
            "tone": pre_context.get("tone", ""),
            "angle": pre_context.get("angle", ""),
        },
    }
    result = await optimizer.run(context=agent_context)

    if result.status == AgentActionStatus.SUCCESS:
        version = latest_script.version + 1
        await save_script(db, job.id, result.payload["script_content"], version)
        await update_job_status(db, job.id, JobStatusEnum.FACT_CHECKING_SCRIPT)
    else:
        raise Exception(f"Optimizer failed: {result.reasoning}")


async def _transition_fact_checking_script(db: AsyncSession, job) -> None:
    red_team = RedTeamAgent(
        model_name=settings.evaluator_model,
        temperature=settings.evaluator_temperature,
    )
    vector_store = ContentFactoryVectorStore(db)

    latest_script_obj = await get_latest_script(db, job.id)
    latest_script = latest_script_obj.content if latest_script_obj else ""

    pre_context = job.pre_context or {}
    try:
        strictness = GuardrailStrictness(
            pre_context.get("guardrail_strictness", "High")
        )
    except ValueError:
        logger.warning(
            f"Invalid guardrail_strictness '{pre_context.get('guardrail_strictness')}' "
            f"for Job {job.id} — falling back to High"
        )
        strictness = GuardrailStrictness.High
    guardrail_cfg = get_guardrail_config(
        strictness=strictness,
        uncertain_pass_through=pre_context.get("uncertain_pass_through", False),
    )

    agent_context = {
        "job_id": job.id,
        "script_content": latest_script,
        "vector_store": vector_store,
        "guardrail_config": guardrail_cfg,
    }
    result = await red_team.run(context=agent_context)

    if result.status == AgentActionStatus.SUCCESS:
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
        agent = CarouselImageAgent()
        context: Dict[str, Any] = {
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
            any_success = True
        else:
            await log_error(
                db,
                job.id,
                carousel_result.reasoning,
                phase="CAROUSEL_IMAGE_GENERATION",
            )

    if any_success:
        await db.commit()
        await update_job_status(db, job.id, JobStatusEnum.COMPLETED)
    else:
        await db.commit()
        await update_job_status(db, job.id, JobStatusEnum.FAILED)


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

    base_context = {
        "script_content": latest_script.content,
        "refined_context": job.refined_context or "",
        "verified_claims": verified_claims,
        "hedge_index": hedge_index,
        "platform": job.platform or "",
    }

    db_format_type = (job.format_type or "all").lower()
    db_platform = job.platform if job.platform else "twitter"

    format_type_enum = FormatTypeEnum(db_format_type)
    platform_enum = PlatformEnum(db_platform)
    target_formats = resolve_formats(platform_enum, format_type_enum)
    target_format_names = [f.value.upper() for f in target_formats]

    formatter_specs: list[tuple[str, FormatterHarness, dict]] = []

    if "BLOG" in target_format_names:
        blog_ctx = {**base_context, "format_type": "blog"}
        blog_harness = FormatterHarness(
            formatter=BlogFormatterAgent(
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
        carousel_harness = FormatterHarness(
            formatter=CarouselFormatterAgent(
                model_name=settings.formatter_model,
                temperature=settings.formatter_temperature,
            ),
            validator=CarouselValidator(platform=job.platform or "default"),
            max_retries=2,
        )
        formatter_specs.append(("CAROUSEL", carousel_harness, carousel_ctx))

    if "VIDEO" in target_format_names:
        video_ctx = {**base_context, "format_type": "video"}
        video_harness = FormatterHarness(
            formatter=VideoFormatterAgent(
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

    if not any_success:
        raise Exception(
            f"All formatters failed for Job {job.id}. Failed script rows recorded."
        )

    await update_job_status(db, job.id, _next_status_after_formatting(target_formats))
