import asyncio
import logging
import traceback
from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.discord_models import FormatJob
from app.db.format_crud import (
    log_format_job_error,
    update_format_job_status,
    update_format_job_format_payload,
    update_format_job_video_url,
)
from app.schemas.shorts import (
    FormatJobStatusEnum,
    FormatTypeEnum,
    PlatformEnum,
    resolve_formats,
)
from app.workers.harness import AgentHarness, HarnessResult
from app.workers.formatters import (
    BlogFormatterAgent,
    CarouselFormatterAgent,
    VideoFormatterAgent,
)
from app.workers.short_formatter import ShortFormatterAgent, _resolve_voice_id
from app.workers.video_generator_agent import VideoGeneratorAgent
from app.workers.carousel_image_agent import CarouselImageAgent, merge_image_urls
from app.workers.short_visual_asset_agent import ShortVisualAssetAgent
from app.workers.short_voiceover_agent import ShortVoiceoverAgent
from app.workers.short_composer_agent import ShortComposerAgent
from app.services.format_validator import (
    BlogValidator,
    CarouselValidator,
    ShortValidator,
    VideoValidator,
)
from app.services.short_config import DEFAULT_SUBTITLE_PRESET_MAP
from app.core.config import settings

logger = logging.getLogger("factory.format_orchestrator")


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

    if format_type == "SHORT":
        scenes = payload.get("scenes", [])
        parts = []
        for scene in scenes:
            num = scene.get("scene_number", "")
            narration = scene.get("narration_text", "")
            visual = scene.get("visual_prompt", "")
            asset_type = scene.get("asset_type", "")
            parts.append(
                f"### Scene {num}\n\n"
                f"**Narration:** {narration}\n\n"
                f"**Visual ({asset_type}):** {visual}"
            )
        return "\n\n".join(parts)

    return payload.get("title", payload.get("thread_title", ""))


def _next_status_after_formatting(
    resolved_formats: list[FormatTypeEnum],
) -> FormatJobStatusEnum:
    if (
        FormatTypeEnum.VIDEO in resolved_formats
        or FormatTypeEnum.CAROUSEL in resolved_formats
        or FormatTypeEnum.SHORT in resolved_formats
    ):
        return FormatJobStatusEnum.ASSET_GENERATION
    return FormatJobStatusEnum.COMPLETED


async def _transition_formatting(db: AsyncSession, format_job: FormatJob) -> None:
    claims = format_job.claims or []

    hedge_index = [
        {"claim_text": c.get("claim_text", ""), "verdict": c.get("verdict", "")}
        for c in claims
        if c.get("hedge_required")
    ]
    format_job.hedge_index = hedge_index
    await db.commit()

    await update_format_job_status(db, format_job.id, FormatJobStatusEnum.FORMATTING)

    base_context = {
        "script_content": format_job.script_content,
        "refined_context": format_job.refined_context or "",
        "verified_claims": claims,
        "hedge_index": hedge_index,
        "epistemic_ledger": format_job.epistemic_ledger or {},
        "platform": format_job.platform or "",
        "story_directives": format_job.story_directives or {},
    }

    db_format_type = (format_job.format_type or "all").lower()
    db_platform = format_job.platform or "twitter"

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
            "platform": format_job.platform or "default",
        }
        carousel_harness = AgentHarness(
            agent=CarouselFormatterAgent(
                model_name=settings.formatter_model,
                temperature=settings.formatter_temperature,
            ),
            validator=CarouselValidator(platform=format_job.platform or "default"),
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

    if "SHORT" in target_format_names:
        story_directives = format_job.story_directives or {}
        voice_id = _resolve_voice_id(story_directives, format_job.platform or "tiktok")
        subtitle_preset = DEFAULT_SUBTITLE_PRESET_MAP.get(
            format_job.platform or "tiktok", "CENTER_POP_YELLOW"
        )
        short_ctx = {
            **base_context,
            "format_type": "short",
            "voice_id": voice_id,
            "loopable": story_directives.get("loopable", True),
            "platform": format_job.platform or "tiktok",
            "visual_style_theme": story_directives.get("visual_style_theme"),
            "subtitle_preset": subtitle_preset,
        }
        short_harness = AgentHarness(
            agent=ShortFormatterAgent(
                model_name=settings.formatter_model,
                temperature=settings.formatter_temperature,
            ),
            validator=ShortValidator(platform=format_job.platform or "tiktok"),
            max_retries=2,
        )
        formatter_specs.append(("SHORT", short_harness, short_ctx))

    if not formatter_specs:
        logger.warning(
            f"No formatters configured for format_type='{db_format_type}' "
            f"on FormatJob {format_job.id}"
        )
        await update_format_job_status(
            db, format_job.id, _next_status_after_formatting(target_formats)
        )
        return

    logger.info(
        f"Running {len(formatter_specs)} formatter(s) in parallel "
        f"for FormatJob {format_job.id} (format_type={db_format_type})"
    )

    harness_results = await asyncio.gather(
        *[h.run_with_harness(ctx) for _, h, ctx in formatter_specs],
        return_exceptions=True,
    )

    format_payload = {}
    any_success = False
    any_escalated = False

    for i, raw_result in enumerate(harness_results):
        fmt_name = formatter_specs[i][0]

        if isinstance(raw_result, Exception):
            logger.error(
                f"Formatter {fmt_name} threw exception "
                f"for FormatJob {format_job.id}: {raw_result}"
            )
            format_payload[fmt_name] = {
                "status": "FAILED",
                "error": str(raw_result),
            }
            continue

        if raw_result.success:
            content = _build_format_content(fmt_name, raw_result.payload)
            format_payload[fmt_name] = {
                "status": "SUCCESS",
                "content": content,
                "payload": raw_result.payload,
            }
            any_success = True
            logger.info(
                f"Formatter {fmt_name} succeeded for FormatJob {format_job.id} "
                f"(attempts={raw_result.attempts})"
            )
        else:
            if getattr(raw_result, "escalated", False):
                any_escalated = True
                logger.warning(
                    f"Formatter {fmt_name} ESCALATED "
                    f"for FormatJob {format_job.id}: {raw_result.error_log}"
                )
            else:
                logger.error(
                    f"Formatter {fmt_name} failed "
                    f"for FormatJob {format_job.id} after "
                    f"{raw_result.attempts} attempts: {raw_result.error_log}"
                )
            format_payload[fmt_name] = {
                "status": "FAILED",
                "error": raw_result.error_log[-1]
                if raw_result.error_log
                else "Unknown error",
            }

    if format_payload:
        await update_format_job_format_payload(db, format_job.id, format_payload)

    if any_escalated:
        logger.warning(
            f"Formatter escalation for FormatJob {format_job.id}; "
            f"routing to HUMAN_REVIEW_NEEDED"
        )
        await update_format_job_status(
            db, format_job.id, FormatJobStatusEnum.HUMAN_REVIEW_NEEDED
        )
        return

    if not any_success:
        raise Exception(f"All formatters failed for FormatJob {format_job.id}")

    await update_format_job_status(
        db, format_job.id, _next_status_after_formatting(target_formats)
    )


async def _transition_asset_generation(db: AsyncSession, format_job: FormatJob) -> None:
    if not format_job.format_payload:
        raise Exception(
            f"Cannot proceed to ASSET_GENERATION for FormatJob {format_job.id}: "
            f"format_payload is empty"
        )

    fmt_type = format_job.format_type.lower() if format_job.format_type else ""
    any_success = False

    if fmt_type == "video":
        sub_payload = format_job.format_payload.get("VIDEO", {}).get(
            "payload", format_job.format_payload
        )
        video_harness = AgentHarness(agent=VideoGeneratorAgent())
        context: Dict[str, Any] = {
            "format_type": "video",
            "job_id": format_job.id,
            "format_payload": sub_payload,
            "platform": format_job.platform or "",
        }
        video_result = await video_harness.run_with_harness(context)

        if video_result.success:
            await update_format_job_video_url(
                db, format_job.id, video_result.payload["video_url"]
            )
            any_success = True
        else:
            error_msg = (
                video_result.error_log[0] if video_result.error_log else "Unknown error"
            )
            await log_format_job_error(
                db, format_job.id, error_msg, phase="VIDEO_ASSET_GENERATION"
            )

    elif fmt_type == "carousel":
        sub_payload = format_job.format_payload.get("CAROUSEL", {}).get(
            "payload", format_job.format_payload
        )
        carousel_harness = AgentHarness(agent=CarouselImageAgent())
        context: Dict[str, Any] = {
            "format_type": "carousel",
            "job_id": format_job.id,
            "format_payload": sub_payload,
            "platform": format_job.platform or "instagram",
        }
        carousel_result = await carousel_harness.run_with_harness(context)

        if carousel_result.success:
            updated_payload = merge_image_urls(
                format_job.format_payload,
                carousel_result.payload["format_payload"],
            )
            await update_format_job_format_payload(db, format_job.id, updated_payload)
            any_success = True
        else:
            error_msg = (
                carousel_result.error_log[0]
                if carousel_result.error_log
                else "Unknown error"
            )
            await log_format_job_error(
                db, format_job.id, error_msg, phase="CAROUSEL_IMAGE_GENERATION"
            )

    # Blog jobs complete during formatting and never reach asset generation.
    if any_success:
        await update_format_job_status(db, format_job.id, FormatJobStatusEnum.COMPLETED)
    else:
        await update_format_job_status(db, format_job.id, FormatJobStatusEnum.FAILED)


async def _run_short_visual_asset(
    format_job: FormatJob, sub_payload: dict
) -> HarnessResult:
    context = {
        "format_type": "short",
        "job_id": format_job.id,
        "format_payload": sub_payload,
        "platform": format_job.platform or "tiktok",
        "device_id": None,
    }
    harness = AgentHarness(agent=ShortVisualAssetAgent())
    return await harness.run_with_harness(context)


async def _run_short_voiceover(
    format_job: FormatJob, sub_payload: dict
) -> HarnessResult:
    context = {
        "format_type": "short",
        "job_id": format_job.id,
        "format_payload": sub_payload,
        "platform": format_job.platform or "tiktok",
        "device_id": None,
    }
    harness = AgentHarness(agent=ShortVoiceoverAgent())
    return await harness.run_with_harness(context)


async def _transition_short_asset_generation(
    db: AsyncSession, format_job: FormatJob
) -> None:
    sub_payload = format_job.format_payload.get("SHORT", {}).get(
        "payload", format_job.format_payload
    )

    story_directives = format_job.story_directives or {}
    resolved_voice_id = _resolve_voice_id(story_directives, format_job.platform or "tiktok")
    if resolved_voice_id:
        sub_payload["voice_id"] = resolved_voice_id

    try:
        async with asyncio.TaskGroup() as tg:
            visual_task = tg.create_task(
                _run_short_visual_asset(format_job, sub_payload)
            )
            voice_task = tg.create_task(_run_short_voiceover(format_job, sub_payload))
    except Exception:
        await log_format_job_error(
            db, format_job.id, traceback.format_exc(), "SHORT_ASSET_GENERATION"
        )
        raise

    visual_result = visual_task.result()
    voice_result = voice_task.result()

    # Merge results into format_payload
    merged = dict(sub_payload)
    if visual_result.success:
        merged.update(visual_result.payload.get("updated_format_payload", merged))
    if voice_result.success:
        merged["voiceover_url"] = voice_result.payload["voiceover_url"]
        merged["vocal_alignment_url"] = voice_result.payload["vocal_alignment_url"]

    if not visual_result.success or not voice_result.success:
        raise Exception(
            f"SHORT asset generation failed: "
            f"visual={visual_result.success}, voice={voice_result.success}"
        )

    payload = dict(format_job.format_payload or {})
    payload["SHORT"] = {"status": "SUCCESS", "payload": merged}
    await update_format_job_format_payload(db, format_job.id, payload)
    await update_format_job_status(db, format_job.id, FormatJobStatusEnum.COMPOSITION)


async def _transition_composition(db: AsyncSession, format_job: FormatJob) -> None:
    sub_payload = format_job.format_payload.get("SHORT", {}).get(
        "payload", format_job.format_payload
    )

    context = {
        "format_type": "short",
        "job_id": format_job.id,
        "format_payload": sub_payload,
        "platform": format_job.platform or "tiktok",
        "device_id": None,
        "voiceover_url": sub_payload.get("voiceover_url"),
        "vocal_alignment_url": sub_payload.get("vocal_alignment_url"),
    }

    composer_harness = AgentHarness(agent=ShortComposerAgent())
    result = await composer_harness.run_with_harness(context)

    if result.success:
        await update_format_job_video_url(
            db, format_job.id, result.payload["final_video_url"]
        )
        await update_format_job_status(db, format_job.id, FormatJobStatusEnum.COMPLETED)
    else:
        error_msg = result.error_log[-1] if result.error_log else "Composition failed"
        await log_format_job_error(db, format_job.id, error_msg, "COMPOSITION")
        await update_format_job_status(db, format_job.id, FormatJobStatusEnum.FAILED)


async def execute_format_state_transition(
    db: AsyncSession, format_job: FormatJob
) -> None:
    try:
        if format_job.status == FormatJobStatusEnum.PENDING:
            await _transition_formatting(db, format_job)
        elif format_job.status == FormatJobStatusEnum.ASSET_GENERATION:
            fmt_type = (format_job.format_type or "").lower()
            if fmt_type == "short":
                await _transition_short_asset_generation(db, format_job)
            else:
                await _transition_asset_generation(db, format_job)
        elif format_job.status == FormatJobStatusEnum.COMPOSITION:
            await _transition_composition(db, format_job)
        elif format_job.status in (
            FormatJobStatusEnum.COMPLETED,
            FormatJobStatusEnum.FAILED,
            FormatJobStatusEnum.HUMAN_REVIEW_NEEDED,
        ):
            logger.warning(
                f"FormatJob {format_job.id} already in terminal state "
                f"{format_job.status}"
            )
    except Exception:
        logger.exception(f"Format job {format_job.id} failed")
        await log_format_job_error(
            db, format_job.id, traceback.format_exc(), "orchestrator"
        )
        await update_format_job_status(db, format_job.id, FormatJobStatusEnum.FAILED)
