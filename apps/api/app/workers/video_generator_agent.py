import asyncio
import logging
from typing import Any, ClassVar, Dict, List, Optional, Set, Type

import aiohttp

from app.core.config import settings
from app.workers.agents import (
    AgentActionStatus,
    AgentResult,
    ServiceAgent,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VideoGeneratorAgent(ServiceAgent):
    _required_di_tools: ClassVar[List[str]] = [
        "generate_video",
        "poll_video",
        "upload_video",
    ]
    _permissions: ClassVar[Set[str]] = {"VideoGeneratorAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        format_payload = context.get("format_payload", {})
        job_id = context.get("job_id")

        if not isinstance(format_payload, dict):
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="format_payload must be a dict",
                confidence_score=0.0,
            )

        unified_visual_prompt = format_payload.get("unified_visual_prompt", "")
        total_duration = format_payload.get("total_duration_seconds", 30)

        if not unified_visual_prompt:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No unified_visual_prompt found in format_payload",
                confidence_score=0.0,
            )

        for tool_name in self._required_di_tools:
            if tool_name not in self.di_tools:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Required DI tool '{tool_name}' not injected into VideoGeneratorAgent",
                    confidence_score=0.0,
                )

        gen_tool = self.di_tools["generate_video"]
        poll_tool = self.di_tools["poll_video"]
        upload_tool = self.di_tools["upload_video"]

        # Step 1: Submit video generation job
        try:
            gen_result = await gen_tool.callable(
                prompt=unified_visual_prompt,
                model=settings.video_gen_model,
                duration=int(total_duration),
            )
        except Exception as exc:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning=f"Video generation submission failed: {exc}",
                confidence_score=0.0,
            )

        video_job_id = gen_result.get("job_id")
        if not video_job_id:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No job_id returned from video generation tool",
                confidence_score=0.0,
            )

        # Step 2: Poll for completion
        model_used = settings.video_gen_model or "default"
        download_url: Optional[str] = None
        poll_interval = settings.video_gen_poll_interval_seconds
        max_retries = settings.video_gen_max_poll_retries

        for attempt in range(max_retries):
            await asyncio.sleep(poll_interval)

            try:
                poll_result = await poll_tool.callable(job_id=video_job_id)
            except Exception as exc:
                logger.warning(
                    "Poll attempt %d/%d failed for job %s: %s",
                    attempt + 1,
                    max_retries,
                    video_job_id,
                    exc,
                )
                continue

            status = poll_result.get("status", "")
            if status == "completed":
                download_url = poll_result.get("download_url")
                if download_url:
                    break
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning="Video job completed but no download_url returned",
                    confidence_score=0.0,
                )
            if status == "failed":
                failure_reason = poll_result.get("failure_reason") or "Unknown failure"
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Video generation failed: {failure_reason}",
                    confidence_score=0.0,
                )

        if not download_url:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning=f"Video generation timed out after {max_retries} polls",
                confidence_score=0.0,
            )

        # Step 3: Download video bytes
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as resp:
                    if resp.status != 200:
                        return AgentResult(
                            status=AgentActionStatus.ERROR,
                            payload={},
                            reasoning=f"Failed to download video: HTTP {resp.status}",
                            confidence_score=0.0,
                        )
                    video_bytes = await resp.read()
        except Exception as exc:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning=f"Failed to download video bytes: {exc}",
                confidence_score=0.0,
            )

        # Step 4: Upload to storage
        filename = f"video_{job_id}.mp4" if job_id else "generated_video.mp4"
        folder = f"{context.get('device_id', '__anonymous__')}/{job_id or 'standalone'}"

        try:
            s3_url = await upload_tool.callable(video_bytes, filename, folder=folder)
        except Exception as exc:
            logger.warning("Video upload failed after successful generation: %s", exc)
            return AgentResult(
                status=AgentActionStatus.SUCCESS,
                payload={
                    "video_url": download_url,
                    "duration": total_duration,
                    "model_used": model_used,
                },
                reasoning=f"Video generated but upload failed: {exc}. Returning download URL.",
                confidence_score=0.7,
            )

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={
                "video_url": s3_url,
                "duration": total_duration,
                "model_used": model_used,
            },
            reasoning="Video generated and uploaded successfully",
            confidence_score=1.0,
            metadata={
                "job_id": str(video_job_id),
                "duration_seconds": total_duration,
                "model": model_used,
            },
        )
