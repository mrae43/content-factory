import asyncio
import logging
import time
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


class ShortVisualAssetAgent(ServiceAgent):
    _required_di_tools: ClassVar[List[str]] = [
        "generate_video",
        "poll_video",
        "generate_image",
        "upload_video",
        "upload_image",
    ]
    _permissions: ClassVar[Set[str]] = {"ShortVisualAssetAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        format_payload = context.get("format_payload", {})
        job_id = context.get("job_id")
        platform = context.get("platform", "tiktok")
        device_id = context.get("device_id")

        if not isinstance(format_payload, dict):
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="format_payload must be a dict",
                confidence_score=0.0,
            )

        scenes = format_payload.get("scenes", [])
        if not scenes:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No scenes found in format_payload",
                confidence_score=0.0,
            )

        for tool_name in self._required_di_tools:
            if tool_name not in self.di_tools:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=(
                        f"Required DI tool '{tool_name}' not injected "
                        f"into ShortVisualAssetAgent"
                    ),
                    confidence_score=0.0,
                )

        gen_video_tool = self.di_tools["generate_video"]
        poll_video_tool = self.di_tools["poll_video"]
        gen_image_tool = self.di_tools["generate_image"]
        upload_video_tool = self.di_tools["upload_video"]
        upload_image_tool = self.di_tools["upload_image"]

        folder = f"{device_id or '__anonymous__'}/{job_id or 'standalone'}"

        failures: list[dict] = []
        scene_urls: list[dict] = []

        # Partition scenes
        video_scenes: list[dict] = []
        kb_scenes: list[dict] = []
        for i, scene in enumerate(scenes):
            scene["scene_number"] = scene.get("scene_number", i + 1)
            if scene.get("asset_type") == "video_clip":
                video_scenes.append(scene)
            else:
                kb_scenes.append(scene)

        # Run original Ken Burns scenes in parallel
        kb_tasks = [
            self._generate_ken_burns_scene(
                scene=scene,
                platform=platform,
                gen_image_tool=gen_image_tool,
                upload_image_tool=upload_image_tool,
                folder=folder,
            )
            for scene in kb_scenes
        ]
        kb_results = await asyncio.gather(*kb_tasks, return_exceptions=True)
        for scene, result in zip(kb_scenes, kb_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Scene %d: Ken Burns generation failed: %s",
                    scene["scene_number"],
                    result,
                )
                failures.append(
                    {
                        "scene_number": scene["scene_number"],
                        "reason": str(result),
                    }
                )
            else:
                scene["image_url"] = result["url"]
                scene_urls.append(result)

        # Run video_clip scenes sequentially
        kb_fallback_scenes: list[dict] = []
        for scene in video_scenes:
            scene_number = scene["scene_number"]
            visual_prompt = scene.get("visual_prompt", "")
            if not visual_prompt:
                failures.append(
                    {
                        "scene_number": scene_number,
                        "reason": "Empty visual_prompt",
                    }
                )
                continue

            raw_duration = scene.get("target_duration_seconds", 5)
            url = await self._try_generate_video_clip(
                visual_prompt=visual_prompt,
                target_duration=raw_duration,
                gen_tool=gen_video_tool,
                poll_tool=poll_video_tool,
                upload_tool=upload_video_tool,
                folder=folder,
                scene_number=scene_number,
                max_retries=2,
            )
            if url:
                scene["video_url"] = url
                scene_urls.append(
                    {
                        "scene_number": scene_number,
                        "url": url,
                        "asset_type": "video_clip",
                    }
                )
                continue

            # Fallback to Ken Burns still
            logger.warning(
                "Scene %d video generation failed, falling back to Ken Burns",
                scene_number,
            )
            scene["asset_type"] = "ken_burns"
            scene["kb_motion"] = "zoom_in"
            kb_fallback_scenes.append(scene)

        # Run any fallback Ken Burns scenes in parallel
        if kb_fallback_scenes:
            fb_tasks = [
                self._generate_ken_burns_scene(
                    scene=scene,
                    platform=platform,
                    gen_image_tool=gen_image_tool,
                    upload_image_tool=upload_image_tool,
                    folder=folder,
                )
                for scene in kb_fallback_scenes
            ]
            fb_results = await asyncio.gather(*fb_tasks, return_exceptions=True)
            for scene, result in zip(kb_fallback_scenes, fb_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Scene %d: fallback Ken Burns generation failed: %s",
                        scene["scene_number"],
                        result,
                    )
                    failures.append(
                        {
                            "scene_number": scene["scene_number"],
                            "reason": str(result),
                        }
                    )
                else:
                    scene["image_url"] = result["url"]
                    scene_urls.append(result)

        # Preserve original scene order in outputs
        scene_urls.sort(key=lambda u: u["scene_number"])
        failures.sort(key=lambda f: f["scene_number"])

        success_count = len(scenes) - len(failures)
        status = (
            AgentActionStatus.ERROR if success_count == 0 else AgentActionStatus.SUCCESS
        )

        return AgentResult(
            status=status,
            payload={
                "scene_urls": scene_urls,
                "updated_format_payload": format_payload,
            },
            reasoning=(
                f"Generated visual assets for {success_count}/{len(scenes)} scenes"
                if status == AgentActionStatus.SUCCESS
                else f"All {len(scenes)} scenes failed"
            ),
            confidence_score=1.0 if success_count == len(scenes) else 0.0,
            metadata={
                "total_scenes": len(scenes),
                "successful_scenes": success_count,
                "failed_scenes": len(failures),
                "failures": failures,
            },
        )

    async def _generate_ken_burns_scene(
        self,
        scene: dict,
        platform: str,
        gen_image_tool: Any,
        upload_image_tool: Any,
        folder: str,
    ) -> dict:
        """Generate a single Ken Burns scene. Returns scene_url dict."""
        scene_number = scene["scene_number"]
        visual_prompt = scene.get("visual_prompt", "")
        if not visual_prompt:
            raise RuntimeError("Empty visual_prompt")
        img_result = await gen_image_tool.callable(visual_prompt, platform)
        if img_result["success"] and img_result["image_bytes"]:
            filename = f"scene_{scene_number:02d}.png"
            url = await upload_image_tool.callable(
                img_result["image_bytes"], filename, folder=folder
            )
            scene["image_url"] = url
            return {
                "scene_number": scene_number,
                "url": url,
                "asset_type": "ken_burns",
            }
        failure_reason = img_result.get("failure_reason", "Unknown error")
        raise RuntimeError(failure_reason)

    async def _try_generate_video_clip(
        self,
        visual_prompt: str,
        target_duration: float,
        gen_tool: Any,
        poll_tool: Any,
        upload_tool: Any,
        folder: str,
        scene_number: int,
        max_retries: int = 2,
    ) -> Optional[str]:
        """Attempt video generation with retries. Returns S3 URL or None."""
        for attempt in range(1, max_retries + 1):
            try:
                # Step 1: Submit video generation job
                gen_result = await gen_tool.callable(
                    prompt=visual_prompt,
                    model=settings.video_gen_model,
                    duration=int(target_duration),
                )
                video_job_id = gen_result.get("job_id")
                if not video_job_id:
                    logger.warning(
                        "Scene %d: No job_id on attempt %d",
                        scene_number,
                        attempt,
                    )
                    continue

                # Step 2: Poll for completion
                download_url = await self._poll_video_until_done(
                    poll_tool, video_job_id, scene_number
                )
                if not download_url:
                    logger.warning(
                        "Scene %d: video generation attempt %d failed (no download URL)",
                        scene_number,
                        attempt,
                    )
                    continue

                # Step 3: Download video bytes
                video_bytes = await self._download_video_bytes(
                    download_url, scene_number
                )
                if not video_bytes:
                    continue

                # Step 4: Upload to storage
                filename = f"scene_{scene_number:02d}.mp4"
                url = await upload_tool.callable(video_bytes, filename, folder=folder)
                return url

            except Exception as exc:
                logger.warning(
                    "Scene %d video generation attempt %d failed: %s",
                    scene_number,
                    attempt,
                    exc,
                )

        return None

    async def _poll_video_until_done(
        self,
        poll_tool: Any,
        video_job_id: str,
        scene_number: int,
        total_timeout: int = 450,
    ) -> Optional[str]:
        """Poll with exponential backoff until completion or deadline.

        Returns download URL or None.
        Backoff: base_interval * (backoff_factor ** attempt), capped at max_interval.
        """
        base_interval = settings.video_gen_poll_interval_seconds
        max_interval = 60
        backoff_factor = 1.5
        max_retries = settings.video_gen_max_poll_retries
        deadline = time.monotonic() + total_timeout

        for attempt in range(max_retries):
            if time.monotonic() >= deadline:
                logger.warning(
                    "Scene %d: video generation timed out after %ds",
                    scene_number,
                    total_timeout,
                )
                return None

            try:
                poll_result = await poll_tool.callable(job_id=video_job_id)
            except Exception as exc:
                logger.warning(
                    "Scene %d poll attempt %d failed: %s",
                    scene_number,
                    attempt + 1,
                    exc,
                )
                continue

            status = poll_result.get("status", "")
            if status == "completed":
                download_url = poll_result.get("download_url")
                if download_url:
                    return download_url
                logger.warning(
                    "Scene %d: completed but no download_url",
                    scene_number,
                )
                return None
            if status == "failed":
                failure_reason = poll_result.get("failure_reason") or "Unknown failure"
                logger.warning(
                    "Scene %d: video generation failed: %s",
                    scene_number,
                    failure_reason,
                )
                return None

            wait = min(base_interval * (backoff_factor**attempt), max_interval)
            await asyncio.sleep(wait)

        logger.warning(
            "Scene %d: exhausted %d poll attempts",
            scene_number,
            max_retries,
        )
        return None

    async def _download_video_bytes(
        self,
        download_url: str,
        scene_number: int,
    ) -> Optional[bytes]:
        """Download video bytes from the given URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Scene %d: HTTP %d downloading video",
                            scene_number,
                            resp.status,
                        )
                        return None
                    return await resp.read()
        except Exception as exc:
            logger.warning(
                "Scene %d: Failed to download video bytes: %s",
                scene_number,
                exc,
            )
            return None
