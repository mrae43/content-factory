import asyncio
import logging
from typing import Any, Dict

from app.core.config import settings
from app.storage.adapter import get_storage
from app.services.image_gen import ImageGenerationService
from app.workers.agents import AgentActionStatus, AgentResult

logger = logging.getLogger(__name__)


def merge_image_urls(existing_payload: dict | None, new_payload: dict) -> dict:
    existing = existing_payload or {}
    for new_slide in new_payload.get("slides", []):
        num = new_slide.get("slide_number")
        for existing_slide in existing.get("slides", []):
            if existing_slide.get("slide_number") == num:
                existing_slide["image_url"] = new_slide.get("image_url")
                break
    return existing


class CarouselImageAgent:
    """
    Generates images for carousel slides using ImageGenerationService.
    Does NOT extend BaseAgent since no LLM call is needed.

    Uses the storage adapter (local/S3) to persist images and returns
    URL paths in slide.image_url. The orchestrator saves the updated
    format_payload to the DB.
    """

    def __init__(
        self,
        image_service: ImageGenerationService | None = None,
    ):
        self.image_service = image_service or ImageGenerationService()
        self.storage = get_storage()

    async def run(self, context: Dict[str, Any]) -> AgentResult:
        format_payload = context.get("format_payload", {})
        job_id = context.get("job_id")
        platform = context.get("platform", "instagram")
        device_id = context.get("device_id")

        if not isinstance(format_payload, dict):
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="format_payload must be a dict",
                confidence_score=0.0,
            )

        slides = format_payload.get("slides", [])
        if not slides:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No slides found in format_payload",
                confidence_score=0.0,
            )

        failures: list[dict] = []
        for i, slide in enumerate(slides):
            visual_description = slide.get("visual_description", "")
            if not visual_description:
                slide["image_url"] = None
                continue

            if i > 0:
                await asyncio.sleep(settings.image_gen_slide_delay)

            result = await self.image_service.generate(visual_description, platform)

            if result.success and result.image_bytes:
                filename = f"slide_{slide['slide_number']:02d}.png"
                folder = f"{device_id or '__anonymous__'}/{job_id or 'standalone'}"
                url = self.storage.upload_image(
                    result.image_bytes, filename, folder=folder
                )
                slide["image_url"] = url
            else:
                slide["image_url"] = None
                failures.append(
                    {
                        "slide_number": slide.get("slide_number"),
                        "reason": result.failure_reason or "Unknown error",
                    }
                )

        success_count = len(slides) - len(failures)
        status = (
            AgentActionStatus.ERROR if success_count == 0 else AgentActionStatus.SUCCESS
        )

        return AgentResult(
            status=status,
            payload={"format_payload": format_payload},
            reasoning=(
                f"Generated images for {success_count}/{len(slides)} slides"
                if status == AgentActionStatus.SUCCESS
                else f"All {len(slides)} slides failed: {failures[0]['reason']}"
            ),
            confidence_score=1.0 if success_count == len(slides) else 0.0,
            metadata={
                "total_slides": len(slides),
                "successful_slides": success_count,
                "failed_slides": len(failures),
                "failures": failures,
            },
        )
