import asyncio
import logging
from typing import Any, ClassVar, Dict, List, Optional, Set, Type

from app.core.config import settings
from app.workers.agents import (
    AgentActionStatus,
    AgentResult,
    ServiceAgent,
)
from pydantic import BaseModel

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


class CarouselImageAgent(ServiceAgent):
    _required_di_tools: ClassVar[List[str]] = ["generate_image", "upload_image"]
    _permissions: ClassVar[Set[str]] = {"CarouselImageAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
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

        for tool_name in self._required_di_tools:
            if tool_name not in self.di_tools:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Required DI tool '{tool_name}' not injected into CarouselImageAgent",
                    confidence_score=0.0,
                )

        gen_tool = self.di_tools["generate_image"]
        upload_tool = self.di_tools["upload_image"]

        failures: list[dict] = []
        for i, slide in enumerate(slides):
            visual_description = slide.get("visual_description", "")
            if not visual_description:
                slide["image_url"] = None
                continue

            if i > 0:
                await asyncio.sleep(settings.image_gen_slide_delay)

            result = await gen_tool.callable(visual_description, platform)

            if result["success"] and result["image_bytes"]:
                filename = f"slide_{slide['slide_number']:02d}.png"
                folder = f"{device_id or '__anonymous__'}/{job_id or 'standalone'}"
                url = await upload_tool.callable(
                    result["image_bytes"], filename, folder=folder
                )
                slide["image_url"] = url
            else:
                slide["image_url"] = None
                failures.append(
                    {
                        "slide_number": slide.get("slide_number"),
                        "reason": result.get("failure_reason") or "Unknown error",
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
