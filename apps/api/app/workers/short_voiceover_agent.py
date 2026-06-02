import json
import logging
from typing import Any, ClassVar, Dict, List, Optional, Set, Type

from app.workers.agents import (
    AgentActionStatus,
    AgentResult,
    ServiceAgent,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ShortVoiceoverAgent(ServiceAgent):
    _required_di_tools: ClassVar[List[str]] = [
        "generate_voiceover",
        "get_alignment",
        "upload_voiceover",
    ]
    _permissions: ClassVar[Set[str]] = {"ShortVoiceoverAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        format_payload = context.get("format_payload", {})
        job_id = context.get("job_id")
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

        voice_id = format_payload.get("voice_id")
        if not voice_id:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="voice_id is required in format_payload",
                confidence_score=0.0,
            )

        for tool_name in self._required_di_tools:
            if tool_name not in self.di_tools:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=(
                        f"Required DI tool '{tool_name}' not injected "
                        f"into ShortVoiceoverAgent"
                    ),
                    confidence_score=0.0,
                )

        gen_voiceover_tool = self.di_tools["generate_voiceover"]
        upload_voiceover_tool = self.di_tools["upload_voiceover"]

        # Concatenate narration_text values with ". " separator
        narration_parts: List[str] = []
        for scene in scenes:
            narration_text = scene.get("narration_text", "")
            if narration_text:
                narration_parts.append(narration_text.strip())

        if not narration_parts:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No narration_text found in any scene",
                confidence_score=0.0,
            )

        full_text = ". ".join(narration_parts)

        try:
            tts_result = await gen_voiceover_tool.callable(
                text=full_text,
                voice_id=voice_id,
            )
        except Exception as exc:
            logger.warning("Voiceover generation failed: %s", exc)
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning=f"Voiceover generation failed: {exc}",
                confidence_score=0.0,
            )

        audio_bytes = tts_result.get("audio_bytes")
        vocal_alignment_data = tts_result.get("vocal_alignment_data", [])
        duration_seconds = tts_result.get("duration_seconds", 0.0)

        if not audio_bytes:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="generate_voiceover returned no audio_bytes",
                confidence_score=0.0,
            )

        folder = f"{device_id or '__anonymous__'}/{job_id or 'standalone'}"
        voiceover_filename = f"voiceover_{job_id or 'standalone'}.mp3"

        try:
            voiceover_url = await upload_voiceover_tool.callable(
                audio_bytes,
                voiceover_filename,
                folder=folder,
                content_type="audio/mpeg",
            )
        except Exception as exc:
            logger.warning("Voiceover upload failed: %s", exc)
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning=f"Voiceover upload failed: {exc}",
                confidence_score=0.0,
            )

        # Upload vocal alignment data as JSON
        alignment_json = json.dumps(vocal_alignment_data).encode("utf-8")
        alignment_filename = f"vocal_alignment_{job_id or 'standalone'}.json"

        try:
            alignment_url = await upload_voiceover_tool.callable(
                alignment_json,
                alignment_filename,
                folder=folder,
                content_type="application/json",
            )
        except Exception as exc:
            logger.warning("Vocal alignment upload failed: %s", exc)
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning=f"Vocal alignment upload failed: {exc}",
                confidence_score=0.0,
            )

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={
                "voiceover_url": voiceover_url,
                "vocal_alignment_url": alignment_url,
                "duration_seconds": duration_seconds,
            },
            reasoning=(
                f"Generated voiceover for {len(scenes)} scenes "
                f"({duration_seconds:.1f}s)"
            ),
            confidence_score=1.0,
            metadata={
                "total_scenes": len(scenes),
                "duration_seconds": duration_seconds,
                "text_length": len(full_text),
            },
        )
