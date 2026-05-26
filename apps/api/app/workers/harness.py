import hashlib
import json
import logging
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from app.services.format_validator import FormatValidator
from app.workers.agents import AgentActionStatus, BaseAgent

logger = logging.getLogger(__name__)


class HarnessResult(BaseModel):
    success: bool
    format_type: str
    payload: Optional[dict] = None
    error_log: List[str] = Field(default_factory=list)
    attempts: int = 0
    escalated: bool = False


class FormatterHarness:
    def __init__(
        self,
        formatter: BaseAgent,
        validator: FormatValidator,
        max_retries: int = 2,
    ):
        self.formatter = formatter
        self.validator = validator
        self.max_retries = max_retries

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def run_with_harness(self, context: Dict) -> HarnessResult:
        error_log: List[str] = []
        seen_payload_hashes: Set[str] = set()
        max_attempts = self.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            result = await self.formatter.run(context=context)

            if result.status == AgentActionStatus.ESCALATE:
                logger.warning(
                    "Formatter ESCALATE for format_type=%s: %s",
                    context.get("format_type", "unknown"),
                    result.reasoning,
                )
                return HarnessResult(
                    success=False,
                    format_type=context.get("format_type", "unknown"),
                    error_log=[f"Agent escalated: {result.reasoning}"],
                    attempts=attempt,
                    escalated=True,
                )

            if result.status != AgentActionStatus.SUCCESS:
                error_log.append(
                    f"Agent failed (attempt {attempt}): {result.reasoning}"
                )
                context["correction_hint"] = f"Agent failed: {result.reasoning}"
                continue

            payload_hash = self._hash_payload(result.payload)
            if payload_hash in seen_payload_hashes:
                error_log.append(
                    f"Doom loop detected (attempt {attempt}): agent produced "
                    f"identical output to a previous attempt. Breaking retry cycle."
                )
                logger.warning(
                    "Doom loop detected for format_type=%s at attempt %d",
                    context.get("format_type", "unknown"),
                    attempt,
                )
                break
            seen_payload_hashes.add(payload_hash)

            validation = self.validator.validate(result.payload)
            if validation.valid:
                return HarnessResult(
                    success=True,
                    format_type=context.get("format_type", "unknown"),
                    payload=validation.validated_payload,
                    error_log=error_log,
                    attempts=attempt,
                )

            error_log.append(
                f"Validation failed (attempt {attempt}): {validation.error_message}"
            )
            context["correction_hint"] = validation.error_message

        return HarnessResult(
            success=False,
            format_type=context.get("format_type", "unknown"),
            error_log=error_log,
            attempts=attempt,
        )
