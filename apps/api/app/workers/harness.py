import hashlib
import json
import logging
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from app.services.format_validator import FormatValidator
from app.services.tools import ToolRegistry, register_standard_tools
from app.workers.agents import AgentActionStatus, BaseAgent, ServiceAgent

logger = logging.getLogger(__name__)

register_standard_tools()


class HarnessResult(BaseModel):
    success: bool
    format_type: str
    payload: Optional[dict] = None
    error_log: List[str] = Field(default_factory=list)
    attempts: int = 0
    escalated: bool = False


class AgentHarness:
    def __init__(
        self,
        agent: BaseAgent,
        validator: Optional[FormatValidator] = None,
        max_retries: int = 2,
    ):
        self.agent = agent
        self.validator = validator
        self.max_retries = max_retries
        self._inject_tools()

    def _inject_tools(self) -> None:
        registry = ToolRegistry()
        agent_name = type(self.agent).__name__
        permitted = registry.get_permitted_tools(agent_name)
        if permitted:
            self.agent.inject_tools({t.name: t for t in permitted})

    def _inject_llm_tools(self) -> None:
        """Bind permitted LLM-callable tools to the agent's LLM.

        Sets ``agent.llm_with_tools`` (LLM bound with tool schemas) and
        ``agent.llm_tools`` (list of Tool objects for execution) so the
        agent can implement LLM-driven tool-calling inside ``_execute``.
        """
        registry = ToolRegistry()
        agent_name = type(self.agent).__name__
        llm_tools = registry.get_llm_tools(agent_name)
        if llm_tools and hasattr(self.agent, "llm") and self.agent.llm is not None:
            self.agent.llm_with_tools = self.agent.llm.bind_tools(
                [t.llm_schema for t in llm_tools]
            )
            self.agent.llm_tools = {t.name: t for t in llm_tools}

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def run_with_harness(self, context: Dict) -> HarnessResult:
        if isinstance(self.agent, ServiceAgent):
            return await self._run_service_agent(context)
        return await self._run_llm_agent_with_retry(context)

    async def _run_service_agent(self, context: Dict) -> HarnessResult:
        result = await self.agent.run(context=context)

        if result.status == AgentActionStatus.ESCALATE:
            logger.warning(
                "ServiceAgent ESCALATE for %s: %s",
                type(self.agent).__name__,
                result.reasoning,
            )
            return HarnessResult(
                success=False,
                format_type=context.get("format_type", "unknown"),
                error_log=[f"Agent escalated: {result.reasoning}"],
                attempts=1,
                escalated=True,
            )

        if result.status != AgentActionStatus.SUCCESS:
            return HarnessResult(
                success=False,
                format_type=context.get("format_type", "unknown"),
                error_log=[f"Agent failed: {result.reasoning}"],
                attempts=1,
            )

        return HarnessResult(
            success=True,
            format_type=context.get("format_type", "unknown"),
            payload=result.payload,
            attempts=1,
        )

    async def _run_llm_agent_with_retry(self, context: Dict) -> HarnessResult:
        self._inject_llm_tools()
        error_log: List[str] = []
        seen_payload_hashes: Set[str] = set()
        max_attempts = self.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            result = await self.agent.run(context=context)

            if result.status == AgentActionStatus.ESCALATE:
                logger.warning(
                    "Agent ESCALATE for format_type=%s: %s",
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

            if self.validator is not None:
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
            else:
                return HarnessResult(
                    success=True,
                    format_type=context.get("format_type", "unknown"),
                    payload=result.payload,
                    error_log=error_log,
                    attempts=attempt,
                )

        return HarnessResult(
            success=False,
            format_type=context.get("format_type", "unknown"),
            error_log=error_log,
            attempts=attempt,
        )
