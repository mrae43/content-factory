import logging
from typing import Any, ClassVar, Dict, List, Optional, Set, Type

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.workers.agents import LLMAgent, AgentResult, AgentActionStatus

logger = logging.getLogger(__name__)


class OptimizerOutput(BaseModel):
    patched_script_content: str = Field(
        description="The revised script with only broken claims patched"
    )
    patch_summary: str = Field(description="What was changed and why, for audit trail")
    reasoning: str = Field(description="Step-by-step reasoning for each patch")
    confidence: float = Field(
        description="Confidence that patches preserve narrative coherence"
    )


OPTIMIZER_SYSTEM_PROMPT = (
    "You are a Surgical Script Optimizer at the AI Content Factory. "
    "You receive a script that has FAILED fact-checking and a list of specific broken claims.\n"
    "Your job is to patch ONLY those claims while preserving the rest of the script exactly as-is.\n\n"
    "You receive three sources of information:\n"
    "1. `refined_context` — user-provided reference context with editorial directives "
    "(tone, angle, target audience).\n"
    "2. `retrieved_evidence` — raw evidence chunks from the knowledge base.\n"
    "3. `optimization_history` — a record of what was patched in previous iterations "
    "and the resulting verdicts.\n\n"
    "The `retrieved_evidence` is your PRIMARY source for factual corrections.\n"
    "The `refined_context` provides the editorial framing and reference background.\n"
    "The `optimization_history` shows you what has already been tried — avoid re-attempting "
    "the same failed patch, and do not revert a previously-successful patch.\n\n"
    "You also receive `story_directives` — target_audience, tone, and angle — that define the "
    "original framing. Ensure your patches remain consistent with these directives.\n\n"
    "## RULES\n"
    "1. DO NOT rewrite the entire script. Patch only the broken claims.\n"
    "2. For each UNSUPPORTED/CONTESTED claim:\n"
    "   a. If the retrieved_evidence or refined_context has correct information -> replace the "
    "claim with the correct version\n"
    "   b. If both sources lack evidence -> remove or soften the claim\n"
    "   c. If the claim is a statistic -> find the correct number in retrieved_evidence "
    "(fall back to refined_context)\n"
    "3. Preserve narrative flow, hook, and closer structure.\n"
    "4. Preserve all SUPPORTED claims exactly as they are.\n"
    "5. Maintain the same tone and pacing, respecting the story_directives tone and angle.\n"
    "6. If patching creates a narrative gap, bridge it minimally.\n"
    "7. Return the FULL patched script (not just diffs).\n"
    "8. Every patched claim MUST be traceable to retrieved_evidence or refined_context "
    "— zero new hallucinations.\n\n"
    "STOP CONDITIONS — evaluate as you read the input, before producing any output:\n"
    '- If the active_failures list is unclear about what makes a claim "fixed," state\n'
    "  your interpretation of the success criteria before patching. Do not silently\n"
    "  assume the most lenient interpretation.\n"
    "- Do not modify claims that are not in the active_failures list, even if they\n"
    "  appear questionable. Out-of-scope modifications are a regression risk.\n"
    "  If you believe an unlisted claim requires attention, flag it — do not patch it.\n"
    "- Do not skip or defer a claim that appears in the active_failures list. Every\n"
    "  listed claim requires an explicit patch attempt. If a claim cannot be fixed\n"
    "  with the available evidence, set status=ESCALATE and identify which claims\n"
    "  remain unresolved and why.\n"
    "- Use the optimization_history to understand what patches have already been tried. "
    "Avoid repeating failed approaches. Do not revert a patch that previously resolved a claim."
)

OPTIMIZER_HUMAN_TEMPLATE = (
    "Patch the following script. Only modify the broken claims listed below.\n\n"
    "<original_script>\n{original_script}\n</original_script>\n\n"
    "<active_failures>\n{active_failures}\n</active_failures>\n\n"
    "<optimization_history>\n{optimization_history}\n</optimization_history>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<retrieved_evidence>\n{evidence_sections}\n</retrieved_evidence>\n\n"
    "<story_directives>\n{story_directives}\n</story_directives>\n\n"
    "For each failed claim, explain your patch while respecting the story_directives. "
    "Then provide the complete patched script."
)


def format_active_failures(claims: List[dict]) -> str:
    sections = []
    for i, claim in enumerate(claims, 1):
        text = claim.get("text") or claim.get("claim_text", "")
        verdict = claim.get("latest_verdict") or claim.get("verdict", "UNKNOWN")
        reason = claim.get("failure_reason") or claim.get("evidence_text", "N/A")
        sections.append(
            f"Claim {i}: {text}\nVerdict: {verdict}\nFailure reason: {reason}"
        )
    return "\n\n".join(sections)


def format_optimization_history(history: List[dict]) -> str:
    if not history:
        return "No prior optimization history."
    sections = []
    for entry in history:
        iteration = entry.get("iteration", "?")
        patches = entry.get("patches_applied", [])
        patches_text = "\n".join(f"  - {p}" for p in patches) if patches else "  (none)"
        snapshots = entry.get("claims_snapshot", [])
        snapshot_lines = []
        for sc in snapshots:
            snapshot_lines.append(
                f"  - {sc.get('text', '')} → {sc.get('verdict', '?')}"
            )
        snapshot_text = "\n".join(snapshot_lines) if snapshot_lines else "  (no claims)"
        sections.append(
            f"Iteration {iteration}:\n"
            f"Patches applied:\n{patches_text}\n"
            f"Resulting verdicts:\n{snapshot_text}"
        )
    return "\n\n".join(sections)


class ScriptOptimizerAgent(LLMAgent):
    _required_di_tools: ClassVar[List[str]] = []
    _required_llm_tools: ClassVar[List[str]] = []
    _permissions: ClassVar[Set[str]] = {"ScriptOptimizerAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        original_script = context.get("script_content", "")
        active_failures = context.get("active_failures", [])
        optimization_history = context.get("optimization_history", [])
        refined_context = context.get("refined_context", "")

        if not original_script:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No script content provided for optimization.",
                confidence_score=0.0,
            )

        if not active_failures:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No active failures provided. Nothing to optimize.",
                confidence_score=0.0,
            )

        evidence_sections = context.get("evidence_sections", "")
        evidence_prompt = (
            evidence_sections
            if evidence_sections
            else "No additional evidence was retrieved. Rely solely on the refined_context."
        )

        story_directives = context.get("story_directives", {})
        target_audience = story_directives.get("target_audience", "General")
        tone = story_directives.get("tone", "")
        angle = story_directives.get("angle", "")
        story_directives_text = (
            f"Target Audience: {target_audience}\nTone: {tone}\nAngle: {angle}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", OPTIMIZER_SYSTEM_PROMPT),
                ("human", OPTIMIZER_HUMAN_TEMPLATE),
            ]
        )

        chain = prompt | self.llm.with_structured_output(OptimizerOutput)
        result: OptimizerOutput = await chain.ainvoke(
            {
                "original_script": original_script,
                "active_failures": format_active_failures(active_failures),
                "optimization_history": format_optimization_history(
                    optimization_history
                ),
                "refined_context": refined_context,
                "evidence_sections": evidence_prompt,
                "story_directives": story_directives_text,
            }
        )

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={
                "script_content": result.patched_script_content,
                "patch_summary": result.patch_summary,
            },
            reasoning=result.reasoning,
            confidence_score=result.confidence,
            metadata={"model": self.model_name, "agent": "optimizer"},
        )
