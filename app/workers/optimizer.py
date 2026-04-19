import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.workers.agents import BaseAgent, AgentResult, AgentActionStatus

logger = logging.getLogger(__name__)


class OptimizerOutput(BaseModel):
    patched_script_content: str = Field(
        description="The revised script with only broken claims patched"
    )
    patched_storyboard: List[Dict[str, str]] = Field(
        description="Updated storyboard if visual/audio cues changed due to patches"
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
    "## RULES\n"
    "1. DO NOT rewrite the entire script. Patch only the broken claims.\n"
    "2. For each UNSUPPORTED/CONTESTED claim:\n"
    "   a. If the refined_context has correct information -> replace the claim with the correct version\n"
    "   b. If the refined_context lacks evidence -> remove or soften the claim\n"
    "   c. If the claim is a statistic -> find the correct number in refined_context\n"
    "3. Preserve narrative flow, hook, and loop structure.\n"
    "4. Preserve all SUPPORTED claims exactly as they are.\n"
    "5. Maintain the same tone, pacing, and scene structure.\n"
    "6. If patching creates a narrative gap, bridge it minimally.\n"
    "7. Return the FULL patched script (not just diffs).\n"
    "8. Keep the same scene structure in the storyboard unless a visual/audio cue directly "
    "contradicts a patched claim.\n"
    "9. Every patched claim MUST be traceable to the refined_context — zero new hallucinations."
)

OPTIMIZER_HUMAN_TEMPLATE = (
    "Patch the following script. Only modify the broken claims listed below.\n\n"
    "<original_script>\n{original_script}\n</original_script>\n\n"
    "<failed_claims>\n{failed_claims}\n</failed_claims>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "For each failed claim, explain your patch. Then provide the complete patched script and updated storyboard."
)


def format_failed_claims(claims: List[dict]) -> str:
    sections = []
    for i, claim in enumerate(claims, 1):
        sections.append(
            f"Claim {i}: {claim['claim_text']}\n"
            f"Verdict: {claim['verdict']}\n"
            f"Confidence: {claim.get('confidence', 0.0):.2f}\n"
            f"Evidence: {claim.get('evidence_text', 'N/A')}"
        )
    return "\n\n".join(sections)


class ScriptOptimizerAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        original_script = context.get("script_content", "")
        failed_claims = context.get("failed_claims", [])
        refined_context = context.get("refined_context", "")

        if not original_script:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No script content provided for optimization.",
                confidence_score=0.0,
            )

        if not failed_claims:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No failed claims provided. Nothing to optimize.",
                confidence_score=0.0,
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
                "failed_claims": format_failed_claims(failed_claims),
                "refined_context": refined_context,
            }
        )

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={
                "script_content": result.patched_script_content,
                "storyboard": result.patched_storyboard,
                "patch_summary": result.patch_summary,
            },
            reasoning=result.reasoning,
            confidence_score=result.confidence,
            metadata={"model": self.model_name, "agent": "optimizer"},
        )
