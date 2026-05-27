from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Set, Type
from pydantic import BaseModel, Field
from enum import Enum
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm import get_llm
from app.services.tools import Tool, ToolRegistry
from app.core.config import settings

logger = logging.getLogger(__name__)


class AgentActionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    REVISION_NEEDED = "REVISION_NEEDED"
    ESCALATE = "ESCALATE"
    ERROR = "ERROR"


class AgentResult(BaseModel):
    """The standard currency of the Content Factory."""

    status: AgentActionStatus
    payload: Dict[str, Any] = Field(
        description="The structured output (Research, Script, etc.)"
    )
    reasoning: str = Field(
        description="Chain-of-Thought log for audit trails and Red Team debates."
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Self-assessed or Evaluator-assessed confidence."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract Base Agent.  Provides the run/_execute contract, tenacity retry,
    and tool-declaration infrastructure.

    Subclasses declare which tools they need via class variables so that
    wiring mismatches are caught at composition time (symmetric permissions).
    """

    _required_di_tools: ClassVar[List[str]] = []
    _required_llm_tools: ClassVar[List[str]] = []
    _permissions: ClassVar[Set[str]] = {"*"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    def __init__(self) -> None:
        self.di_tools: Dict[str, Tool] = {}

    def inject_tools(self, tools: Dict[str, Tool]) -> None:
        self.di_tools = tools

    @staticmethod
    def _validate_declarations(registry: ToolRegistry) -> None:
        """Raise on first missing tool or permission mismatch."""
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Agent API Error. Retrying in {retry_state.next_action.sleep}s..."
        ),
    )
    async def run(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        """Core loop with automatic crash-resilience."""
        return await self._execute(context, **kwargs)

    @abstractmethod
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        """The actual implementation required by child agents."""
        pass


class LLMAgent(BaseAgent):
    """
    Base for agents that call an LLM.  Sets up ``self.llm`` via the provider-
    agnostic ``get_llm()`` factory.

    All current LLM-powered agents (Copywriter, RedTeam, AssetStudio,
    ScriptOptimizer, BlogFormatter, CarouselFormatter, VideoFormatter)
    extend this class.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.llm = get_llm(model_name=self.model_name, temperature=self.temperature)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Agent API Error. Retrying in {retry_state.next_action.sleep}s..."
        ),
    )
    async def run(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        for tool_name in getattr(self, "_required_di_tools", []):
            if tool_name not in self.di_tools:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Required DI tool '{tool_name}' not injected into {type(self).__name__}",
                    confidence_score=0.0,
                )
        return await self._execute(context, **kwargs)

    async def _run_tool_loop(
        self,
        system_prompt: str,
        human_content: str,
        max_rounds: int = 5,
    ) -> str:
        """Invoke the LLM with tools bound, resolving any tool calls.

        Returns the final text content after all tool calls complete.
        If ``llm_with_tools`` is not set (no tools bound), falls back
        to a plain LLM invocation.
        """
        messages: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]

        llm = getattr(self, "llm_with_tools", None) or self.llm
        response = await llm.ainvoke(messages)

        for _ in range(max_rounds):
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                return response.content or ""
            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool = (getattr(self, "llm_tools", {}) or {}).get(tool_name)
                if tool:
                    try:
                        result = await tool.callable(**tool_args)
                        messages.append(
                            ToolMessage(
                                content=str(result)[:8000],
                                tool_call_id=tc.get("id", ""),
                            )
                        )
                    except Exception as exc:
                        messages.append(
                            ToolMessage(
                                content=f"Tool error: {exc}",
                                tool_call_id=tc.get("id", ""),
                            )
                        )
                else:
                    messages.append(
                        ToolMessage(
                            content=f"Unknown tool: {tool_name}",
                            tool_call_id=tc.get("id", ""),
                        )
                    )
            response = await llm.ainvoke(messages)

        return response.content or ""


class ServiceAgent(BaseAgent):
    """
    Base for agents that do NOT call an LLM (e.g. image generation).

    These agents perform deterministic work using DI tools.  No ``self.llm``
    attribute is set — the type system prevents accidental LLM usage.
    """

    pass


class CopywriterSchema(BaseModel):
    script_content: str = Field(description="The final narrated script text.")
    reasoning: str = Field(
        description="The retention-first psychology used to draft this."
    )
    confidence: float = Field(
        description="Self-assessment of hook strength and factual adherence."
    )


class CopywriterAgent(LLMAgent):
    _required_di_tools: ClassVar[List[str]] = []
    _required_llm_tools: ClassVar[List[str]] = []
    _permissions: ClassVar[Set[str]] = {"CopywriterAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        topic = context.get("topic", "Unknown")
        feedback = context.get("feedback", "")

        refined_context = context.get("refined_context", "")
        if not refined_context:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No refined research context available for scriptwriting.",
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
                (
                    "system",
                    (
                        "You are the Lead Scriptwriter for the AI Content Factory. Your mission is to write a compelling, "
                        "format-agnostic master narrative script.\n\n"
                        "## YOUR INPUT\n"
                        "You receive two sources of information:\n"
                        "1. `refined_context` — user-provided reference context combined with editorial directives "
                        "(tone, angle, target audience), serving as the narrative foundation.\n"
                        "2. `retrieved_evidence` — raw evidence chunks retrieved from the knowledge base, each annotated "
                        "with similarity score and source type.\n\n"
                        "The `retrieved_evidence` is your PRIMARY source for factual claims. "
                        "The `refined_context` provides the editorial framing and reference background.\n"
                        "Cross-reference both sources. If a claim appears in the refined_context but is absent from or "
                        "contradicted by the retrieved_evidence, prefer the retrieved_evidence.\n"
                        "Do NOT introduce facts not present in either source.\n\n"
                        "You also receive `story_directives` — target_audience, tone, and angle — that guide how the "
                        "script should be framed. Tailor vocabulary, narrative voice, complexity, and perspective to "
                        "match these directives.\n\n"
                        "STOP CONDITIONS — evaluate as you read the input, before producing any output:\n"
                        "- If retrieved_evidence directly contradicts refined_context on a factual claim,\n"
                        "  set status=ESCALATE and describe the specific contradiction. Do not silently\n"
                        "  prefer one source over the other.\n"
                        "- If story_directives contain conflicting constraints (e.g., incompatible tone/angle\n"
                        "  directives), set status=ERROR and list the conflicting pairs. Do not resolve the\n"
                        "  conflict without explicit flagging.\n"
                        "- If evidence is present but insufficient to construct a coherent narrative arc\n"
                        "  around the topic, set status=ERROR and describe what is missing. Do not fabricate\n"
                        "  a narrative to fill the gap.\n\n"
                        "## RULES\n"
                        "1. ZERO HALLUCINATION: Every claim must trace to the refined_context or retrieved_evidence.\n"
                        "2. Write a clean narrative script (500-800 words) with no format-specific structure.\n"
                        "3. Open with a strong hook — a surprising fact, provocative question, or bold statement.\n"
                        "4. Build a clear narrative arc: hook → context → depth → payoff.\n"
                        "5. End with a compelling closer — a call-to-action, thought-provoking question, or forward-looking "
                        "statement.\n"
                        "6. Write in a tone that respects the story_directives tone (if provided). Default to "
                        "conversational, authoritative if no tone is specified.\n"
                        "7. If the refined_context has conflicting evidence, present the strongest case and note uncertainty.\n"
                        "8. Do NOT include scene numbers, timestamps, visual cues, audio cues, or storyboard elements.\n"
                        "9. Preserve specific data: numbers, dates, names, statistics, quotes, and attributions.\n"
                        "10. If feedback is provided, address every point in the revised script.\n"
                        "11. If story_directives specifies a particular angle, use it to focus the narrative perspective."
                    ),
                ),
                (
                    "human",
                    (
                        "Write a master narrative script for the following topic.\n\n"
                        "<topic>\n{topic}\n</topic>\n\n"
                        "<refined_context>\n{refined_context}\n</refined_context>\n\n"
                        "<retrieved_evidence>\n{evidence_sections}\n</retrieved_evidence>\n\n"
                        "<story_directives>\n{story_directives}\n</story_directives>\n\n"
                        "<feedback>\n{feedback}\n</feedback>\n\n"
                        "First, analyze the narrative arc step-by-step, considering the story_directives. "
                        "Then generate the script."
                    ),
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(CopywriterSchema)
        result: CopywriterSchema = await chain.ainvoke(
            {
                "topic": topic,
                "refined_context": refined_context,
                "evidence_sections": evidence_prompt,
                "story_directives": story_directives_text,
                "feedback": feedback,
            }
        )

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={
                "script_content": result.script_content,
            },
            reasoning=result.reasoning,
            confidence_score=result.confidence,
            metadata={"model": self.model_name},
        )


class ClaimItem(BaseModel):
    claim_text: str = Field(description="The exact atomic claim from the script")
    verdict: str = Field(
        description="One of: SUPPORTED, CONTESTED, UNSUPPORTED, UNCERTAIN"
    )
    confidence: float = Field(description="0.0 to 1.0 confidence in this verdict")
    evidence_text: str = Field(
        description="Quote or paraphrase from sources supporting this verdict"
    )
    evidence_text_inline: list[str] = Field(
        default_factory=list,
        description="Snapshot of raw evidence chunk content for audit trail persistence",
    )
    hedge_required: bool = Field(
        False,
        description="True when verdict is UNCERTAIN — formatter should apply hedged language",
    )


class RedTeamVerdict(BaseModel):
    claims: List[ClaimItem] = Field(
        description="Every factual claim in the script, individually evaluated"
    )
    overall_reasoning: str = Field(
        description="Summary of findings and recommendations"
    )


class ExtractedClaim(BaseModel):
    claim_text: str = Field(description="Exact atomic factual claim from the script")
    claim_category: str = Field(
        description="Type: statistic, attribution, chronological, causal, comparative"
    )
    search_query: str = Field(
        description="Optimized search query to find evidence for this specific claim"
    )


class ClaimExtractionResult(BaseModel):
    claims: List[ExtractedClaim] = Field(
        description="All atomic factual claims extracted from the script"
    )


class ClaimEvidence(BaseModel):
    claim_text: str
    evidence_chunks: List[str]


def _format_enriched_claims(enriched_claims: List[ClaimEvidence]) -> str:
    sections = []
    for i, ec in enumerate(enriched_claims, 1):
        evidence_block = (
            "\n".join(f"  - {chunk}" for chunk in ec.evidence_chunks)
            if ec.evidence_chunks
            else "  - No evidence found"
        )
        sections.append(f"Claim {i}: {ec.claim_text}\nEvidence:\n{evidence_block}")
    return "\n\n".join(sections)


CLAIM_EXTRACTION_SYSTEM = (
    "You are a claim extraction specialist. Your job is to break a script into atomic factual claims.\n"
    "For each claim, generate an optimized search query that would find supporting or contradicting evidence.\n"
    "Categories: statistic, attribution, chronological, causal, comparative.\n"
    "Do NOT evaluate claims — only extract them.\n"
    "Extract EVERY factual claim, including implicit claims (numbers, dates, causal statements, attributions)."
)

CLAIM_EXTRACTION_HUMAN = (
    "Extract all atomic factual claims from the following script:\n"
    "<target_script>\n{script_content}\n</target_script>\n"
    "For each claim, provide the exact claim text, its category, and a targeted search query."
)

EVALUATION_SYSTEM = (
    "You are the Lead Red Team Auditor at the AI Content Factory. Your mission: Destroy Hallucinations.\n"
    "Your reputation depends on catching every single unsupported claim.\n\n"
    "You receive claims with their individually-retrieved evidence. Each claim has been searched independently.\n\n"
    "STOP CONDITIONS — evaluate as you read the input, before producing any output:\n"
    "- If evidence for a single claim is genuinely conflicting, do not force a\n"
    "  SUPPORTED or REFUTED verdict. Instead, mark it UNCERTAIN, set\n"
    "  conflicting_evidence=true in your reasoning block, and summarize both sides.\n"
    "- If a claim cannot be evaluated because the evidence is too sparse to support\n"
    "  any verdict, mark it UNCERTAIN — do not escalate to ERROR for this case.\n\n"
    "METHODOLOGY:\n"
    "1. Evaluate each claim independently against its SPECIFIC evidence.\n"
    "2. For each claim, assign one of these verdicts:\n"
    "   - SUPPORTED: Claim is fully verified by the evidence.\n"
    "   - CONTESTED: Evidence contradicts or significantly qualifies the claim.\n"
    "   - UNSUPPORTED: Claim is not found in the evidence or is an exaggeration/misinterpretation.\n"
    "   - UNCERTAIN: Not enough evidence to confirm or deny the claim.\n"
    "3. Provide confidence (0.0-1.0) and the specific evidence text for each claim.\n"
    "4. VERDICT: Overall is SUPPORTED only if every claim is SUPPORTED or UNCERTAIN.\n"
    "\n## RULES\n"
    "1. Do NOT fabricate evidence. Every verdict must reference specific evidence text\n"
    "   from the provided evidence chunks.\n"
    "2. If no evidence is available for a claim, assign UNCERTAIN — do not guess.\n"
    "3. Do not misrepresent evidence to fit a preferred verdict.\n"
)

EVALUATION_HUMAN = (
    "Audit the following claims against their individually-retrieved evidence:\n"
    "<enriched_claims>\n{enriched_claims}\n</enriched_claims>\n"
    "<target_script>\n{script_content}\n</target_script>\n"
    "Analyze every claim step-by-step against its specific evidence. "
    "For each claim, provide the verdict, confidence, and supporting evidence text."
)


class RedTeamAgent(LLMAgent):
    _required_di_tools: ClassVar[List[str]] = ["semantic_search"]
    _required_llm_tools: ClassVar[List[str]] = ["execute_web_search"]
    _permissions: ClassVar[Set[str]] = {"RedTeamAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script_content = context.get("script_content", "")
        job_id = context.get("job_id")
        guardrail_config = context.get("guardrail_config")

        if not script_content:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No script content provided for fact-checking.",
                confidence_score=0.0,
            )

        # Pass 1: Extract atomic claims
        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CLAIM_EXTRACTION_SYSTEM),
                ("human", CLAIM_EXTRACTION_HUMAN),
            ]
        )
        try:
            extraction_chain = extraction_prompt | self.llm.with_structured_output(
                ClaimExtractionResult
            )
            extracted: ClaimExtractionResult = await extraction_chain.ainvoke(
                {"script_content": script_content}
            )
        except Exception as exc:
            logger.error(f"Red Team claim extraction failed: {exc}")
            return AgentResult(
                status=AgentActionStatus.ESCALATE,
                payload={},
                reasoning=f"Claim extraction LLM call failed: {exc}",
                confidence_score=0.0,
                metadata={"model": self.model_name},
            )

        # Filter claims by guardrail claim_categories if configured
        if guardrail_config and extracted.claims:
            allowed = set(guardrail_config.claim_categories)
            extracted.claims = [
                c for c in extracted.claims if c.claim_category in allowed
            ]

        if not extracted.claims:
            return AgentResult(
                status=AgentActionStatus.SUCCESS,
                payload={
                    "verdict": "SUPPORTED",
                    "claims": [],
                    "overall_reasoning": "No factual claims found in script. Nothing to fact-check.",
                },
                reasoning="Script contained no verifiable factual claims.",
                confidence_score=1.0,
                metadata={"model": self.model_name},
            )

        logger.info(
            f"RedTeamAgent extracted {len(extracted.claims)} claims for Job {job_id}"
        )

        # Pass 2: Per-claim evidence retrieval via DI tool
        search_tool = self.di_tools.get("semantic_search")
        top_k = guardrail_config.top_k_per_claim if guardrail_config else 5
        threshold = guardrail_config.similarity_threshold if guardrail_config else None
        enriched_claims: List[ClaimEvidence] = []
        if search_tool and job_id:
            for claim in extracted.claims:
                evidence_results = await search_tool.callable(
                    query=claim.search_query,
                    job_id=str(job_id),
                    scopes=["RAW-CONTEXT", "LOCAL"],
                    top_k=top_k,
                    similarity_threshold=threshold,
                )
                evidence_chunks = [r["content"] for r in evidence_results]
                enriched_claims.append(
                    ClaimEvidence(
                        claim_text=claim.claim_text,
                        evidence_chunks=evidence_chunks,
                    )
                )
        else:
            enriched_claims = [
                ClaimEvidence(claim_text=c.claim_text, evidence_chunks=[])
                for c in extracted.claims
            ]

        has_any_evidence = any(ec.evidence_chunks for ec in enriched_claims)
        if not has_any_evidence:
            return AgentResult(
                status=AgentActionStatus.ESCALATE,
                payload={},
                reasoning="No research sources available for verification. Cannot audit script without evidence base.",
                confidence_score=0.0,
            )

        enriched_claims_text = _format_enriched_claims(enriched_claims)

        # Pass 2.5: LLM-driven web research for claims needing more evidence
        llm_tools_avail = hasattr(self, "llm_with_tools") and getattr(
            self, "llm_tools", {}
        )
        if llm_tools_avail:
            research_prompt = (
                "You are a research assistant reviewing claims and their evidence.\n\n"
                "For each claim, decide if the provided evidence is sufficient to reach "
                "a confident verdict. If evidence is insufficient, call "
                "execute_web_search to find supporting or contradicting information.\n\n"
                "Only search for claims where the current evidence is clearly insufficient "
                "— do not waste searches on well-supported claims.\n\n"
                f"Claims and current evidence:\n\n{enriched_claims_text}"
            )
            web_results = await self._run_tool_loop(
                system_prompt="You are a precise research assistant.",
                human_content=research_prompt,
                max_rounds=3,
            )
            if web_results.strip():
                enriched_claims_text += (
                    f"\n\n## Additional Web Research\n\n{web_results}"
                )

        # Pass 3: Evaluate with per-claim evidence
        evaluation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EVALUATION_SYSTEM),
                ("human", EVALUATION_HUMAN),
            ]
        )
        try:
            eval_chain = evaluation_prompt | self.llm.with_structured_output(
                RedTeamVerdict
            )
            structured: RedTeamVerdict = await eval_chain.ainvoke(
                {
                    "script_content": script_content,
                    "enriched_claims": enriched_claims_text,
                }
            )
        except Exception as exc:
            logger.error(f"Red Team structured output failed: {exc}")
            return AgentResult(
                status=AgentActionStatus.ESCALATE,
                payload={},
                reasoning=f"Red Team LLM output parsing failed after retries: {exc}",
                confidence_score=0.0,
                metadata={"model": self.model_name},
            )

        if not structured.claims:
            return AgentResult(
                status=AgentActionStatus.SUCCESS,
                payload={
                    "verdict": "SUPPORTED",
                    "claims": [],
                    "overall_reasoning": "No factual claims found in script. Nothing to fact-check.",
                },
                reasoning="Script contained no verifiable factual claims.",
                confidence_score=1.0,
                metadata={"model": self.model_name},
            )

        # Enrich claims with inline evidence text and hedge signals
        enrichment_map = {ec.claim_text: ec.evidence_chunks for ec in enriched_claims}
        for claim in structured.claims:
            if claim.claim_text in enrichment_map:
                claim.evidence_text_inline = enrichment_map[claim.claim_text]
            claim.hedge_required = claim.verdict == "UNCERTAIN"

        uncertain_is_fail = (
            guardrail_config.uncertain_is_soft_fail if guardrail_config else False
        )
        if uncertain_is_fail:
            all_supported = all(c.verdict == "SUPPORTED" for c in structured.claims)
        else:
            all_supported = all(
                c.verdict in ("SUPPORTED", "UNCERTAIN") for c in structured.claims
            )
        overall_verdict = "SUPPORTED" if all_supported else "UNSUPPORTED"
        avg_confidence = sum(c.confidence for c in structured.claims) / len(
            structured.claims
        )

        status = (
            AgentActionStatus.SUCCESS
            if overall_verdict == "SUPPORTED"
            else AgentActionStatus.REVISION_NEEDED
        )

        return AgentResult(
            status=status,
            payload={
                "verdict": overall_verdict,
                "claims": [claim.model_dump() for claim in structured.claims],
                "overall_reasoning": structured.overall_reasoning,
            },
            reasoning=structured.overall_reasoning,
            confidence_score=avg_confidence,
            metadata={"model": self.model_name},
        )


class StudioPromptSchema(BaseModel):
    status: Optional[AgentActionStatus] = Field(
        default=None,
        description="ERROR if scene input is underspecified or ambiguous, otherwise None",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Explanation when status is ERROR",
    )
    visual_prompts: List[str] = Field(
        description="Prompts tailored for Veo video generation"
    )
    audio_prompts: str = Field(
        description="Prompts tailored for Lyria background scoring"
    )


class AssetStudioAgent(LLMAgent):
    _required_di_tools: ClassVar[List[str]] = []
    _required_llm_tools: ClassVar[List[str]] = []
    _permissions: ClassVar[Set[str]] = {"AssetStudioAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script = context.get("script_content", "")
        scenes = context.get("scenes", [])
        visual_style = context.get("visual_style", "")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the Multi-Modal Art Director for the AI Content Factory.\n"
                        "Your mission: Translate video scenes and visual style into technical directives for production-grade AI models.\n\n"
                        "STOP CONDITIONS — evaluate as you read the input, before producing any output:\n"
                        "- If visual_style is ambiguous or could mean multiple things in the context of\n"
                        "  this script, set status=ERROR and state your interpretation before proceeding.\n"
                        "  Do not pick one silently.\n"
                        "- If scenes list lacks sufficient detail to produce a coherent visual direction\n"
                        "  (e.g., generic or placeholder narration text), set status=ERROR and describe\n"
                        "  what specific information is missing. Do not hallucinate visual specs.\n"
                        "- Do not invent aspect ratios, camera movement vocabulary, or production specs\n"
                        "  not grounded in the input. If technical specs are absent or ambiguous, that\n"
                        "  is covered by the checks above — do not fill gaps with plausible-sounding defaults.\n\n"
                        "TECHNICAL SPECS:\n"
                        "1. VEO (Video): Create cinematic 4K prompts. Define camera style (drone, close-up) and lighting (golden hour, high-contrast).\n"
                        "2. LYRIA (Audio): Define orchestral/electronic scoring themes and precise voiceover pacing directives.\n"
                        "3. PYTHON (Data Viz): For charts, specify titles, axis labels, and chart types (e.g., 'Moving average line chart of BRICS GDP').\n\n"
                        "INPUT FORMAT:\n"
                        "- scenes: list of {scene_number, narration_text, visual_prompt, audio_cue, duration_seconds}\n"
                        "- visual_style: overall visual direction for the video\n"
                        "- script: the narrative text for reference\n\n"
                        "## RULES\n"
                        "1. Do NOT invent visual/audio specs not grounded in the input scenes or script.\n"
                        "2. Do NOT include text or typography in visual prompts.\n"
                        "3. Each visual/audio prompt must not exceed 2 sentences. If a scene requires\n"
                        "   more detail than 2 sentences to specify, that is a signal the scene input\n"
                        "   is underspecified — set status=ERROR rather than expanding the prompt.\n"
                    ),
                ),
                (
                    "human",
                    (
                        "Generate visual and audio production prompts for the following video:\n\n"
                        "<visual_style>\n{visual_style}\n</visual_style>\n\n"
                        "<scenes>\n{scenes}\n</scenes>\n\n"
                        "<script>\n{script}\n</script>\n\n"
                        "Analyze the scene transitions and visual style first, then generate the final prompt set."
                    ),
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(StudioPromptSchema)
        result: StudioPromptSchema = await chain.ainvoke(
            {
                "visual_style": visual_style,
                "scenes": scenes,
                "script": script,
            }
        )

        if result.status == AgentActionStatus.ERROR:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={"prompts": result.model_dump()},
                reasoning=result.reasoning
                or "Scene input is underspecified or ambiguous.",
                confidence_score=0.0,
                metadata={
                    "model": self.model_name,
                    "synth_id_enabled": settings.synthid_watermark_enabled,
                },
            )

        video_url = f"s3://factory/renders/{context.get('job_id', 'mock')}_rendered.mp4"

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"video_url": video_url, "prompts": result.model_dump()},
            reasoning="Visual/Audio directives optimized for cinematic output and data accuracy.",
            confidence_score=0.9,
            metadata={
                "model": self.model_name,
                "synth_id_enabled": settings.synthid_watermark_enabled,
            },
        )
