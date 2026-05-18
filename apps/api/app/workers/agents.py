from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from enum import Enum
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm import get_llm
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
    Abstract Base Agent leveraging LangChain and Gemini 2.5 Flash.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature: float = 0.2,
    ):
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
        """Core loop with automatic crash-resilience."""
        return await self._execute(context, **kwargs)

    @abstractmethod
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        """The actual implementation required by child agents."""
        pass


class CitationEntry(BaseModel):
    claim_fragment: str = Field(
        description="Short phrase or claim from the synthesized research"
    )
    source_url: str = Field(description="URL of the source supporting this claim")
    chunk_id: str = Field(description="ID of the ResearchChunk this citation traces to")


class ResearchSchema(BaseModel):
    chunks: List[str] = Field(description="Extracted highly-credible data chunks.")
    refined_context: str = Field(
        description=(
            "A comprehensive, self-contained research summary synthesizing all "
            "retrieved evidence into a single coherent narrative. Must include: "
            "key facts, dates, statistics, quotes, causal relationships, and "
            "competing viewpoints. This summary will be the SOLE input for "
            "script writing — it must be complete enough that a scriptwriter "
            "never needs to consult raw sources."
        )
    )
    citation_index: List[CitationEntry] = Field(
        default_factory=list,
        description=(
            "Sidecar provenance mapping each significant claim to its source URL "
            "and ResearchChunk ID. Fact-Check can trace claims without re-searching "
            "the vector store."
        ),
    )
    reasoning: str = Field(description="Why these facts were prioritized.")
    confidence: float = Field(
        description="Confidence in factual accuracy (0.0 to 1.0)."
    )


class ResearchAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        topic = context.get("topic", "Unknown Topic")
        vector_store = context.get("vector_store")
        job_id = context.get("job_id")

        if not vector_store or not job_id:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="Vector store or job_id not provided. Cannot research without retrieval infrastructure.",
                confidence_score=0.0,
            )

        retrieved = await vector_store.semantic_search(
            query=topic,
            job_id=job_id,
            scopes=["RAW-CONTEXT", "LOCAL"],
            top_k=10,
        )

        if not retrieved:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No context retrieved from vector store above similarity threshold. Ensure pre_context was provided.",
                confidence_score=0.0,
            )

        avg_score = sum(r["similarity_score"] for r in retrieved) / len(retrieved)
        logger.info(
            f"ResearchAgent retrieved {len(retrieved)} chunks, avg similarity: {avg_score:.3f}"
        )

        retrieved_context_text = "\n\n".join(
            [f"Chunk ID {r['id']}: {r['content']}" for r in retrieved]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the Deep Research Agent of the AI Content Factory. Your mission is to establish the ground truth.\n"
                        "Prioritize historically accurate, verifiable, and high-impact data points. Ignore opinion, fluff, and low-confidence claims.\n"
                        "Truth and Guardrails are first-class citizens. If context is insufficient, state it in your reasoning.\n\n"
                        "You must also produce a `refined_context` — a single, comprehensive research summary that:\n"
                        "1. Synthesizes ALL retrieved evidence into a coherent narrative\n"
                        "2. Preserves specific facts: dates, names, statistics, quotes, source attributions\n"
                        "3. Notes areas of conflicting evidence or uncertainty\n"
                        "4. Is self-contained — a scriptwriter using ONLY this summary can write an accurate script\n"
                        "5. Is concise but complete — aim for 800-1500 words, not a list of bullet points\n\n"
                        "CITATION INDEX:\n"
                        "For each significant claim in your synthesis, record which source URL "
                        "and chunk ID it came from in the `citation_index` field. "
                        "This allows the Fact-Check team to trace claims without re-searching "
                        "the vector store."
                    ),
                ),
                (
                    "human",
                    (
                        "Identify the most critical facts about the following topic using the provided context.\n"
                        "<topic>\n{topic}\n</topic>\n"
                        "<retrieved_context>\n{retrieved_context}\n</retrieved_context>\n"
                        "First, analyze the input step-by-step. Then, extract the data chunks.\n\n"
                        "Additionally, write a comprehensive `refined_context` summary that synthesizes all the evidence above into a single coherent research brief. This summary is the ONLY thing the scriptwriter will see — make it count."
                    ),
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(ResearchSchema)
        result: ResearchSchema = await chain.ainvoke(
            {"topic": topic, "retrieved_context": retrieved_context_text}
        )

        if result.chunks:
            logger.info(
                f"ResearchAgent ingesting {len(result.chunks)} REFINED chunks to vector store for Job {job_id}"
            )
            await vector_store.ingest_chunks(
                job_id=job_id, chunks=result.chunks, scope="LOCAL",
                meta={"source_type": "INFERRED"},
            )

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={
                "chunks": result.chunks,
                "refined_context": result.refined_context,
                "citation_index": [c.model_dump() for c in result.citation_index],
            },
            reasoning=result.reasoning,
            confidence_score=result.confidence,
            metadata={"model": self.model_name},
        )


class CopywriterSchema(BaseModel):
    script_content: str = Field(description="The final narrated script text.")
    reasoning: str = Field(
        description="The retention-first psychology used to draft this."
    )
    confidence: float = Field(
        description="Self-assessment of hook strength and factual adherence."
    )


class CopywriterAgent(BaseAgent):
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

        story_directives = context.get("story_directives", {})
        target_audience = story_directives.get("target_audience", "General")
        tone = story_directives.get("tone", "")
        angle = story_directives.get("angle", "")
        story_directives_text = (
            f"Target Audience: {target_audience}\n"
            f"Tone: {tone}\n"
            f"Angle: {angle}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the Lead Scriptwriter for the AI Content Factory. Your mission is to write a compelling, "
                        "format-agnostic master narrative script.\n\n"
                        "## YOUR INPUT\n"
                        "You receive a `refined_context` — a comprehensive research summary vetted and synthesized by the "
                        "research team. This is your SOLE source of truth. Do NOT introduce facts not present in the "
                        "refined_context.\n\n"
                        "You also receive `story_directives` — target_audience, tone, and angle — that guide how the "
                        "script should be framed. Tailor vocabulary, narrative voice, complexity, and perspective to "
                        "match these directives.\n\n"
                        "## RULES\n"
                        "1. ZERO HALLUCINATION: Every claim must trace to the refined_context.\n"
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
    "METHODOLOGY:\n"
    "1. Evaluate each claim independently against its SPECIFIC evidence.\n"
    "2. For each claim, assign one of these verdicts:\n"
    "   - SUPPORTED: Claim is fully verified by the evidence.\n"
    "   - CONTESTED: Evidence contradicts or significantly qualifies the claim.\n"
    "   - UNSUPPORTED: Claim is not found in the evidence or is an exaggeration/misinterpretation.\n"
    "   - UNCERTAIN: Not enough evidence to confirm or deny the claim.\n"
    "3. Provide confidence (0.0-1.0) and the specific evidence text for each claim.\n"
    "4. VERDICT: Overall is SUPPORTED only if every claim is SUPPORTED or UNCERTAIN."
)

EVALUATION_HUMAN = (
    "Audit the following claims against their individually-retrieved evidence:\n"
    "<enriched_claims>\n{enriched_claims}\n</enriched_claims>\n"
    "<target_script>\n{script_content}\n</target_script>\n"
    "Analyze every claim step-by-step against its specific evidence. "
    "For each claim, provide the verdict, confidence, and supporting evidence text."
)


class RedTeamAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script_content = context.get("script_content", "")
        vector_store = context.get("vector_store")
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

        # Filter claims by guardrail extract_categories if configured
        if guardrail_config and extracted.claims:
            allowed = set(guardrail_config.extract_categories)
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

        # Pass 2: Per-claim evidence retrieval
        top_k = guardrail_config.top_k_per_claim if guardrail_config else 5
        threshold = guardrail_config.similarity_threshold if guardrail_config else None
        enriched_claims: List[ClaimEvidence] = []
        if vector_store and job_id:
            for claim in extracted.claims:
                evidence_results = await vector_store.semantic_search(
                    query=claim.search_query,
                    job_id=job_id,
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
    visual_prompts: List[str] = Field(
        description="Prompts tailored for Veo video generation"
    )
    audio_prompts: str = Field(
        description="Prompts tailored for Lyria background scoring"
    )


class AssetStudioAgent(BaseAgent):
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
                        "TECHNICAL SPECS:\n"
                        "1. VEO (Video): Create cinematic 4K prompts. Define camera style (drone, close-up) and lighting (golden hour, high-contrast).\n"
                        "2. LYRIA (Audio): Define orchestral/electronic scoring themes and precise voiceover pacing directives.\n"
                        "3. PYTHON (Data Viz): For charts, specify titles, axis labels, and chart types (e.g., 'Moving average line chart of BRICS GDP').\n\n"
                        "INPUT FORMAT:\n"
                        "- scenes: list of {scene_number, narration_text, visual_prompt, audio_cue, duration_seconds}\n"
                        "- visual_style: overall visual direction for the video\n"
                        "- script: the narrative text for reference"
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
