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
    Abstract Base Agent leveraging LangChain and Gemini 3.1.
    """

    def __init__(self, model_name: str = "gemini-3.1-flash", temperature: float = 0.2):
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


class ResearchSchema(BaseModel):
    chunks: List[str] = Field(description="Extracted highly-credible data chunks.")
    reasoning: str = Field(description="Why these facts were prioritized.")
    confidence: float = Field(
        description="Confidence in factual accuracy (0.0 to 1.0)."
    )


class ResearchAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        topic = context.get("topic", "Unknown Topic")
        pre_context = context.get("pre_context", "")


        vector_store = context.get("vector_store")
        job_id = context.get("job_id")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the Deep Research Agent of the AI Content Factory. Your mission is to establish the ground truth.\n"
                        "Prioritize historically accurate, verifiable, and high-impact data points. Ignore opinion, fluff, and low-confidence claims.\n"
                        "Truth and Guardrails are first-class citizens. If context is insufficient, state it in your reasoning."
                    ),
                ),
                (
                    "human",
                    (
                        "Identify the most critical facts about the following topic using the provided context.\n"
                        "<topic>\n{topic}\n</topic>\n"
                        "<pre_context>\n{pre_context}\n</pre_context>\n"
                        "First, analyze the input step-by-step. Then, extract the data chunks."
                    ),
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(ResearchSchema)
        result: ResearchSchema = await chain.ainvoke(
            {"topic": topic, "pre_context": pre_context}
        )

        if vector_store and job_id and result.chunks:
            logger.info(f"ResearchAgent ingesting {len(result.chunks)} chunks to vector store for Job {job_id}")
            await vector_store.ingest_chunks(job_id=job_id, chunks=result.chunks, scope="LOCAL")

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"chunks": result.chunks},
            reasoning=result.reasoning,
            confidence_score=result.confidence,
            metadata={"model": self.model_name},
        )


class CopywriterSchema(BaseModel):
    script_content: str = Field(description="The final narrated script text.")
    storyboard: List[Dict[str, str]] = Field(
        description="Sequence of scenes [ {visual_prompt: '...', audio_cue: '...'} ]"
    )
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

        vector_store = context.get("vector_store")
        job_id = context.get("job_id")

        research_chunks_text = ""
        if vector_store and job_id:
            # The copywriter searches the DB based on the topic to find what to write about
            retrieved = await vector_store.semantic_search(query=topic, job_id=job_id, top_k=10)
            research_chunks_text = "\n".join([r["content"] for r in retrieved])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the Lead Scriptwriter for the AI Content Factory. Your mission is to write high-retention scripts.\n"
                        "STRATEGY: Use the Hook-Value-Loop framework. Ensure every 3 seconds has a visual 'pattern interrupt'.\n"
                        "CONSTRAINTS:\n"
                        "1. ZERO HALLUCINATION: Use only facts from <research_chunks>. If a claim isn't there, don't use it.\n"
                        "2. MULTI-MODAL: Provide clear prompts for visual generation (Veo) and audio/SFX (Lyria).\n"
                        "3. DATA VIZ: Specify when to show a Python-generated chart to support key numbers."
                    ),
                ),
                (
                    "human",
                    (
                        "Create a viral script and storyboard for this topic:\n"
                        "<topic>\n{topic}\n</topic>\n"
                        "<research_chunks>\n{research_chunks}\n</research_chunks>\n"
                        "<feedback>\n{feedback}\n</feedback>\n"
                        "Analyze the narrative arc step-by-step, then generate the script and storyboard JSON structure."
                    ),
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(CopywriterSchema)
        result: CopywriterSchema = await chain.ainvoke(
            {"topic": topic, "research_chunks": research_chunks_text, "feedback": feedback}
        )

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={
                "script_content": result.script_content,
                "storyboard": result.storyboard,
            },
            reasoning=result.reasoning,
            confidence_score=result.confidence,
            metadata={"model": self.model_name},
        )


class RedTeamSchema(BaseModel):
    verdict: str = Field(description="Must be exactly 'SUPPORTED' or 'UNSUPPORTED'")
    claims_verified: int = Field(description="Number of claims fact-checked")
    reasoning: str = Field(
        description="Detailed explanation of the verdict, highlighting any hallucinations."
    )


class RedTeamAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script_content = context.get("script_content", "")
        
        vector_store = context.get("vector_store")
        job_id = context.get("job_id")

        research_sources_text = ""
        if vector_store and job_id:
            # The copywriter searches the DB based on the topic to find what to write about
            retrieved = await vector_store.semantic_search(query=script_content, job_id=job_id, top_k=10)
            research_sources_text = "\n".join([r["content"] for r in retrieved])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the Lead Red Team Auditor at the AI Content Factory. Your mission: Destroy Hallucinations.\n"
                        "Your reputation depends on catching every single unsupported claim.\n"
                        "METHODOLOGY:\n"
                        "1. Break the script into atomic factual claims.\n"
                        "2. Cross-reference each claim against the <research_sources>.\n"
                        "3. If a claim is an exaggeration, a misinterpretation, or not mentioned, mark it UNSUPPORTED.\n"
                        "VERDICT: Only 'SUPPORTED' if 100% of claims are verified."
                    ),
                ),
                (
                    "human",
                    (
                        "Audit the following script against the research data:\n"
                        "<research_sources>\n{research_chunks}\n</research_sources>\n"
                        "<target_script>\n{script_content}\n</target_script>\n"
                        "Analyze the claims step-by-step, then output the structured report."
                    ),
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(RedTeamSchema)
        result: RedTeamSchema = await chain.ainvoke(
            {"script_content": script_content, "research_chunks": research_sources_text}
        )

        status = (
            AgentActionStatus.SUCCESS
            if result.verdict == "SUPPORTED"
            else AgentActionStatus.REVISION_NEEDED
        )

        return AgentResult(
            status=status,
            payload={
                "verdict": result.verdict,
                "claims_verified": result.claims_verified,
            },
            reasoning=result.reasoning,
            confidence_score=1.0,
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
        # In V1, we use Gemini Multi-modal to generate the PROMPTS for Veo/Lyria,
        # rather than fully generating the video in python yet.
        script = context.get("script_content", "")
        storyboard = context.get("storyboard", [])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the Multi-Modal Art Director for the AI Content Factory.\n"
                        "Your mission: Translate creative storyboards into technical directives for production-grade AI models.\n"
                        "TECHNICAL SPECS:\n"
                        "1. VEO (Video): Create cinematic 4K prompts. Define camera style (drone, close-up) and lighting (golden hour, high-contrast).\n"
                        "2. LYRIA (Audio): Define orchestral/electronic scoring themes and precise voiceover pacing directives.\n"
                        "3. PYTHON (Data Viz): For charts, specify titles, axis labels, and chart types (e.g., 'Moving average line chart of BRICS GDP')."
                    ),
                ),
                (
                    "human",
                    (
                        "Refine the technical assets for the following script and storyboard:\n"
                        "<script>\n{script}\n</script>\n"
                        "<storyboard>\n{storyboard}\n</storyboard>\n"
                        "Analyze the scene transitions first, then generate the final prompt set."
                    ),
                ),
            ]
        )

        chain = prompt | self.llm.with_structured_output(StudioPromptSchema)
        result: StudioPromptSchema = await chain.ainvoke(
            {"script": script, "storyboard": storyboard}
        )

        # Mocking the actual generation URL return for MVP
        video_url = f"s3://factory/renders/{context.get('job_id', 'mock')}_rendered.mp4"

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"video_url": video_url, "prompts": result.dict()},
            reasoning="Visual/Audio directives optimized for cinematic output and data accuracy.",
            confidence_score=0.9,
            metadata={
                "model": self.model_name,
                "synth_id_enabled": settings.synthid_watermark_enabled,
            },
        )
