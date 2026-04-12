from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm import get_llm

logger = logging.getLogger(__name__)

class AgentActionStatus(str, Enum):
    SUCCESS = "SUCCESS"          
    REVISION_NEEDED = "REVISION_NEEDED" 
    ESCALATE = "ESCALATE"        
    ERROR = "ERROR"              

class AgentResult(BaseModel):
    """The standard currency of the Content Factory."""
    status: AgentActionStatus
    payload: Dict[str, Any] = Field(description="The structured output (Research, Script, etc.)")
    reasoning: str = Field(description="Chain-of-Thought log for audit trails and Red Team debates.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Self-assessed or Evaluator-assessed confidence.")
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
      before_sleep=lambda retry_state: logger.warning(f"Agent API Error. Retrying in {retry_state.next_action.sleep}s...")
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
    confidence: float = Field(description="Confidence in factual accuracy (0.0 to 1.0).")

class ResearchAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        topic = context.get("topic", "Unknown Topic")
        pre_context = context.get("pre_context", "")

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the Deep Research Agent of the AI Content Factory. Your mission is to establish the ground truth.\n"
                "Prioritize historically accurate, verifiable, and high-impact data points. Ignore opinion, fluff, and low-confidence claims.\n"
                "Truth and Guardrails are first-class citizens. If context is insufficient, state it in your reasoning."
            )),
            ("human", (
                "Identify the most critical facts about the following topic using the provided context.\n"
                "<topic>\n{topic}\n</topic>\n"
                "<pre_context>\n{pre_context}\n</pre_context>\n"
                "First, analyze the input step-by-step. Then, extract the data chunks."
            ))
        ])

        chain = prompt | self.llm.with_structured_output(ResearchSchema)
        result: ResearchSchema = await chain.ainvoke({"topic": topic, "pre_context": pre_context})

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"chunks": result.chunks},
            reasoning=result.reasoning,
            confidence_score=result.confidence,
            metadata={"model": self.model_name}
        )

class CopywriterSchema(BaseModel):
    script_content: str = Field(description="The final narrated script text.")
    storyboard: List[Dict[str, str]] = Field(description="Sequence of scenes [ {visual_prompt: '...', audio_cue: '...'} ]")
    reasoning: str = Field(description="The retention-first psychology used to draft this.")
    confidence: float = Field(description="Self-assessment of hook strength and factual adherence.")

class CopywriterAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        topic = context.get("topic", "Unknown")
        research_chunks = context.get("research_chunks", [])
        feedback = context.get("feedback", "")

        prompt = ChatPromptTemplate.from_messages([])

        chain = prompt | self.llm.with_structured_output(CopywriterSchema)
        result: CopywriterSchema = await chain.ainvoke({
            "topic": topic,
            "research_chunks": research_chunks,
            "feedback": feedback
        })

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={
                "script_content": result.script_content,
                "storyboard": result.storyboard
            },
            reasoning=result.reasoning,
            confidence_score=result.confidence,
            metadata={"model": self.model_name}
        )

class RedTeamAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        # Mock simple logic: usually succeeds, but could simulate rejection logic if needed
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"verdict": "SUPPORTED", "claims_verified": 3},
            reasoning="Checked script against ResearchChunks. No hallucinations detected.",
            confidence_score=1.0
        )

class AssetStudioAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"video_url": "s3://factory/renders/mock_video.mp4"},
            reasoning="Generated parallel assets via Veo and Lyria.",
            confidence_score=0.9
        )