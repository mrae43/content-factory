from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
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
  Abstract Base Agent. 
  In 2026, we default to Gemini 3.1 Flash for high-volume tasks, 
  and Gemini 3.1 Pro for deep reasoning (Red Team / Strategy).
  """
  def __init__(self, model_name: str = "gemini-3.1-flash", temperature: float = 0.2):
    self.model_name = model_name
    self.temperature = temperature
    self.llm = get_llm(model_name=self.model_name, temperature=self.temperature)

  # Robustness: Agents automatically retry on failure using exponential backoff (e.g. 2s, 4s, 8s)
  @retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(multiplier=1, min=2, max=10),
      reraise=True,
      before_sleep=lambda retry_state: logger.warning(f"Agent API Error. Retrying in {retry_state.next_action.sleep}s...")
  )
  async def run(self, context: Dict[str, Any], **kwargs) -> AgentResult:
    """
    Executes the agent's core loop, with automatic crash-resilience.
    Call _execute internally.
    """
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
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"chunks": [f"Deep research on {topic}", "Historical data extracted."]},
            reasoning="Simulated RAG extraction using high-context lookup.",
            confidence_score=0.95
        )

class CopywriterAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"script_content": "Simulated Viral Script: The truth about {topic}."},
            reasoning="Drafted using AEO-optimization and retention-first framing.",
            confidence_score=0.88
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