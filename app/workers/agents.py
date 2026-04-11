from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum

class AgentActionStatus(str, Enum):
    SUCCESS = "SUCCESS"          # Passed and ready for next stage
    REVISION_NEEDED = "REVISION_NEEDED" # Red Team rejected, back to drawing board
    ESCALATE = "ESCALATE"        # Failed max retries, humans need to look
    ERROR = "ERROR"              # System/API failure

class AgentResult(BaseModel):
    """The standard currency of the Content Factory."""
    status: AgentActionStatus
    payload: Dict[str, Any] = Field(description="The structured output (Research, Script, etc.)")
    reasoning: str = Field(description="Chain-of-Thought log for audit trails and Red Team debates.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Self-assessed or Evaluator-assessed confidence.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Token usage, model version, latency.")

class BaseAgent(ABC):
  """
  Abstract Base Agent. 
  In 2026, we default to Gemini 3.1 Flash for high-volume tasks, 
  and Gemini 3.1 Pro for deep reasoning (Red Team / Strategy).
  """
  def __init__(self, model_name: str = "gemini-3.1-flash", temperature: float = 0.2):
    self.model_name = model_name
    self.temperature = temperature

  @abstractmethod
  async def run(self, context: Dict[str, Any], **kwargs) -> AgentResult:
    """
    Executes the agent's core loop.
    Must return a strictly typed AgentResult.
    """
    pass

class ResearchAgent(BaseAgent):
    async def run(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        topic = context.get("topic", "Unknown Topic")
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"chunks": [f"Deep research on {topic}", "Historical data extracted."]},
            reasoning="Simulated RAG extraction using high-context lookup.",
            confidence_score=0.95
        )

class CopywriterAgent(BaseAgent):
    async def run(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"script_content": "Simulated Viral Script: The truth about {topic}."},
            reasoning="Drafted using AEO-optimization and retention-first framing.",
            confidence_score=0.88
        )

class RedTeamAgent(BaseAgent):
    async def run(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        # Mock simple logic: usually succeeds, but could simulate rejection logic if needed
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"verdict": "SUPPORTED", "claims_verified": 3},
            reasoning="Checked script against ResearchChunks. No hallucinations detected.",
            confidence_score=1.0
        )

class AssetStudioAgent(BaseAgent):
    async def run(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"video_url": "s3://factory/renders/mock_video.mp4"},
            reasoning="Generated parallel assets via Veo and Lyria.",
            confidence_score=0.9
        )