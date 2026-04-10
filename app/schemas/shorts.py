from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from typing import List, Dict, Optional, Any
from uuid import UUID
from datetime import datetime
from enum import Enum

# ==========================================
# 1. ENUMS (Mapped directly to DB Enums)
# ==========================================
class JobStatusEnum(str, Enum):
    PENDING = 'PENDING'
    RESEARCHING = 'RESEARCHING'
    FACT_CHECKING_RESEARCH = 'FACT_CHECKING_RESEARCH'
    SCRIPTING = 'SCRIPTING'
    FACT_CHECKING_SCRIPT = 'FACT_CHECKING_SCRIPT'
    ASSET_GENERATION = 'ASSET_GENERATION'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    HUMAN_REVIEW_NEEDED = 'HUMAN_REVIEW_NEEDED'

class AssetTypeEnum(str, Enum):
    VISUAL_VEO = 'VISUAL_VEO'
    AUDIO_LYRIA = 'AUDIO_LYRIA'
    VOICEOVER = 'VOICEOVER'
    SUBTITLE_JSON = 'SUBTITLE_JSON'
    DATA_CHART = 'DATA_CHART'

class VerdictEnum(str, Enum):
    SUPPORTED = 'SUPPORTED'
    CONTESTED = 'CONTESTED'
    UNSUPPORTED = 'UNSUPPORTED'
    UNCERTAIN = 'UNCERTAIN'

# ==========================================
# 2. SHARED CONTEXT MODELS (For JSONB columns)
# ==========================================
class PreContextPayload(BaseModel):
    """Schema for the JSONB pre_context column provided by the user."""
    source_urls: List[HttpUrl] = Field(default_factory=list, description="URLs to scrape for research")
    raw_text: Optional[str] = Field(None, description="Raw copied text or book excerpts")
    target_audience: str = Field("General", description="e.g., Academics, TikTok, Investors")
    guardrail_strictness: str = Field("High", description="Defines how aggressively the Red Team operates")

class AssetRenderMeta(BaseModel):
    """Schema for the JSONB render_meta column in Assets."""
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    synthid_watermark: Optional[str] = Field(None, description="Google SynthID signature for compliance")
    prompt_used: Optional[str] = Field(None, description="The exact Veo/Lyria prompt used")

# ==========================================
# 3. REQUEST SCHEMAS (Input via API)
# ==========================================
class JobCreateRequest(BaseModel):
    """Payload for Step 1: User inserts topic & context."""
    topic: str = Field(..., min_length=3, max_length=200, example="BRICS De-dollarization 2025")
    pre_context: PreContextPayload
    strict_compliance_mode: bool = Field(True, description="Enforce rigorous fact-checking")

class ScriptApprovalRequest(BaseModel):
    """Payload for Human-in-the-loop overrides."""
    is_approved: bool = Field(..., description="Approve or reject the script")
    human_feedback: Optional[str] = Field(None, description="Feedback to send back to the Script Agent")

# ==========================================
# 4. RESPONSE SCHEMAS (Output via API)
# ==========================================
class FactCheckClaimResponse(BaseModel):
    """Outputs for Step 4 & 6: Red Team Evaluation Results."""
    id: UUID
    claim_text: str
    verdict: VerdictEnum
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_references: List[UUID] = Field(default_factory=list, description="IDs of ResearchChunks")

    model_config = ConfigDict(from_attributes=True)

class ScriptResponse(BaseModel):
    """Outputs for Step 5: The Agentic Script."""
    id: UUID
    version: int
    content: str
    is_approved: bool
    feedback_history: List[Dict[str, Any]]
    claims: List[FactCheckClaimResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AssetResponse(BaseModel):
    """Outputs for Step 7: Multi-modal generated assets."""
    id: UUID
    asset_type: AssetTypeEnum
    url_or_path: str
    render_meta: AssetRenderMeta
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class RenderJobResponse(BaseModel):
    """Outputs for Step 8: The Master Object State."""
    id: UUID
    topic: str
    status: JobStatusEnum
    strict_compliance_mode: bool
    final_video_url: Optional[str]
    error_log: Optional[Dict[str, Any]]
    
    # We only expose the most recently active script to keep payloads light
    scripts: List[ScriptResponse] = Field(default_factory=list)
    assets: List[AssetResponse] = Field(default_factory=list)
    
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
