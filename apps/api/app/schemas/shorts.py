from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    HttpUrl,
    Discriminator,
    Tag,
    model_validator,
)
from typing import Annotated, List, Dict, Optional, Any, Union
from uuid import UUID
from datetime import datetime
from enum import Enum

from app.schemas.formats import (
    VideoFormatPayload,
    BlogFormatPayload,
    CarouselFormatPayload,
)


def _discriminate_format(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("_format") or v.get("format") or ""
    return getattr(v, "format", "")


FormatPayload = Annotated[
    Union[
        Annotated[VideoFormatPayload, Tag("video")],
        Annotated[BlogFormatPayload, Tag("blog")],
        Annotated[CarouselFormatPayload, Tag("carousel")],
    ],
    Discriminator(_discriminate_format),
]


# ==========================================
# 1. ENUMS (Mapped directly to DB Enums)
# ==========================================
class JobStatusEnum(str, Enum):
    # Editorial desk names (used in UI/outward-facing):
    #   PENDING              → Queued
    #   RESEARCHING          → Research Desk
    #   RETRIEVAL            → Retrieval Desk
    #   FACT_CHECKING_RESEARCH → Source Verification
    #   SCRIPTING            → Writer's Desk
    #   FACT_CHECKING_SCRIPT → Fact-Check Desk
    #   FORMATTING           → Layout Desk
    #   ASSET_GENERATION     → Production Studio
    #   COMPLETED            → Published
    #   FAILED               → Killed
    #   HUMAN_REVIEW_NEEDED  → Your Review
    PENDING = "PENDING"
    RESEARCHING = "RESEARCHING"
    RETRIEVAL = "RETRIEVAL"
    FACT_CHECKING_RESEARCH = "FACT_CHECKING_RESEARCH"
    SCRIPTING = "SCRIPTING"
    FACT_CHECKING_SCRIPT = "FACT_CHECKING_SCRIPT"
    FORMATTING = "FORMATTING"
    ASSET_GENERATION = "ASSET_GENERATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HUMAN_REVIEW_NEEDED = "HUMAN_REVIEW_NEEDED"


class FormatTypeEnum(str, Enum):
    ALL = "all"
    VIDEO = "video"
    BLOG = "blog"
    CAROUSEL = "carousel"


class AssetTypeEnum(str, Enum):
    CAROUSEL_SLIDE = "CAROUSEL_SLIDE"
    VISUAL_VEO = "VISUAL_VEO"
    AUDIO_LYRIA = "AUDIO_LYRIA"
    VOICEOVER = "VOICEOVER"
    SUBTITLE_JSON = "SUBTITLE_JSON"
    DATA_CHART = "DATA_CHART"


class VerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTESTED = "CONTESTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTAIN = "UNCERTAIN"


class FormatJobStatusEnum(str, Enum):
    PENDING = "PENDING"
    FORMATTING = "FORMATTING"
    ASSET_GENERATION = "ASSET_GENERATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HUMAN_REVIEW_NEEDED = "HUMAN_REVIEW_NEEDED"


class PlatformEnum(str, Enum):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"


def next_status_after_fact_check(format_type: str) -> "JobStatusEnum":
    return JobStatusEnum.FORMATTING


PLATFORM_FORMAT_MAP: dict[PlatformEnum, list[FormatTypeEnum]] = {
    PlatformEnum.TWITTER: [
        FormatTypeEnum.BLOG,
        FormatTypeEnum.CAROUSEL,
        FormatTypeEnum.VIDEO,
    ],
    PlatformEnum.LINKEDIN: [FormatTypeEnum.CAROUSEL, FormatTypeEnum.BLOG],
    PlatformEnum.INSTAGRAM: [FormatTypeEnum.CAROUSEL, FormatTypeEnum.VIDEO],
    PlatformEnum.TIKTOK: [FormatTypeEnum.CAROUSEL, FormatTypeEnum.VIDEO],
    PlatformEnum.YOUTUBE: [FormatTypeEnum.BLOG, FormatTypeEnum.VIDEO],
}


def resolve_formats(
    platform: PlatformEnum, format_type: FormatTypeEnum
) -> list[FormatTypeEnum]:
    """Expand 'all' into platform-specific formats, or return [format_type] if specific."""
    if format_type == FormatTypeEnum.ALL:
        if platform not in PLATFORM_FORMAT_MAP:
            raise ValueError(f"Unknown platform '{platform.value}' in resolve_formats")
        return PLATFORM_FORMAT_MAP[platform]
    if (
        platform not in PLATFORM_FORMAT_MAP
        or format_type not in PLATFORM_FORMAT_MAP[platform]
    ):
        raise ValueError(
            f"Format '{format_type.value}' is not valid for platform '{platform.value}'. "
            f"Valid formats: {[f.value for f in PLATFORM_FORMAT_MAP[platform]]}"
        )
    return [format_type]


# ==========================================
# 2. SHARED CONTEXT MODELS (For JSONB columns)
# ==========================================
class ResearchInputs(BaseModel):
    """User-provided source material — consumed by the Indexing sub-phase."""

    source_urls: List[HttpUrl] = Field(
        default_factory=list, description="URLs to scrape for research"
    )


class StoryDirectives(BaseModel):
    """Editorial guardrails and creative direction — consumed by Synthesis & Scripting."""

    target_audience: str = Field(
        "General", description="e.g., Academics, TikTok, Investors"
    )
    guardrail_strictness: str = Field(
        "High",
        pattern="^(Low|Medium|High)$",
        description="Defines how aggressively the Red Team operates",
    )
    uncertain_pass_through: bool = Field(
        False,
        description="When True, UNCERTAIN verdicts pass through without soft-fail (High profile only)",
    )
    tone: Optional[str] = Field(
        None, description="Desired narrative tone (e.g., urgent, analytical, hopeful)"
    )
    angle: Optional[str] = Field(
        None, description="Specific editorial angle or framing"
    )


class AssetRenderMeta(BaseModel):
    """Schema for the JSONB render_meta column in Assets."""

    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    synthid_watermark: Optional[str] = Field(
        None, description="Google SynthID signature for compliance"
    )
    prompt_used: Optional[str] = Field(None, description="The exact prompt used")
    failure_reason: Optional[str] = Field(
        None, description="Reason for asset generation failure, if any"
    )


class FailedClaim(BaseModel):
    claim_text: str = Field(description="The exact claim that failed fact-checking")
    verdict: str = Field(description="UNSUPPORTED or CONTESTED")
    evidence_text: str = Field(description="Evidence found during evaluation")
    confidence: float = Field(
        description="Evaluator confidence 0.0-1.0", ge=0.0, le=1.0
    )


class AssembledContext(BaseModel):
    """Context Builder output: narrative summary + formatted evidence for agent injection."""

    narrative_summary: str = Field(
        description="refined_context verbatim — unchanged research narrative"
    )
    evidence_sections: str = Field(
        description="Formatted text block with retrieved chunks, scores, and source_type for prompt injection"
    )
    raw_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full payload of retrieved chunks for future structured use (Option B)",
    )


class OptimizerFeedbackEntry(BaseModel):
    feedback_type: str = Field(
        default="structured_claims", description="Discriminator for feedback format"
    )
    failed_claims: List[FailedClaim] = Field(description="Claims that need patching")
    overall_reasoning: str = Field(description="Evaluator's overall reasoning")
    revision_number: int = Field(
        description="Which revision cycle produced this feedback"
    )


# ==========================================
# 3. REQUEST SCHEMAS (Input via API)
# ==========================================
class JobCreateRequest(BaseModel):
    """Payload for Step 1: User inserts title & context."""

    title: str = Field(
        ..., min_length=3, max_length=200, example="BRICS De-dollarization 2025"
    )
    user_reference: str = Field(
        ...,
        min_length=1,
        description="User-provided background text as narrative foundation",
    )
    research_inputs: ResearchInputs
    story_directives: StoryDirectives = Field(default_factory=StoryDirectives)
    format_type: FormatTypeEnum = Field(
        FormatTypeEnum.ALL,
        description="Output format: all, video, blog, or carousel",
    )
    platform: PlatformEnum = Field(
        ...,
        description="Target platform: twitter, linkedin, instagram, youtube, tiktok",
    )
    device_id: Optional[str] = Field(
        None,
        description="Client device identifier for S3 key prefixing. Sent from localStorage.",
    )

    @model_validator(mode="after")
    def validate_format_for_platform(self):
        if self.format_type != FormatTypeEnum.ALL:
            valid = PLATFORM_FORMAT_MAP.get(self.platform, [])
            if self.format_type not in valid:
                raise ValueError(
                    f"Format '{self.format_type.value}' is not supported on '{self.platform.value}'. "
                    f"Valid formats: {[f.value for f in valid]}"
                )
        return self


class ScriptApprovalRequest(BaseModel):
    """Payload for Human-in-the-loop overrides."""

    is_approved: bool = Field(..., description="Approve or reject the script")
    human_feedback: Optional[str] = Field(
        None, description="Feedback to send back to the Script Agent"
    )


# ==========================================
# 4. RESPONSE SCHEMAS (Output via API)
# ==========================================
class FactCheckClaimResponse(BaseModel):
    """Outputs for Step 4 & 6: Red Team Evaluation Results."""

    id: UUID
    claim_text: str
    verdict: VerdictEnum
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_text: Optional[str] = Field(
        None, description="Evidence text supporting this verdict"
    )
    evidence_text_inline: List[str] = Field(
        default_factory=list,
        description="Snapshot of raw evidence chunk content for audit trail persistence",
    )
    hedge_required: bool = Field(
        False,
        description="True when verdict is UNCERTAIN — formatter should apply hedged language",
    )
    evidence_references: List[UUID] = Field(
        default_factory=list, description="IDs of ResearchChunks"
    )

    model_config = ConfigDict(from_attributes=True)


class ScriptResponse(BaseModel):
    """Outputs for Step 5: The Agentic Script."""

    id: UUID
    role: str = "master"
    version: int
    content: str
    is_approved: bool
    feedback_history: List[Union[str, Dict[str, Any]]]
    claims: List[FactCheckClaimResponse] = Field(default_factory=list)
    format_type: Optional[str] = None
    format_payload: Optional[FormatPayload] = None
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
    title: str
    status: JobStatusEnum
    user_reference: str = Field(..., description="User-provided reference text")
    source_urls: List[str] = Field(
        default_factory=list, description="Source URLs for Tavily extraction"
    )
    story_directives: Dict[str, Any] = Field(
        default_factory=dict, description="Editorial directives"
    )
    format_type: Optional[FormatTypeEnum] = FormatTypeEnum.ALL
    platform: Optional[PlatformEnum] = None
    final_video_url: Optional[str]
    refined_context: Optional[str] = None
    error_log: Optional[Dict[str, Any]]

    # We only expose the most recently active script to keep payloads light
    scripts: List[ScriptResponse] = Field(default_factory=list)
    assets: List[AssetResponse] = Field(default_factory=list)

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
