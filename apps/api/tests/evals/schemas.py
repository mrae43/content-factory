"""
Pydantic models for the Golden Dataset and eval framework.

Mirrors the schema defined in GOLDEN_DATASET_FOUNDATION.md §4.
Enums align with app/schemas/shorts.py and app/workers/agents.py.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict


# ==========================================
# 1. SHARED ENUMS
# ==========================================


class TraceType(str, Enum):
    HAPPY_PATH = "happy_path"
    REVISION_LOOP = "revision_loop"
    FALLBACK_CHAIN = "fallback_chain"
    NEGATIVE_GOLDEN = "negative_golden"


class CaseCategory(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    HALLUCINATION_TRAP = "hallucination_trap"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    EDGE_CASE_MINIMAL = "edge_case_minimal"
    EDGE_CASE_LONG = "edge_case_long"
    SAFETY_REFUSAL = "safety_refusal"
    PII_PROTECTION = "pii_protection"


class DifficultyTier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class EvalVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTESTED = "CONTESTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTAIN = "UNCERTAIN"


class AgentActionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    REVISION_NEEDED = "REVISION_NEEDED"
    ESCALATE = "ESCALATE"
    ERROR = "ERROR"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RESEARCHING = "RESEARCHING"
    FACT_CHECKING_RESEARCH = "FACT_CHECKING_RESEARCH"
    SCRIPTING = "SCRIPTING"
    FACT_CHECKING_SCRIPT = "FACT_CHECKING_SCRIPT"
    ASSET_GENERATION = "ASSET_GENERATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HUMAN_REVIEW_NEEDED = "HUMAN_REVIEW_NEEDED"


class SafetyConstraintType(str, Enum):
    PII_MUST_NOT_PROPAGATE = "PII_MUST_NOT_PROPAGATE"
    MEDICAL_DISCLAIMER_REQUIRED = "MEDICAL_DISCLAIMER_REQUIRED"
    REFUSE_HARMFUL_CONTENT = "REFUSE_HARMFUL_CONTENT"
    DEFAMATION_PREVENTION = "DEFAMATION_PREVENTION"


class CheckStage(str, Enum):
    CHUNKING = "chunking"
    RESEARCH = "research"
    SCRIPT = "script"
    FACT_CHECK = "fact_check"


class RubricSet(str, Enum):
    RESEARCH = "research"
    SCRIPT = "script"
    FACT_CHECK = "fact_check"
    OPTIMIZER = "optimizer"


# ==========================================
# 2. INPUT SCHEMAS
# ==========================================


class PreContextInput(BaseModel):
    raw_text: Optional[str] = Field(
        None, description="Raw copied text or book excerpts"
    )
    source_urls: List[str] = Field(default_factory=list, description="URLs to scrape")
    target_audience: str = Field(
        "General", description="e.g. Academics, TikTok, Investors"
    )
    guardrail_strictness: str = Field(
        "High", description="How aggressively the Red Team operates"
    )


class CaseInput(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    pre_context: PreContextInput
    strict_compliance_mode: bool = Field(True)


# ==========================================
# 3. TRACE SPEC SCHEMAS
# ==========================================


class AgentCallSpec(BaseModel):
    state: str = Field(
        ..., description="Pipeline state, e.g. 'SCRIPTING:2' for revision cycle"
    )
    agent: str = Field(..., description="Agent class name, e.g. 'ResearchAgent'")
    model: Optional[str] = Field(
        None, description="Expected LLM model, e.g. 'gemini-1.5-pro'"
    )
    reason: Optional[str] = Field(
        None, description="Why this agent is called, e.g. 'first draft'"
    )
    expected_verdict: Optional[AgentActionStatus] = Field(
        None, description="Expected AgentResult.status for this call"
    )


class ToolCallSpec(BaseModel):
    agent: str = Field(..., description="Agent that invokes the tool")
    tool: str = Field(..., description="Tool/function name, e.g. 'semantic_search'")
    args: Dict[str, Any] = Field(default_factory=dict, description="Expected arguments")
    per_claim: bool = Field(
        False,
        description="True if this tool is called once per extracted claim (Red Team)",
    )


class TraceStep(BaseModel):
    step: int
    agent: Optional[str] = None
    stage: Optional[str] = None
    tool_call: Optional[str] = None
    expected_result: Optional[str] = None
    fallback: Optional[str] = None
    assertion: Optional[str] = None


class FeedbackHistorySpec(BaseModel):
    feedback_type: str = Field(
        "structured_claims", description="Expected feedback_type value"
    )
    expected_failed_count: int = Field(
        ..., ge=0, description="How many claims should fail (0 for string feedback)"
    )
    expected_verdicts: List[EvalVerdict] = Field(
        default_factory=list, description="Expected verdicts for failed claims"
    )


class TraceSpec(BaseModel):
    expected_state_sequence: List[JobStatus]
    expected_agent_calls: List[AgentCallSpec] = Field(default_factory=list)
    expected_tool_calls: List[ToolCallSpec] = Field(default_factory=list)
    expected_feedback_history: List[FeedbackHistorySpec] = Field(default_factory=list)
    fallback_expectations: List[TraceStep] = Field(default_factory=list)
    rejection_expectations: List[TraceStep] = Field(default_factory=list)
    note: Optional[str] = None


# ==========================================
# 4. EXPECTED OUTCOMES SCHEMAS
# ==========================================


class ResearchOutcome(BaseModel):
    must_include_facts: List[str] = Field(
        default_factory=list, description="Key facts the research must surface"
    )
    must_avoid: List[str] = Field(
        default_factory=list, description="Patterns the research must not produce"
    )
    min_chunks: int = Field(2, description="Minimum number of research chunks")
    min_confidence: float = Field(0.7, ge=0.0, le=1.0)
    refined_context_word_range: Tuple[int, int] = Field(
        (800, 1500), description="Min/max word count for refined_context"
    )


class ScriptOutcome(BaseModel):
    must_include_topics: List[str] = Field(default_factory=list)
    must_avoid: List[str] = Field(default_factory=list)
    scene_count_range: Tuple[int, int] = Field(
        (3, 8), description="Min/max storyboard scenes"
    )
    word_count_range: Tuple[int, int] = Field(
        (150, 500), description="Min/max script word count"
    )
    must_have_hook: bool = Field(True)
    must_have_loop: bool = Field(True)
    storyboard_fields: List[str] = Field(
        default_factory=lambda: ["visual_prompt", "audio_cue"],
        description="Required fields in each storyboard item",
    )


class ClaimVerdict(BaseModel):
    claim_text: str
    expected_verdict: EvalVerdict


class FactCheckOutcome(BaseModel):
    expected_overall_verdict: str = Field(
        "SUPPORTED",
        description="Expected overall verdict: SUPPORTED or REVISION_NEEDED",
    )
    max_unsupported_claims: int = Field(0, ge=0)
    claims_with_known_verdicts: List[ClaimVerdict] = Field(default_factory=list)
    min_claim_count: int = Field(1, ge=1, description="Minimum extracted claims")
    evidence_must_have_references: bool = Field(
        True, description="Every claim must reference ResearchChunk IDs"
    )


class OptimizationOutcome(BaseModel):
    must_preserve_claims: List[str] = Field(
        default_factory=list,
        description="Claim texts that must survive patching unchanged",
    )
    must_patch_claims: List[str] = Field(
        default_factory=list, description="Claim texts that must be modified or removed"
    )
    patch_must_be_surgical: bool = Field(
        True, description="Only failed claims changed, rest preserved verbatim"
    )
    narrative_must_flow: bool = Field(
        True, description="Script must read coherently after patching"
    )


class NegativeAssertion(BaseModel):
    stage: CheckStage
    field: str = Field(..., description="Output field to check, e.g. 'refined_context'")
    must_not_contain: List[str] = Field(
        default_factory=list, description="Strings that must not appear in the output"
    )


class ResearchReference(BaseModel):
    refined_context: str = Field("", description="Pre-recorded research output")
    chunks: List[str] = Field(
        default_factory=list, description="Pre-recorded research chunks"
    )


class ScriptReference(BaseModel):
    script_content: str = Field("", description="Pre-recorded script output")
    storyboard: List[Dict[str, Any]] = Field(
        default_factory=list, description="Pre-recorded storyboard"
    )


class FactCheckReference(BaseModel):
    verdict: str = Field("SUPPORTED", description="Pre-recorded overall verdict")
    claims: List[Dict[str, Any]] = Field(
        default_factory=list, description="Pre-recorded claim items"
    )
    overall_reasoning: str = Field("", description="Pre-recorded overall reasoning")


class OptimizationReference(BaseModel):
    script_content: str = Field("", description="Pre-recorded patched script")
    storyboard: List[Dict[str, Any]] = Field(
        default_factory=list, description="Pre-recorded patched storyboard"
    )
    patch_summary: str = Field("", description="Pre-recorded patch summary")


class ReferenceOutputs(BaseModel):
    research: Optional[ResearchReference] = None
    script: Optional[ScriptReference] = None
    fact_check: Optional[FactCheckReference] = None
    optimization: Optional[OptimizationReference] = None


class ExpectedOutcomes(BaseModel):
    research: Optional[ResearchOutcome] = None
    script: Optional[ScriptOutcome] = None
    fact_check: Optional[FactCheckOutcome] = None
    optimization: Optional[OptimizationOutcome] = None
    final_status: str = Field(
        ...,
        description="Expected terminal status, e.g. 'FACT_CHECKING_SCRIPT (approved)'",
    )
    assertions: List[NegativeAssertion] = Field(
        default_factory=list,
        description="Stage-level negative assertions (for negative goldens)",
    )


# ==========================================
# 5. SAFETY CONSTRAINT (Negative Goldens)
# ==========================================


class SafetyConstraint(BaseModel):
    type: SafetyConstraintType
    pii_entities: List[str] = Field(
        default_factory=list, description="PII strings to check for propagation"
    )
    check_stages: List[CheckStage] = Field(
        default_factory=list, description="Pipeline stages to check"
    )
    pass_criteria: str = Field(..., description="Human-readable pass condition")


# ==========================================
# 6. SCORING SCHEMA
# ==========================================


class ScoringSpec(BaseModel):
    rubric_set: RubricSet = Field(..., description="Which rubric to apply")
    pass_threshold: float = Field(0.75, ge=0.0, le=1.0)
    dimension_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension minimums, e.g. {'accuracy': 0.8}",
    )


# ==========================================
# 7. CASE METADATA
# ==========================================


class CaseMetadata(BaseModel):
    created_by: str = Field("system")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


# ==========================================
# 8. TOP-LEVEL GOLDEN CASE
# ==========================================


class GoldenCase(BaseModel):
    id: str = Field(
        ..., pattern=r"^[HRFENM]-\d{3}$", description="Case ID, e.g. 'H-001'"
    )
    trace_type: TraceType
    category: CaseCategory
    domain: str = Field(..., description="Knowledge domain, e.g. 'economics'")
    difficulty: DifficultyTier

    input: CaseInput
    trace_spec: TraceSpec
    expected_outcomes: ExpectedOutcomes
    reference_outputs: Optional[ReferenceOutputs] = Field(
        None, description="Pre-recorded agent outputs for deterministic eval"
    )
    scoring: Optional[ScoringSpec] = None

    safety_constraint: Optional[SafetyConstraint] = Field(
        None, description="Only set for negative_golden trace_type"
    )

    metadata: CaseMetadata = Field(default_factory=CaseMetadata)


# ==========================================
# 9. DATASET CONTAINER
# ==========================================


class GoldenDataset(BaseModel):
    cases: List[GoldenCase] = Field(..., description="All golden cases in the dataset")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cases": [
                    {
                        "id": "H-001",
                        "trace_type": "happy_path",
                        "category": "factual_accuracy",
                        "domain": "economics",
                        "difficulty": "easy",
                    }
                ]
            }
        }
    )
