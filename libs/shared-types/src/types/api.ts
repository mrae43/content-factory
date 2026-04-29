export type JobStatusEnum =
  | "PENDING"
  | "RESEARCHING"
  | "FACT_CHECKING_RESEARCH"
  | "SCRIPTING"
  | "FACT_CHECKING_SCRIPT"
  | "ASSET_GENERATION"
  | "COMPLETED"
  | "FAILED"
  | "HUMAN_REVIEW_NEEDED";

export type AssetTypeEnum =
  | "VISUAL_VEO"
  | "AUDIO_LYRIA"
  | "VOICEOVER"
  | "SUBTITLE_JSON"
  | "DATA_CHART";

export type VerdictEnum =
  | "SUPPORTED"
  | "CONTESTED"
  | "UNSUPPORTED"
  | "UNCERTAIN";

export interface PreContextPayload {
  source_urls: string[];
  raw_text?: string | null;
  target_audience: string;
  guardrail_strictness: string;
}

export interface AssetRenderMeta {
  start_time_sec?: number | null;
  end_time_sec?: number | null;
  synthid_watermark?: string | null;
  prompt_used?: string | null;
}

export interface FailedClaim {
  claim_text: string;
  verdict: string;
  evidence_text: string;
  confidence: number;
}

export interface OptimizerFeedbackEntry {
  feedback_type: string;
  failed_claims: FailedClaim[];
  overall_reasoning: string;
  revision_number: number;
}

export interface JobCreateRequest {
  topic: string;
  pre_context: PreContextPayload;
  strict_compliance_mode?: boolean;
}

export interface ScriptApprovalRequest {
  is_approved: boolean;
  human_feedback?: string | null;
}

export interface FactCheckClaimResponse {
  id: string;
  claim_text: string;
  verdict: VerdictEnum;
  confidence: number;
  evidence_references: string[];
}

export interface ScriptResponse {
  id: string;
  version: number;
  content: string;
  is_approved: boolean;
  feedback_history: Array<string | Record<string, unknown>>;
  claims: FactCheckClaimResponse[];
  created_at: string;
  updated_at: string;
}

export interface AssetResponse {
  id: string;
  asset_type: AssetTypeEnum;
  url_or_path: string;
  render_meta: AssetRenderMeta;
  created_at: string;
}

export interface RenderJobResponse {
  id: string;
  topic: string;
  status: JobStatusEnum;
  strict_compliance_mode: boolean;
  final_video_url: string | null;
  refined_context: string | null;
  error_log: Record<string, unknown> | null;
  scripts: ScriptResponse[];
  assets: AssetResponse[];
  created_at: string;
  updated_at: string;
}
