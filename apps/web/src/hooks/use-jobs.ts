import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";

type JobStatusEnum =
  | "PENDING"
  | "RESEARCHING"
  | "FACT_CHECKING_RESEARCH"
  | "SCRIPTING"
  | "FACT_CHECKING_SCRIPT"
  | "ASSET_GENERATION"
  | "COMPLETED"
  | "FAILED"
  | "HUMAN_REVIEW_NEEDED";

type VerdictEnum = "SUPPORTED" | "CONTESTED" | "UNSUPPORTED" | "UNCERTAIN";

type AssetTypeEnum =
  | "VISUAL_VEO"
  | "AUDIO_LYRIA"
  | "VOICEOVER"
  | "SUBTITLE_JSON"
  | "DATA_CHART";

interface AssetRenderMeta {
  start_time_sec?: number | null;
  end_time_sec?: number | null;
  synthid_watermark?: string | null;
  prompt_used?: string | null;
}

interface FactCheckClaimResponse {
  id: string;
  claim_text: string;
  verdict: VerdictEnum;
  confidence: number;
  evidence_references: string[];
}

interface ScriptResponse {
  id: string;
  version: number;
  content: string;
  is_approved: boolean;
  feedback_history: Array<string | Record<string, unknown>>;
  claims: FactCheckClaimResponse[];
  created_at: string;
  updated_at: string;
}

interface AssetResponse {
  id: string;
  asset_type: AssetTypeEnum;
  url_or_path: string;
  render_meta: AssetRenderMeta;
  created_at: string;
}

interface RenderJobResponse {
  id: string;
  topic: string;
  status: JobStatusEnum;
  strict_compliance_mode: boolean;
  final_video_url: string | null;
  refined_context: string | null;
  error_log: Record<string, unknown> | null;
  pre_context: Record<string, unknown> | null;
  scripts: ScriptResponse[];
  assets: AssetResponse[];
  created_at: string;
  updated_at: string;
}

interface PreContextPayload {
  source_urls: string[];
  raw_text?: string | null;
  target_audience: string;
  guardrail_strictness: string;
}

interface CreateJobRequest {
  topic: string;
  pre_context: PreContextPayload;
  strict_compliance_mode?: boolean;
}

export function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => apiClient<RenderJobResponse[]>("/api/v1/jobs/"),
    refetchInterval: 5000,
  });
}

export function useJobDetail(jobId: string) {
  return useQuery({
    queryKey: ["jobs", jobId],
    queryFn: () => apiClient<RenderJobResponse>(`/api/v1/jobs/${jobId}`),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "COMPLETED" || status === "HUMAN_REVIEW_NEEDED"
        ? false
        : 3000;
    },
    enabled: !!jobId,
  });
}

export function useCreateJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateJobRequest) =>
      apiClient<RenderJobResponse>("/api/v1/jobs/", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useApproveScript(jobId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (vars: { isApproved: boolean; feedback?: string }) =>
      apiClient(`/api/v1/jobs/${jobId}/approve-script`, {
        method: "POST",
        body: JSON.stringify({
          is_approved: vars.isApproved,
          human_feedback: vars.feedback,
        }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs", jobId] });
    },
  });
}
