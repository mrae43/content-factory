import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";

interface RenderJobResponse {
  id: string;
  status: string;
  topic: string;
  raw_text: string;
  created_at: string;
  updated_at: string;
  scripts: ScriptResponse[];
  claims: FactCheckClaimResponse[];
  assets: AssetResponse[];
}

interface ScriptResponse {
  id: string;
  content: string;
  storyboard: string;
  version: number;
  created_at: string;
}

interface FactCheckClaimResponse {
  id: string;
  claim_text: string;
  verdict: string;
  evidence: string;
  search_query: string;
}

interface AssetResponse {
  id: string;
  asset_type: string;
  url: string;
  prompt: string;
}

interface CreateJobRequest {
  topic: string;
  raw_text: string;
  platform_constraints?: Record<string, unknown>;
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
    mutationFn: (scriptId: string) =>
      apiClient(`/api/v1/jobs/${jobId}/approve-script/${scriptId}`, {
        method: "POST",
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs", jobId] });
    },
  });
}
