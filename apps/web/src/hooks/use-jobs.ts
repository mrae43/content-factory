import { useQuery, useMutation, useQueryClient, type Query } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import type {
  RenderJobResponse,
  JobCreateRequest,
} from "@content-factory/shared-types";

export type { RenderJobResponse, JobCreateRequest } from "@content-factory/shared-types";

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
    refetchInterval: (query: Query<RenderJobResponse, Error>) => {
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
    mutationFn: (data: JobCreateRequest) =>
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
