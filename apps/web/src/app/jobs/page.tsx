"use client";

import { useJobs } from "@/hooks/use-jobs";
import { JobStatusBadge } from "@/components/jobs/job-status-badge";
import { FormatBadge } from "@/components/jobs/format-badge";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { JobCardSkeleton } from "@/components/jobs/job-card-skeleton";
import { useUIStore } from "@/stores/ui-store";
import Link from "next/link";

const statusFilters = [
  "all",
  "PENDING",
  "RESEARCHING",
  "FACT_CHECKING_RESEARCH",
  "SCRIPTING",
  "FACT_CHECKING_SCRIPT",
  "FORMATTING",
  "ASSET_GENERATION",
  "COMPLETED",
  "HUMAN_REVIEW_NEEDED",
  "FAILED",
];

export default function JobsPage() {
  const { data: jobs, isLoading, isError, error } = useJobs();
  const { selectedJobFilter, setJobFilter } = useUIStore();

  const filtered =
    selectedJobFilter === "all"
      ? jobs
      : jobs?.filter((j) => j.status === selectedJobFilter);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Jobs</h2>
        <Link
          href="/jobs/new"
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
        >
          New Job
        </Link>
      </div>

      <div
        className="flex gap-2 flex-wrap"
        role="group"
        aria-label="Filter by status"
      >
        {statusFilters.map((status) => (
          <button
            key={status}
            onClick={() => setJobFilter(status)}
            aria-pressed={selectedJobFilter === status}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              selectedJobFilter === status
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-accent"
            }`}
          >
            {status.replace(/_/g, " ")}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <JobCardSkeleton key={i} />
          ))}
        </div>
      ) : isError ? (
        <p className="text-sm text-red-600">
          Failed to load jobs: {error?.message}
        </p>
      ) : (
        <div className="space-y-2">
          {filtered?.map((job) => (
            <Link key={job.id} href={`/jobs/${job.id}`}>
              <Card className="mb-2 hover:bg-accent transition-colors">
                <CardContent className="flex items-center justify-between p-4">
                  <div className="flex items-center gap-3">
                    <div>
                      <p className="font-medium">{job.topic}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(job.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <FormatBadge formatType={job.format_type} />
                  </div>
                  <JobStatusBadge status={job.status} />
                </CardContent>
              </Card>
            </Link>
          ))}
          {filtered?.length === 0 && (
            <div className="text-center py-8">
              <p className="text-sm text-muted-foreground">
                {selectedJobFilter === "all"
                  ? "No jobs found."
                  : `No ${selectedJobFilter.replace(/_/g, " ")} jobs found.`}
              </p>
              {selectedJobFilter !== "all" && (
                <Button
                  variant="link"
                  onClick={() => setJobFilter("all")}
                >
                  Clear filter
                </Button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
