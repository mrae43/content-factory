"use client";

import { useJobs } from "@/hooks/use-jobs";
import { JobStatusBadge } from "@/components/jobs/job-status-badge";
import { Card, CardContent } from "@/components/ui/card";
import { useUIStore } from "@/stores/ui-store";
import Link from "next/link";

const statusFilters = [
  "all",
  "PENDING",
  "RESEARCHING",
  "SCRIPTING",
  "FACT_CHECKING_SCRIPT",
  "ASSET_GENERATION",
  "COMPLETED",
  "HUMAN_REVIEW_NEEDED",
];

export default function JobsPage() {
  const { data: jobs, isLoading } = useJobs();
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

      <div className="flex gap-2 flex-wrap">
        {statusFilters.map((status) => (
          <button
            key={status}
            onClick={() => setJobFilter(status)}
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
        <div className="text-muted-foreground">Loading...</div>
      ) : (
        <div className="space-y-2">
          {filtered?.map((job) => (
            <Link key={job.id} href={`/jobs/${job.id}`}>
              <Card className="mb-2 hover:bg-accent transition-colors">
                <CardContent className="flex items-center justify-between p-4">
                  <div>
                    <p className="font-medium">{job.topic}</p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(job.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <JobStatusBadge status={job.status} />
                </CardContent>
              </Card>
            </Link>
          ))}
          {filtered?.length === 0 && (
            <p className="text-muted-foreground text-sm">No jobs found.</p>
          )}
        </div>
      )}
    </div>
  );
}
