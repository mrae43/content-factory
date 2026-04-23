"use client";

import { useJobs } from "@/hooks/use-jobs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { JobStatusBadge } from "@/components/jobs/job-status-badge";
import Link from "next/link";

export default function HomePage() {
  const { data: jobs, isLoading } = useJobs();

  if (isLoading) {
    return <div className="text-muted-foreground">Loading...</div>;
  }

  const activeJobs = jobs?.filter(
    (j) => !["COMPLETED", "FAILED"].includes(j.status)
  );
  const completedJobs = jobs?.filter((j) => j.status === "COMPLETED");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <Link
          href="/jobs/new"
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
        >
          New Job
        </Link>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Total Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{jobs?.length || 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Active</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{activeJobs?.length || 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Completed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {completedJobs?.length || 0}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-2">
        <h3 className="text-lg font-semibold">Recent Jobs</h3>
        {jobs?.slice(0, 10).map((job) => (
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
      </div>
    </div>
  );
}
