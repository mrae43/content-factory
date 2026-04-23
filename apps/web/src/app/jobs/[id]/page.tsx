"use client";

import { useJobDetail } from "@/hooks/use-jobs";
import { use } from "react";
import { JobStatusBadge } from "@/components/jobs/job-status-badge";
import { StateMachineProgress } from "@/components/jobs/state-machine-progress";
import { ClaimCard } from "@/components/script/claim-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function JobDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: job, isLoading } = useJobDetail(id);

  if (isLoading || !job) {
    return <div className="text-muted-foreground">Loading job...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">{job.topic}</h2>
          <p className="text-sm text-muted-foreground">
            Job ID: {job.id} &middot; Created:{" "}
            {new Date(job.created_at).toLocaleString()}
          </p>
        </div>
        <JobStatusBadge status={job.status} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Pipeline Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <StateMachineProgress currentStatus={job.status} />
        </CardContent>
      </Card>

      {job.scripts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Script (v{job.scripts[job.scripts.length - 1].version})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="whitespace-pre-wrap text-sm">
              {job.scripts[job.scripts.length - 1].content}
            </pre>
          </CardContent>
        </Card>
      )}

      {job.claims.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold">
            Fact Check Claims ({job.claims.length})
          </h3>
          {job.claims.map((claim) => (
            <ClaimCard
              key={claim.id}
              claim_text={claim.claim_text}
              verdict={claim.verdict}
              evidence={claim.evidence}
              search_query={claim.search_query}
            />
          ))}
        </div>
      )}

      {job.assets.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Assets ({job.assets.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 md:grid-cols-2">
              {job.assets.map((asset) => (
                <div key={asset.id} className="rounded-lg border p-3">
                  <Badge className="mb-2">{asset.asset_type}</Badge>
                  <p className="text-xs text-muted-foreground">{asset.prompt}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
