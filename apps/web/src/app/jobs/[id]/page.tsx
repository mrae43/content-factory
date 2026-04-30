"use client";

import { useJobDetail, useApproveScript } from "@/hooks/use-jobs";
import { use, useState } from "react";
import { JobStatusBadge } from "@/components/jobs/job-status-badge";
import { StateMachineProgress } from "@/components/jobs/state-machine-progress";
import { ClaimCard } from "@/components/script/claim-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

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

  const allClaims = job.scripts.flatMap((s) => s.claims);

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

      {allClaims.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold">
            Fact Check Claims ({allClaims.length})
          </h3>
          {allClaims.map((claim) => (
            <ClaimCard
              key={claim.id}
              claim_text={claim.claim_text}
              verdict={claim.verdict}
              confidence={claim.confidence}
              evidence_references={claim.evidence_references}
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
                  <p className="text-xs text-muted-foreground break-all">
                    {asset.url_or_path}
                  </p>
                  {asset.render_meta?.prompt_used && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {asset.render_meta.prompt_used}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
