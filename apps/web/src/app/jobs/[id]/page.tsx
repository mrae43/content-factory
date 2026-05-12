"use client";

import { useJobDetail, useApproveScript } from "@/hooks/use-jobs";
import { use, useState, type ChangeEvent } from "react";
import type {
  ScriptResponse,
  FactCheckClaimResponse,
  AssetResponse,
  BlogFormatPayload,
  CarouselFormatPayload,
  VideoFormatPayload,
} from "@content-factory/shared-types";
import { JobStatusBadge } from "@/components/jobs/job-status-badge";
import { FormatBadge } from "@/components/jobs/format-badge";
import { StateMachineProgress } from "@/components/jobs/state-machine-progress";
import { ClaimCard } from "@/components/script/claim-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { BlogViewer } from "@/components/viewers/blog-viewer";
import { CarouselViewer } from "@/components/viewers/carousel-viewer";
import { VideoScriptViewer } from "@/components/viewers/video-script-viewer";

function renderFormatViewer(
  formatType: string,
  payload: Record<string, unknown>
) {
  switch (formatType) {
    case "BLOG":
      return <BlogViewer payload={payload as BlogFormatPayload} />;
    case "CAROUSEL":
      return <CarouselViewer payload={payload as CarouselFormatPayload} />;
    case "VIDEO":
      return <VideoScriptViewer payload={payload as VideoFormatPayload} />;
    default:
      return (
        <pre className="whitespace-pre-wrap text-sm bg-muted p-4 rounded">
          {JSON.stringify(payload, null, 2)}
        </pre>
      );
  }
}

type FeedbackEntry =
  | string
  | {
      overall_reasoning?: string;
      failed_claims?: Array<{ verdict?: string; claim_text?: string }>;
      [key: string]: unknown;
    };

export default function JobDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: job, isLoading } = useJobDetail(id);
  const approvalMutation = useApproveScript(id);
  const [feedbackText, setFeedbackText] = useState("");
  const [showRejectForm, setShowRejectForm] = useState(false);
  const [showFullRawText, setShowFullRawText] = useState(false);

  if (isLoading || !job) {
    return <div className="text-muted-foreground">Loading job...</div>;
  }

  const scripts = job.scripts ?? [];
  const assets = job.assets ?? [];
  const allClaims = scripts.flatMap(
    (s: ScriptResponse) => s.claims ?? []
  );

  const masterScript = scripts.find(
    (s: ScriptResponse) => !s.format_payload
  );
  const formatScripts = scripts.filter(
    (s: ScriptResponse) => s.format_payload
  );

  const latestScript = masterScript ?? scripts[scripts.length - 1];

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
        <div className="flex items-center gap-2">
          <FormatBadge formatType={job.format_type} />
          <JobStatusBadge status={job.status} />
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Pipeline Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <StateMachineProgress
            currentStatus={job.status}
            formatType={job.format_type}
          />
        </CardContent>
      </Card>

      {job.status === "FAILED" && job.error_log && (
        <Card className="border-red-300 bg-red-50">
          <CardHeader>
            <CardTitle className="text-sm text-red-800">Error Log</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="whitespace-pre-wrap text-sm text-red-900">
              {JSON.stringify(job.error_log, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      {job.refined_context && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Research Summary (refined_context)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="whitespace-pre-wrap text-sm">
              {job.refined_context}
            </pre>
          </CardContent>
        </Card>
      )}

      {job.pre_context && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Input Context (pre_context)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div>
              <span className="font-medium">Source URLs:</span>{" "}
              {Array.isArray(job.pre_context.source_urls) &&
              job.pre_context.source_urls.length > 0
                ? (job.pre_context.source_urls as string[]).map(
                    (url, i) => (
                      <a
                        key={i}
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 underline mr-2"
                      >
                        {url}
                      </a>
                    )
                  )
                : "None provided"}
            </div>
            {typeof job.pre_context.raw_text === "string" && (
              <div>
                <span className="font-medium">Raw Text:</span>
                <pre className="mt-1 whitespace-pre-wrap text-xs bg-muted p-2 rounded">
                  {showFullRawText
                    ? job.pre_context.raw_text
                    : job.pre_context.raw_text.slice(0, 500) +
                      (job.pre_context.raw_text.length > 500
                        ? "..."
                        : "")}
                </pre>
                {job.pre_context.raw_text.length > 500 && (
                  <Button
                    variant="link"
                    className="p-0 h-auto text-xs"
                    onClick={() =>
                      setShowFullRawText(!showFullRawText)
                    }
                  >
                    {showFullRawText ? "Show less" : "Show more"}
                  </Button>
                )}
              </div>
            )}
            <div>
              <span className="font-medium">Target Audience:</span>{" "}
              {typeof job.pre_context.target_audience === "string"
                ? job.pre_context.target_audience
                : "General"}
            </div>
            <div>
              <span className="font-medium">Guardrail Strictness:</span>{" "}
              {typeof job.pre_context.guardrail_strictness === "string"
                ? job.pre_context.guardrail_strictness
                : "High"}
            </div>
            {job.platform && (
              <div>
                <span className="font-medium">Platform:</span>{" "}
                {job.platform}
              </div>
            )}
            <div>
              <span className="font-medium">Strict Compliance:</span>{" "}
              <Badge
                variant={
                  job.strict_compliance_mode ? "default" : "secondary"
                }
              >
                {job.strict_compliance_mode ? "Enabled" : "Disabled"}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {latestScript && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Script (v{latestScript.version})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="whitespace-pre-wrap text-sm">
              {latestScript.content}
            </pre>
            {latestScript.feedback_history.length > 0 && (
              <div className="mt-4 space-y-2 border-t pt-3">
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Feedback History (
                  {latestScript.feedback_history.length} revision
                  {latestScript.feedback_history.length !== 1
                    ? "s"
                    : ""}
                  )
                </h4>
                {latestScript.feedback_history.map(
                  (entry: FeedbackEntry, i: number) => {
                    if (typeof entry === "string") {
                      return (
                        <div
                          key={i}
                          className="text-xs text-muted-foreground bg-muted p-2 rounded"
                        >
                          <span className="font-medium">
                            Feedback #{i + 1}:
                          </span>{" "}
                          {entry}
                        </div>
                      );
                    }
                    return (
                      <div
                        key={i}
                        className="text-xs bg-muted p-2 rounded space-y-1"
                      >
                        <span className="font-medium">
                          Revision #{i + 1}
                        </span>
                        {typeof entry.overall_reasoning ===
                          "string" && (
                          <p className="text-muted-foreground">
                            {entry.overall_reasoning}
                          </p>
                        )}
                        {Array.isArray(entry.failed_claims) && (
                          <ul className="list-disc pl-4 space-y-0.5">
                            {entry.failed_claims.map((fc, j) => (
                              <li key={j}>
                                <Badge
                                  className="text-[10px] px-1 py-0"
                                  variant={
                                    typeof fc.verdict === "string" &&
                                    fc.verdict === "UNSUPPORTED"
                                      ? "destructive"
                                      : "secondary"
                                  }
                                >
                                  {typeof fc.verdict === "string"
                                    ? fc.verdict
                                    : "UNKNOWN"}
                                </Badge>{" "}
                                {typeof fc.claim_text === "string"
                                  ? fc.claim_text
                                  : ""}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    );
                  }
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {formatScripts.length === 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              {formatScripts[0].format_type
                ? formatScripts[0].format_type!.charAt(0).toUpperCase() +
                  formatScripts[0].format_type!.slice(1).toLowerCase() +
                  " Output"
                : "Format Output"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {renderFormatViewer(
              formatScripts[0].format_type ?? "",
              formatScripts[0].format_payload as Record<string, unknown>
            )}
          </CardContent>
        </Card>
      )}

      {formatScripts.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Format Outputs</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue={formatScripts[0].format_type ?? `fmt-${formatScripts[0].id}`}>
              <TabsList>
                {formatScripts.map((s: ScriptResponse) => (
                  <TabsTrigger
                    key={s.id}
                    value={s.format_type ?? `fmt-${s.id}`}
                  >
                    {s.format_type
                      ? s.format_type.charAt(0).toUpperCase() +
                        s.format_type.slice(1).toLowerCase()
                      : "Unknown"}
                  </TabsTrigger>
                ))}
              </TabsList>
              {formatScripts.map((s: ScriptResponse) => (
                <TabsContent
                  key={s.id}
                  value={s.format_type ?? `fmt-${s.id}`}
                >
                  {renderFormatViewer(
                    s.format_type ?? "",
                    s.format_payload as Record<string, unknown>
                  )}
                </TabsContent>
              ))}
            </Tabs>
          </CardContent>
        </Card>
      )}

      {allClaims.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold">
            Fact Check Claims ({allClaims.length})
          </h3>
          {allClaims.map((claim: FactCheckClaimResponse) => (
            <ClaimCard
              key={claim.id}
              claim_text={claim.claim_text}
              verdict={claim.verdict}
              confidence={claim.confidence}
              evidence_references={claim.evidence_references ?? []}
            />
          ))}
        </div>
      )}

      {job.status === "HUMAN_REVIEW_NEEDED" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Script Review</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Review the script and claims above, then approve or request
              a revision.
            </p>
            {approvalMutation.isError && (
              <p className="text-sm text-red-600">
                Error:{" "}
                {(approvalMutation.error as Error)?.message ||
                  "Failed to submit"}
              </p>
            )}
            {showRejectForm ? (
              <div className="space-y-3">
                <Textarea
                  placeholder="Optional feedback for the script agent..."
                  value={feedbackText}
                  onChange={(
                    e: ChangeEvent<HTMLTextAreaElement>
                  ) => setFeedbackText(e.target.value)}
                />
                <div className="flex gap-2">
                  <Button
                    variant="destructive"
                    disabled={approvalMutation.isPending}
                    onClick={() => {
                      approvalMutation.mutate(
                        {
                          isApproved: false,
                          feedback: feedbackText || undefined,
                        },
                        {
                          onSettled: () => {
                            setShowRejectForm(false);
                            setFeedbackText("");
                          },
                        }
                      );
                    }}
                  >
                    {approvalMutation.isPending
                      ? "Submitting..."
                      : "Confirm Revision"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setShowRejectForm(false)}
                    disabled={approvalMutation.isPending}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex gap-2">
                <Button
                  disabled={approvalMutation.isPending}
                  onClick={() =>
                    approvalMutation.mutate({ isApproved: true })
                  }
                >
                  {approvalMutation.isPending
                    ? "Approving..."
                    : "Approve Script"}
                </Button>
                <Button
                  variant="destructive"
                  disabled={approvalMutation.isPending}
                  onClick={() => setShowRejectForm(true)}
                >
                  Request Revision
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {assets.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Assets ({assets.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 md:grid-cols-2">
              {assets.map((asset: AssetResponse) => (
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
