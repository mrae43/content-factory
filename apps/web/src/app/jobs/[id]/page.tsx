"use client";

import { useJobDetail, useApproveScript } from "@/hooks/use-jobs";
import { use, useState, type ChangeEvent } from "react";
import { toast } from "sonner";
import type {
  RenderJobResponse,
  ScriptResponse,
  FactCheckClaimResponse,
  AssetResponse,
  BlogFormatPayload,
  CarouselFormatPayload,
  VideoFormatPayload,
  PlatformEnum,
} from "@content-factory/shared-types";
import { FormatBadge } from "@/components/jobs/format-badge";
import { StatusDot } from "@/components/jobs/status-dot";
import { EditorialTimeline } from "@/components/editorial/editorial-timeline";
import { CollapsibleSection } from "@/components/editorial/collapsible-section";
import { BlogViewer } from "@/components/viewers/blog-viewer";
import { CarouselViewer } from "@/components/viewers/carousel-viewer";
import { VideoScriptViewer } from "@/components/viewers/video-script-viewer";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";

function renderFormatViewer(
  formatType: string,
  payload: Record<string, unknown>,
  platform?: string | null
) {
  switch (formatType) {
    case "BLOG":
      return <BlogViewer payload={payload as BlogFormatPayload} />;
    case "CAROUSEL":
      return (
        <CarouselViewer
          payload={payload as CarouselFormatPayload}
          platform={platform as PlatformEnum | null}
        />
      );
    case "VIDEO":
      return <VideoScriptViewer payload={payload as VideoFormatPayload} />;
    default:
      return (
        <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-muted-foreground bg-muted p-4 rounded-md">
          {JSON.stringify(payload, null, 2)}
        </pre>
      );
  }
}

function SectionHeader({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3">
      <h3 className="font-heading text-lg font-semibold tracking-tight text-foreground">
        {label}
      </h3>
      <div className="flex-1 h-px bg-border" />
    </div>
  );
}

function SkeletonBlock({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className="h-4 w-full" />
      ))}
    </div>
  );
}

export default function JobDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: job, isLoading } = useJobDetail(id);

  if (isLoading || !job) {
    return <LoadingSkeleton />;
  }

  return <JobDetailContent job={job} jobId={id} />;
}

function LoadingSkeleton() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      <div className="space-y-3">
        <Skeleton className="h-9 w-full max-w-96" />
        <Skeleton className="h-4 w-64" />
        <div className="flex gap-2">
          <Skeleton className="h-5 w-12 rounded-[4px]" />
          <Skeleton className="h-5 w-20" />
        </div>
      </div>
      <div className="space-y-2">
        <Skeleton className="h-5 w-48" />
        <div className="rounded-lg border border-border p-6">
          <SkeletonBlock lines={5} />
        </div>
      </div>
      <div className="space-y-2">
        <Skeleton className="h-5 w-48" />
        <div className="rounded-lg border border-border p-6 space-y-4">
          <div className="flex gap-4">
            <Skeleton className="h-3 w-3 rounded-full" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-5 w-32" />
              <SkeletonBlock lines={2} />
            </div>
          </div>
          <div className="flex gap-4">
            <Skeleton className="h-3 w-3 rounded-full" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-5 w-32" />
              <SkeletonBlock lines={2} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function JobDetailContent({
  job,
  jobId,
}: {
  job: RenderJobResponse;
  jobId: string;
}) {
  const approvalMutation = useApproveScript(jobId);
  const [feedbackText, setFeedbackText] = useState("");
  const [showRejectForm, setShowRejectForm] = useState(false);

  const scripts = job.scripts ?? [];
  //const assets = job.assets ?? [];
  const isTerminal: boolean =
    job.status === "COMPLETED" ||
    job.status === "FAILED" ||
    job.status === "HUMAN_REVIEW_NEEDED";
  const isActive = !isTerminal;

  const formatScripts = scripts.filter(
    (s: ScriptResponse) => s.role === "format"
  );
  const allClaims = scripts.flatMap(
    (s: ScriptResponse) => s.claims ?? []
  );

  function renderSection1() {
    if (job.status === "FAILED") {
      return <FailedSection job={job} />;
    }
    if (job.status === "COMPLETED") {
      return (
        <div className="space-y-4">
          <FormatTabs formatScripts={formatScripts} platform={job.platform} />
          {allClaims.length > 0 && (
            <div className="rounded-lg border border-border p-5 space-y-3">
              <h4 className="font-heading text-sm font-semibold text-muted-foreground">
                Fact Check Audit
              </h4>
              <ClaimsSection claims={allClaims} />
            </div>
          )}
        </div>
      );
    }
    return <ActiveOutput job={job} />;
  }

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
      <div className="space-y-2">
        <h1 className="font-heading text-3xl font-bold tracking-tight">
          {job.topic}
        </h1>
        <p className="text-sm text-muted-foreground">
          Job{" "}
          <span className="font-mono text-xs">
            #{job.id.slice(0, 6)}
          </span>
          {" \u00B7 "}
          Commissioned{" "}
          {new Date(job.created_at).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
          })}
        </p>
        <div className="flex items-center gap-2">
          <FormatBadge formatType={job.format_type} />
          <StatusDot status={job.status} />
          {isActive && (
            <span className="inline-flex items-center gap-1 text-xs text-primary">
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
              Live
            </span>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <SectionHeader label="THE PUBLISHED PIECE" />
        {renderSection1()}
      </div>

      <div className="space-y-4">
        <SectionHeader label="THE EDITORIAL TRAIL" />
        <div className="rounded-lg border border-border p-6">
          <EditorialTimeline job={job} />
        </div>
      </div>

      {job.status === "HUMAN_REVIEW_NEEDED" && (
        <ReviewSection
          job={job}
          approvalMutation={approvalMutation}
          feedbackText={feedbackText}
          setFeedbackText={setFeedbackText}
          showRejectForm={showRejectForm}
          setShowRejectForm={setShowRejectForm}
        />
      )}
    </div>
  );
}

function FailedSection({ job }: { job: RenderJobResponse }) {
  const errorLog = job.error_log as Record<string, unknown> | null;
  const stage = typeof errorLog?.stage === "string" ? errorLog.stage : "";
  const message =
    typeof errorLog?.message === "string"
      ? errorLog.message
      : typeof errorLog?.error === "string"
        ? errorLog.error
        : "An unknown error occurred during processing.";
  const agent = typeof errorLog?.agent === "string" ? errorLog.agent : "";

  return (
    <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-6 space-y-4">
      <div className="flex items-start gap-3">
        <span className="text-lg leading-none mt-0.5 text-destructive">&#9888;</span>
        <div>
          <h4 className="font-heading text-lg font-semibold text-destructive">
            Story Killed
          </h4>
          <p className="text-sm text-muted-foreground mt-1">{message}</p>
        </div>
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
        {stage && <p>Phase: {stage}</p>}
        {agent && <p>Agent: {agent}</p>}
        <p>
          Time:{" "}
          {new Date(job.updated_at).toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
            hour: "numeric",
            minute: "2-digit",
          })}
        </p>
      </div>
      {errorLog && (
        <CollapsibleSection label="Technical Details">
          {JSON.stringify(errorLog, null, 2)}
        </CollapsibleSection>
      )}
      <p className="text-xs text-muted-foreground">
        This story won&apos;t retry automatically.{" "}
        <button
          type="button"
          className="text-primary hover:underline cursor-pointer"
          onClick={() => {
            const rawText =
              (job.pre_context as Record<string, unknown>)?.raw_text ?? "";
            window.location.href = `/jobs/new?topic=${encodeURIComponent(job.topic)}&raw_text=${encodeURIComponent(typeof rawText === "string" ? rawText : "")}`;
          }}
        >
          Commission it again
        </button>
      </p>
    </div>
  );
}

function FormatTabs({
  formatScripts,
  platform,
}: {
  formatScripts: ScriptResponse[];
  platform: string | null | undefined;
}) {
  if (formatScripts.length === 0) {
    return (
      <div className="rounded-lg border border-border p-6 text-center">
        <p className="text-sm text-muted-foreground">
          No format outputs generated.
        </p>
      </div>
    );
  }

  if (formatScripts.length === 1) {
    const s = formatScripts[0];
    return (
      <div className="rounded-lg border border-border">
        <div className="border-b border-border bg-muted/30 px-4 py-2">
          <span className="font-heading text-sm font-semibold">
            {(s.format_type ?? "").charAt(0).toUpperCase() +
              (s.format_type ?? "").slice(1).toLowerCase()}
          </span>
        </div>
        <div className="p-5">
          {renderFormatViewer(
            s.format_type ?? "",
            s.format_payload as Record<string, unknown>,
            platform
          )}
        </div>
      </div>
    );
  }

  return (
    <Tabs defaultValue={formatScripts[0].format_type ?? `fmt-${formatScripts[0].id}`}>
      <div className="overflow-x-auto -mx-1 px-1">
        <TabsList variant="line" className="mb-0">
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
      </div>
      {formatScripts.map((s: ScriptResponse) => (
        <TabsContent
          key={s.id}
          value={s.format_type ?? `fmt-${s.id}`}
        >
          <div className="rounded-lg border border-border p-5">
            {renderFormatViewer(
              s.format_type ?? "",
              s.format_payload as Record<string, unknown>,
              platform
            )}
          </div>
        </TabsContent>
      ))}
    </Tabs>
  );
}

function ClaimsSection({ claims }: { claims: FactCheckClaimResponse[] }) {
  if (claims.length === 0) return null;
  return (
    <div className="space-y-2">
      {claims.map((claim: FactCheckClaimResponse) => (
        <div
          key={claim.id}
          className="rounded-md border border-border p-3 text-sm space-y-1"
        >
          <p className="italic text-muted-foreground">
            &ldquo;{claim.claim_text}&rdquo;
          </p>
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`inline-flex items-center gap-1 font-semibold ${
                claim.verdict === "SUPPORTED"
                  ? "text-success"
                  : claim.verdict === "CONTESTED"
                    ? "text-info"
                    : claim.verdict === "UNSUPPORTED"
                      ? "text-destructive"
                      : "text-warning"
              }`}
            >
              <span
                className={`inline-block h-1.5 w-1.5 rounded-full ${
                  claim.verdict === "SUPPORTED"
                    ? "bg-success"
                    : claim.verdict === "CONTESTED"
                      ? "bg-info"
                      : claim.verdict === "UNSUPPORTED"
                        ? "bg-destructive"
                        : "bg-warning"
                }`}
              />
              {claim.verdict}
            </span>
            <span className="text-muted-foreground">
              {(claim.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ActiveOutput({ job }: { job: RenderJobResponse }) {
  const status = job.status;
  const scripts = job.scripts ?? [];
  const assets = job.assets ?? [];
  const masterScript = scripts.find(
    (s: ScriptResponse) => s.role === "master"
  );
  const formatScripts = scripts.filter(
    (s: ScriptResponse) => s.role === "format"
  );
  const allClaims = scripts.flatMap(
    (s: ScriptResponse) => s.claims ?? []
  );

  if (status === "PENDING") {
    return (
      <div className="rounded-lg border border-border p-6 space-y-3">
        <h4 className="font-heading text-base font-semibold">{job.topic}</h4>
        <div className="flex flex-wrap gap-2">
          <FormatBadge formatType={job.format_type} />
          {job.platform && (
            <span className="inline-flex items-center rounded-[4px] bg-muted px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-[0.05em] text-muted-foreground">
              {job.platform}
            </span>
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          Awaiting assignment&hellip;
        </p>
      </div>
    );
  }

  if (
    (status === "RESEARCHING" || status === "FACT_CHECKING_RESEARCH") &&
    job.refined_context
  ) {
    return (
      <div className="rounded-lg border border-border p-5 space-y-3">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-xs text-primary">
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
            Research in progress
          </span>
        </div>
        <p className="text-sm leading-relaxed whitespace-pre-wrap">
          {job.refined_context}
        </p>
      </div>
    );
  }

  if (status === "RESEARCHING" || status === "FACT_CHECKING_RESEARCH") {
    return (
      <div className="rounded-lg border border-border p-6 text-center space-y-2">
        <p className="font-heading text-base font-semibold text-muted-foreground">
          Researching&hellip;
        </p>
        <p className="text-sm text-muted-foreground">
          Gathering and analyzing sources.
        </p>
      </div>
    );
  }

  if (status === "SCRIPTING" || status === "FACT_CHECKING_SCRIPT") {
    if (masterScript) {
      const hasRevisions = masterScript.feedback_history.length > 0;
      return (
        <div className="rounded-lg border border-border p-5 space-y-4">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs text-primary">
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
              Draft v{masterScript.version}
              {hasRevisions ? " (revised)" : ""}
            </span>
          </div>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {masterScript.content}
          </p>
          {status === "FACT_CHECKING_SCRIPT" && allClaims.length > 0 && (
            <div className="space-y-2 pt-2 border-t border-border">
              <ClaimsSection claims={allClaims} />
            </div>
          )}
        </div>
      );
    }
    return (
      <div className="rounded-lg border border-border p-6 text-center">
        <p className="font-heading text-base font-semibold text-muted-foreground">
          {status === "SCRIPTING" ? "Writing&hellip;" : "Evaluating&hellip;"}
        </p>
      </div>
    );
  }

  if (status === "FORMATTING") {
    if (formatScripts.length > 0) {
      return <FormatTabs formatScripts={formatScripts} platform={job.platform} />;
    }
    return (
      <div className="rounded-lg border border-border p-6 text-center">
        <p className="font-heading text-base font-semibold text-muted-foreground">
          Typesetting&hellip;
        </p>
        <p className="text-sm text-muted-foreground mt-1">
          Generating format outputs.
        </p>
      </div>
    );
  }

  if (status === "ASSET_GENERATION") {
    const videoScript = formatScripts.find(
      (s: ScriptResponse) => s.format_type === "VIDEO"
    );
    return (
      <div className="space-y-4">
        {videoScript && videoScript.format_payload && (
          <div className="rounded-lg border border-border p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">
              Video Script
            </p>
            {renderFormatViewer(
              "VIDEO",
              videoScript.format_payload as Record<string, unknown>
            )}
          </div>
        )}
        {assets.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2">
            {assets.map((asset: AssetResponse) => (
              <div
                key={asset.id}
                className="rounded-lg border border-border p-4 space-y-2"
              >
                <span className="inline-flex items-center rounded-[4px] bg-primary/10 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.05em] text-primary">
                  {asset.asset_type.replace(/_/g, " ")}
                </span>
                <p className="font-mono text-xs text-muted-foreground break-all">
                  {asset.url_or_path || "Pending render\u2026"}
                </p>
                {asset.render_meta?.prompt_used && (
                  <CollapsibleSection label="Generation Prompt">
                    {asset.render_meta.prompt_used}
                  </CollapsibleSection>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return null;
}

function ReviewSection({
  job,
  approvalMutation,
  feedbackText,
  setFeedbackText,
  showRejectForm,
  setShowRejectForm,
}: {
  job: RenderJobResponse;
  approvalMutation: ReturnType<typeof useApproveScript>;
  feedbackText: string;
  setFeedbackText: (v: string) => void;
  showRejectForm: boolean;
  setShowRejectForm: (v: boolean) => void;
}) {
  const scripts = job.scripts ?? [];
  const masterScript = scripts.find(
    (s: ScriptResponse) => s.role === "master"
  );

  return (
    <div className="space-y-4">
      <SectionHeader label="EDITORIAL REVIEW" />
      <div className="rounded-lg border border-primary/30 bg-primary/5 p-6 space-y-4">
        <div>
          <h4 className="font-heading text-base font-semibold">
            This story needs your review
          </h4>
          <p className="text-sm text-muted-foreground mt-1">
            {masterScript && masterScript.feedback_history.length > 0
              ? `${masterScript.feedback_history.length} revision cycle${masterScript.feedback_history.length !== 1 ? "s" : ""} exhausted.`
              : "The script requires your approval before proceeding."}
          </p>
        </div>
        {approvalMutation.isError && (
          <p className="text-sm text-destructive">
            Error:{" "}
            {(approvalMutation.error as Error)?.message || "Failed to submit"}
          </p>
        )}
        {showRejectForm ? (
          <div className="space-y-3">
            <Textarea
              placeholder="Optional feedback for the script agent\u2026"
              value={feedbackText}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) =>
                setFeedbackText(e.target.value)
              }
            />
            <div className="flex gap-2">
              <Button
                variant="outline"
                className="border-destructive text-destructive hover:bg-destructive/10"
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
                      onSuccess: () =>
                        toast.info("Revision requested. Polling resumed."),
                      onError: () =>
                        toast.error("Action failed. Please try again."),
                    }
                  );
                }}
              >
                {approvalMutation.isPending
                  ? "Submitting\u2026"
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
          <div className="flex gap-3">
            <Button
              className="bg-primary text-primary-foreground hover:bg-primary/90"
              disabled={approvalMutation.isPending}
              onClick={() =>
                approvalMutation.mutate(
                  { isApproved: true },
                  {
                    onSuccess: () =>
                      toast.success("Script approved. Pipeline resuming."),
                    onError: () =>
                      toast.error("Action failed. Please try again."),
                  }
                )
              }
            >
              {approvalMutation.isPending
                ? "Approving\u2026"
                : "Approve & Publish"}
            </Button>
            <Button
              variant="outline"
              disabled={approvalMutation.isPending}
              onClick={() => setShowRejectForm(true)}
            >
              Request Revision
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
