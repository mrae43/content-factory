"use client";

import { useJobDetail, useApproveScript, useRegenerateAssets } from "@/hooks/use-jobs";
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
import { ClaimCard } from "@/components/script/claim-card";

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
          <FormatTabs formatScripts={formatScripts} platform={job.platform} jobId={jobId} />
        </div>
      );
    }
    return <ActiveOutput job={job} />;
  }

  const defaultTab =
    job.status === "COMPLETED"
      ? "output"
      : job.status === "HUMAN_REVIEW_NEEDED"
        ? "review"
        : "trail";

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
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

      <Tabs defaultValue={defaultTab}>
        <TabsList variant="line">
          <TabsTrigger value="output">Output</TabsTrigger>
          <TabsTrigger value="trail">Trail</TabsTrigger>
          <TabsTrigger value="review" className="relative">
            Review
            {job.status === "HUMAN_REVIEW_NEEDED" ? (
              <span className="ml-1.5 inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-bold text-white" style={{ backgroundColor: 'var(--warning)' }}>
                1
              </span>
            ) : (
              <span className="ml-1.5 text-[10px] text-muted-foreground">0</span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="output">
          <div className="pt-4 space-y-4">
            <SectionHeader label="THE PUBLISHED PIECE" />
            {renderSection1()}
          </div>
        </TabsContent>

        <TabsContent value="trail">
          <div className="pt-4 space-y-8">
            <div className="space-y-4">
              <SectionHeader label="THE EDITORIAL TRAIL" />
              <div className="rounded-lg border border-border p-6">
                <EditorialTimeline job={job} />
              </div>
            </div>

            <FactCheckAudit allClaims={allClaims} />

            <CitationIndexSection citationIndex={job.citation_index} />
          </div>
        </TabsContent>

        <TabsContent value="review">
          <div className="pt-4 space-y-4">
            {job.status === "HUMAN_REVIEW_NEEDED" ? (
              <ReviewSection
                job={job}
                approvalMutation={approvalMutation}
                feedbackText={feedbackText}
                setFeedbackText={setFeedbackText}
                showRejectForm={showRejectForm}
                setShowRejectForm={setShowRejectForm}
              />
            ) : (
              <div className="rounded-lg border border-border p-6 text-center">
                <p className="text-sm text-muted-foreground">
                  No review required at this stage.
                </p>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
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
  jobId,
}: {
  formatScripts: ScriptResponse[];
  platform: string | null | undefined;
  jobId?: string;
}) {
  const regenerateMutation = useRegenerateAssets(jobId ?? "");

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
    const isCarousel = s.format_type === "CAROUSEL";
    return (
      <div className="rounded-lg border border-border">
        <div className="border-b border-border bg-muted/30 px-4 py-2 flex items-center justify-between">
          <span className="font-heading text-sm font-semibold">
            {(s.format_type ?? "").charAt(0).toUpperCase() +
              (s.format_type ?? "").slice(1).toLowerCase()}
          </span>
          {isCarousel && jobId && (
            <Button
              variant="outline"
              size="sm"
              disabled={regenerateMutation.isPending}
              onClick={() =>
                regenerateMutation.mutate(undefined, {
                  onSuccess: () => toast.success("Images regenerated"),
                  onError: () => toast.error("Image regeneration failed"),
                })
              }
            >
              {regenerateMutation.isPending ? "Regenerating\u2026" : "Regenerate Images"}
            </Button>
          )}
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
      {formatScripts.map((s: ScriptResponse) => {
        const isCarousel = s.format_type === "CAROUSEL";
        return (
          <TabsContent
            key={s.id}
            value={s.format_type ?? `fmt-${s.id}`}
          >
            <div className="rounded-lg border border-border p-5">
              {isCarousel && jobId && (
                <div className="flex justify-end mb-3">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={regenerateMutation.isPending}
                    onClick={() =>
                      regenerateMutation.mutate(undefined, {
                        onSuccess: () => toast.success("Images regenerated"),
                        onError: () => toast.error("Image regeneration failed"),
                      })
                    }
                  >
                    {regenerateMutation.isPending ? "Regenerating\u2026" : "Regenerate Images"}
                  </Button>
                </div>
              )}
              {renderFormatViewer(
                s.format_type ?? "",
                s.format_payload as Record<string, unknown>,
                platform
              )}
            </div>
          </TabsContent>
        );
      })}
    </Tabs>
  );
}

function annotateOutput(
  content: string,
  claims: FactCheckClaimResponse[]
): { html: string; matchedCount: number } {
  let annotated = content;
  let matchedCount = 0;
  const seen = new Set<string>();
  claims.forEach((claim, i) => {
    if (seen.has(claim.claim_text)) return;
    seen.add(claim.claim_text);
    const idx = annotated.indexOf(claim.claim_text);
    if (idx !== -1) {
      const before = annotated.slice(0, idx + claim.claim_text.length);
      const after = annotated.slice(idx + claim.claim_text.length);
      annotated = `${before}<sup class="claim-ref cursor-help" title="${claim.verdict} — ${(claim.confidence * 100).toFixed(0)}%" data-index="${i}">[${i + 1}]</sup>${after}`;
      matchedCount++;
    }
  });
  return { html: annotated, matchedCount };
}

function AnnotatedContent({
  content,
  claims,
}: {
  content: string;
  claims: FactCheckClaimResponse[];
}) {
  const { html, matchedCount } = annotateOutput(content, claims);
  return (
    <div>
      <p
        className="text-sm leading-relaxed whitespace-pre-wrap"
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {matchedCount > 0 && <ClaimReferenceLegend claims={claims} />}
    </div>
  );
}

function ClaimReferenceLegend({
  claims,
}: {
  claims: FactCheckClaimResponse[];
}) {
  const seen = new Set<string>();
  const unique = claims.filter((c) => {
    if (seen.has(c.claim_text)) return false;
    seen.add(c.claim_text);
    return true;
  });
  return (
    <div className="mt-3 pt-3 border-t border-border space-y-1.5">
      <p className="text-[11px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">
        Claim References
      </p>
      <div className="space-y-1">
        {unique.map((claim, i) => (
          <p
            key={i}
            className="text-xs text-muted-foreground flex items-start gap-2"
          >
            <span className="font-mono text-[10px] text-primary font-semibold mt-0.5 shrink-0">
              [{i + 1}]
            </span>
            <span className="italic">&ldquo;{claim.claim_text}&rdquo;</span>
            <span
              className={`shrink-0 font-semibold ${
                claim.verdict === "SUPPORTED"
                  ? "text-success"
                  : claim.verdict === "CONTESTED"
                    ? "text-info"
                    : claim.verdict === "UNSUPPORTED"
                      ? "text-destructive"
                      : "text-warning"
              }`}
            >
              {claim.verdict}
            </span>
            <span className="text-muted-foreground">
              {(claim.confidence * 100).toFixed(0)}%
            </span>
          </p>
        ))}
      </div>
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
    (status === "RESEARCHING" || status === "RETRIEVAL") &&
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

  if (status === "RESEARCHING" || status === "RETRIEVAL") {
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
          <AnnotatedContent
            content={masterScript.content}
            claims={allClaims}
          />

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
      return <FormatTabs formatScripts={formatScripts} platform={job.platform} jobId={job.id} />;
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

function FactCheckAudit({
  allClaims,
}: {
  allClaims: FactCheckClaimResponse[];
}) {
  return (
    <div id="fact-check-audit" className="space-y-4">
      <SectionHeader label="FACT CHECK AUDIT" />
      <div className="rounded-lg border border-border p-5 space-y-3">
        {allClaims.length > 0 ? (
          <div className="space-y-3">
            {allClaims.map((claim, i) => (
              <ClaimCard
                key={i}
                claim_text={claim.claim_text}
                verdict={claim.verdict}
                confidence={claim.confidence}
                evidence_text={claim.evidence_text}
                evidence_references={claim.evidence_references ?? []}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No verifiable factual claims found in this output.
            All content passed editorial review as non-factual or
            opinion-based material.
          </p>
        )}
      </div>
    </div>
  );
}

type CitationEntry = {
  claim_fragment?: string;
  source_url?: string;
  chunk_id?: string;
};

function CitationIndexSection({
  citationIndex,
}: {
  citationIndex: { [key: string]: unknown }[] | null | undefined;
}) {
  if (!citationIndex || citationIndex.length === 0) return null;

  const entries = citationIndex as CitationEntry[];

  return (
    <div id="citation-index" className="space-y-4">
      <SectionHeader label="CITATION INDEX" />
      <div className="rounded-lg border border-border p-5 space-y-3">
        <p className="text-xs text-muted-foreground">
          {entries.length} citation{entries.length > 1 ? "s" : ""} mapped from research synthesis
        </p>
        <div className="space-y-2">
          {entries.map((entry, i) => (
            <div
              key={i}
              className="rounded-md bg-muted p-3 space-y-1 text-xs"
            >
              <p className="font-medium text-foreground">
                &ldquo;{entry.claim_fragment ?? ""}&rdquo;
              </p>
              {entry.source_url && (
                <a
                  href={entry.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline break-all"
                >
                  {entry.source_url}
                </a>
              )}
              {entry.chunk_id && (
                <p className="text-muted-foreground font-mono">
                  Chunk: {entry.chunk_id.slice(0, 8)}&hellip;
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
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
