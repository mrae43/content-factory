"use client";

import { CollapsibleSection } from "./collapsible-section";
import type { ScriptResponse, FactCheckClaimResponse, AssetResponse } from "@content-factory/shared-types";
import { ClaimCard } from "@/components/script/claim-card";

const pipelineStages = [
  "PENDING",
  "RESEARCHING",
  "FACT_CHECKING_RESEARCH",
  "SCRIPTING",
  "FACT_CHECKING_SCRIPT",
  "FORMATTING",
  "ASSET_GENERATION",
] as const;

const deskConfig: Record<string, { name: string; color: string }> = {
  PENDING: { name: "Assignment Queue", color: "bg-muted-foreground" },
  RESEARCHING: { name: "Research Desk", color: "bg-warning" },
  FACT_CHECKING_RESEARCH: { name: "Source Verification", color: "bg-warning" },
  SCRIPTING: { name: "Writer's Desk", color: "bg-info" },
  FACT_CHECKING_SCRIPT: { name: "Fact-Check Desk", color: "bg-info" },
  FORMATTING: { name: "Layout Desk", color: "bg-accent-purple" },
  ASSET_GENERATION: { name: "Production Studio", color: "bg-accent-teal" },
};

type StageState = "completed" | "active" | "future";

function getStageIndex(status: string): number {
  if (status === "COMPLETED") return pipelineStages.length;
  if (status === "FAILED" || status === "HUMAN_REVIEW_NEEDED") {
    const idx = pipelineStages.indexOf(status as (typeof pipelineStages)[number]);
    return idx >= 0 ? idx : pipelineStages.length;
  }
  return pipelineStages.indexOf(status as (typeof pipelineStages)[number]);
}

function getStageState(stage: string, currentStatus: string): StageState {
  if (currentStatus === "COMPLETED") return "completed";
  if (currentStatus === "FAILED" || currentStatus === "HUMAN_REVIEW_NEEDED") {
    const currentIdx = getStageIndex(currentStatus);
    const stageIdx = pipelineStages.indexOf(stage as (typeof pipelineStages)[number]);
    if (stageIdx < currentIdx) return "completed";
    if (stageIdx === currentIdx) return "completed";
    return "future";
  }
  const currentIdx = getStageIndex(currentStatus);
  const stageIdx = pipelineStages.indexOf(stage as (typeof pipelineStages)[number]);
  if (stageIdx < currentIdx) return "completed";
  if (stageIdx === currentIdx) return "active";
  return "future";
}

function formatDuration(startDate: string, endDate?: string): string {
  const start = new Date(startDate).getTime();
  const end = endDate ? new Date(endDate).getTime() : Date.now();
  const diffSec = Math.round((end - start) / 1000);
  if (diffSec < 60) return `${diffSec}s`;
  if (diffSec < 3600) return `${Math.round(diffSec / 60)} min`;
  return `${Math.round(diffSec / 3600)}h ${Math.round((diffSec % 3600) / 60)}m`;
}

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.round(diffMs / 1000);
  if (diffSec < 60) return "just now";
  if (diffSec < 3600) return `${Math.round(diffSec / 60)} min ago`;
  if (diffSec < 86400) return `${Math.round(diffSec / 3600)}h ago`;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

type FeedbackEntry =
  | string
  | {
      overall_reasoning?: string;
      feedback_type?: string;
      failed_claims?: Array<{ verdict?: string; claim_text?: string; evidence?: string }>;
      [key: string]: unknown;
    };

interface TimelineNodeProps {
  stage: string;
  state: StageState;
  isLast: boolean;
  job: {
    id: string;
    status: string;
    created_at: string;
    updated_at: string;
    refined_context?: string | null;
    pre_context?: Record<string, unknown> | null;
    scripts?: ScriptResponse[];
    assets?: AssetResponse[];
    format_type?: string | null;
  };
}

function TimelineNode({ stage, state, isLast, job }: TimelineNodeProps) {
  const config = deskConfig[stage] ?? { name: stage.replace(/_/g, " "), color: "bg-muted-foreground" };
  const scripts = job.scripts ?? [];
  const masterScript = scripts.find((s) => !s.format_payload);
  const formatScripts = scripts.filter((s) => s.format_payload);
  const claims = masterScript?.claims ?? [];
  const assets = job.assets ?? [];

  const isCompleted = state === "completed";
  const isActive = state === "active";

  const dotColor = isActive
    ? "bg-primary animate-pulse"
    : isCompleted
      ? config.color
      : "bg-border";

  const checkMark = isCompleted ? "\u2713" : isActive ? "\u25CF" : "";

  let summaryText = "";
  let outputContent: React.ReactNode = null;

  if (stage === "PENDING") {
    summaryText = "Awaiting assignment";
  }

  if (stage === "RESEARCHING") {
    if (isCompleted || isActive) {
      summaryText = isCompleted
        ? `Completed \u00B7 ${job.refined_context ? "Research summary available" : "No summary"}`
        : "Gathering sources\u2026";
      if (job.refined_context) {
        outputContent = (
          <div className="mt-3 rounded-md bg-muted p-4">
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{job.refined_context}</p>
          </div>
        );
      }
    }
  }

  if (stage === "FACT_CHECKING_RESEARCH") {
    summaryText = isCompleted ? "Passthrough \u00B7 <1s" : isActive ? "Verifying\u2026" : "";
  }

  if (stage === "SCRIPTING") {
    if (isCompleted || isActive) {
      if (masterScript) {
        const hasRevisions = masterScript.feedback_history.length > 0;
        summaryText = isCompleted
          ? `Completed \u00B7 v${masterScript.version}${hasRevisions ? " (revised)" : ""}`
          : "Drafting\u2026";
        outputContent = (
          <div className="mt-3 space-y-3">
            <div className="rounded-md bg-muted p-4">
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{masterScript.content}</p>
            </div>
            {masterScript.feedback_history.length > 0 && (
              <div className="space-y-2 border-t border-border pt-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Revision History ({masterScript.feedback_history.length})
                </p>
                {masterScript.feedback_history.map((entry: FeedbackEntry, i: number) => {
                  if (typeof entry === "string") {
                    return (
                      <div key={i} className="rounded-md bg-muted p-3 text-xs text-muted-foreground">
                        <span className="font-medium">Feedback #{i + 1}:</span> {entry}
                      </div>
                    );
                  }
                  return (
                    <div key={i} className="rounded-md bg-muted p-3 space-y-1 text-xs">
                      <span className="font-medium">Revision #{i + 1}</span>
                      {entry.overall_reasoning && (
                        <p className="text-muted-foreground">{entry.overall_reasoning}</p>
                      )}
                      {Array.isArray(entry.failed_claims) && entry.failed_claims.length > 0 && (
                        <ul className="space-y-0.5">
                          {entry.failed_claims.map((fc, j) => (
                            <li key={j} className="flex items-start gap-1.5">
                              <span className="shrink-0 mt-0.5 inline-block h-1.5 w-1.5 rounded-full bg-destructive" />
                              <span className="text-muted-foreground">{fc.claim_text ?? ""}</span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      } else {
        summaryText = isActive ? "Drafting\u2026" : "";
      }
    }
  }

  if (stage === "FACT_CHECKING_SCRIPT") {
    if (isCompleted || isActive) {
      const claimCount = claims.length;
      const supportedCount = claims.filter((c: FactCheckClaimResponse) => c.verdict === "SUPPORTED").length;
      summaryText = isCompleted
        ? `${masterScript?.is_approved ? "Approved" : "Evaluated"} \u00B7 ${claimCount} claims${claimCount > 0 ? ` (${supportedCount} supported)` : ""}`
        : "Evaluating claims\u2026";
      if (claims.length > 0) {
        outputContent = (
          <div className="mt-3 space-y-2">
            {claims.map((claim: FactCheckClaimResponse) => (
              <ClaimCard
                key={claim.id}
                claim_text={claim.claim_text}
                verdict={claim.verdict}
                confidence={claim.confidence}
                evidence_references={claim.evidence_references ?? []}
              />
            ))}
          </div>
        );
      }
    }
  }

  if (stage === "FORMATTING") {
    if (isCompleted || isActive) {
      const fmtLabels = formatScripts.map((s) => {
        const ft = s.format_type ?? "";
        return ft.charAt(0).toUpperCase() + ft.slice(1).toLowerCase();
      });
      summaryText = isCompleted
        ? `${fmtLabels.length > 0 ? fmtLabels.join(" + ") : "Complete"} \u00B7 ${formatScripts.length} format(s)`
        : "Typesetting\u2026";
    }
  }

  if (stage === "ASSET_GENERATION") {
    if (isCompleted || isActive) {
      summaryText = isCompleted
        ? `${assets.length} asset(s) generated`
        : "Rendering\u2026";
      if (assets.length > 0) {
        outputContent = (
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {assets.map((asset: AssetResponse) => (
              <div key={asset.id} className="rounded-md border border-border bg-muted/50 p-3 space-y-1.5">
                <span className="inline-flex items-center rounded-[4px] bg-primary/10 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.05em] text-primary">
                  {asset.asset_type.replace(/_/g, " ")}
                </span>
                <p className="font-mono text-xs text-muted-foreground break-all">{asset.url_or_path}</p>
                {asset.render_meta?.prompt_used && (
                  <CollapsibleSection label="Generation Prompt">
                    {asset.render_meta.prompt_used}
                  </CollapsibleSection>
                )}
              </div>
            ))}
          </div>
        );
      }
    }
  }

  const showOutput = outputContent !== null;
  const showSourceMaterials = stage === "RESEARCHING" && (isCompleted || isActive) && job.pre_context;
  const preCtx = job.pre_context as Record<string, unknown> | null;

  return (
    <div className="relative flex gap-4">
      <div className="flex flex-col items-center">
        <div
          className={`mt-0.5 h-3 w-3 shrink-0 rounded-full ${dotColor}`}
          aria-hidden="true"
        />
        {!isLast && (
          <div className="w-0.5 flex-1 bg-border mt-1" />
        )}
      </div>
      <div className={`flex-1 pb-8 ${isLast ? "pb-0" : ""}`}>
        <div className="flex items-baseline gap-2">
          <h4 className="font-heading text-base font-semibold leading-tight">
            {config.name}
          </h4>
          {summaryText && (
            <span className="text-xs text-muted-foreground">
              {checkMark && <span className="mr-1">{checkMark}</span>}
              {summaryText}
            </span>
          )}
          {isActive && (
            <span className="inline-flex items-center gap-1 text-xs text-primary">
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
              Live
            </span>
          )}
        </div>

        {showSourceMaterials && preCtx && (
          <div className="mt-2 space-y-1.5 text-xs">
            {Array.isArray(preCtx.source_urls) && preCtx.source_urls.length > 0 && (
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                {(preCtx.source_urls as string[]).map((url, i) => (
                  <a
                    key={i}
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline break-all"
                  >
                    {url.replace(/^https?:\/\//, "").replace(/\/$/, "").slice(0, 60)}
                    {url.length > 70 ? "\u2026" : ""}
                  </a>
                ))}
              </div>
            )}
            {typeof preCtx.raw_text === "string" && preCtx.raw_text.length > 0 && (
              <CollapsibleSection label="Reference Text">
                {preCtx.raw_text.slice(0, 3000)}
                {preCtx.raw_text.length > 3000 ? "\n\u2026 (truncated)" : ""}
              </CollapsibleSection>
            )}
            <div className="flex gap-3 text-muted-foreground">
              {typeof preCtx.target_audience === "string" && preCtx.target_audience !== "General" && (
                <span>Audience: {preCtx.target_audience}</span>
              )}
              {typeof preCtx.guardrail_strictness === "string" && (
                <span>Strictness: {preCtx.guardrail_strictness}</span>
              )}
            </div>
          </div>
        )}

        {showOutput && outputContent}

        {(stage === "SCRIPTING" && masterScript) && (
          <CollapsibleSection label="Technical Details">
            <div className="space-y-1">
              <p>Script ID: {masterScript.id.slice(0, 8)}\u2026</p>
              <p>Version: {masterScript.version}</p>
              <p>Approved: {masterScript.is_approved ? "Yes" : "No"}</p>
              <p>Revisions: {masterScript.feedback_history.length}</p>
            </div>
          </CollapsibleSection>
        )}

        {(stage === "FACT_CHECKING_SCRIPT" && claims.length > 0) && (
          <CollapsibleSection label="Full Audit Trail">
            <div className="space-y-2">
              {claims.map((claim: FactCheckClaimResponse) => (
                <div key={claim.id} className="space-y-0.5">
                  <p className="font-medium">{claim.claim_text.slice(0, 100)}\u2026</p>
                  <p className="text-muted-foreground">
                    {claim.verdict} \u00B7 {(claim.confidence * 100).toFixed(0)}% confidence
                  </p>
                  {claim.evidence_references && claim.evidence_references.length > 0 && (
                    <p className="text-muted-foreground">
                      Evidence: {claim.evidence_references.length} chunk(s)
                    </p>
                  )}
                </div>
              ))}
            </div>
          </CollapsibleSection>
        )}

        {(stage === "FORMATTING" && formatScripts.length > 0) && (
          <CollapsibleSection label="Harness Details">
            {formatScripts.map((s) => (
              <div key={s.id} className="space-y-0.5 mb-2">
                <p className="font-medium">{(s.format_type ?? "UNKNOWN").toUpperCase()}</p>
                <p className="text-muted-foreground">Script ID: {s.id.slice(0, 8)}\u2026</p>
                {s.format_payload && (
                  <pre className="text-muted-foreground mt-1">
                    {JSON.stringify(s.format_payload, null, 2).slice(0, 500)}\u2026
                  </pre>
                )}
              </div>
            ))}
          </CollapsibleSection>
        )}
      </div>
    </div>
  );
}

interface EditorialTimelineProps {
  job: {
    id: string;
    status: string;
    created_at: string;
    updated_at: string;
    refined_context?: string | null;
    pre_context?: Record<string, unknown> | null;
    scripts?: ScriptResponse[];
    assets?: AssetResponse[];
    format_type?: string | null;
  };
}

export function EditorialTimeline({ job }: EditorialTimelineProps) {
  const fmt = (job.format_type ?? "all").toLowerCase();
  const skipAssetGeneration = fmt === "blog" || fmt === "carousel";

  const stages = pipelineStages.filter((s) => {
    if (s === "ASSET_GENERATION" && skipAssetGeneration) return false;
    return true;
  });

  return (
    <div>
      {stages.map((stage, i) => {
        const state = getStageState(stage, job.status);
        if (state === "future") return null;
        return (
          <TimelineNode
            key={stage}
            stage={stage}
            state={state}
            isLast={i === stages.length - 1 || state === "active"}
            job={job}
          />
        );
      })}
    </div>
  );
}

export { formatRelativeTime };
