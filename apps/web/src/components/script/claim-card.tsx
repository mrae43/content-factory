"use client";

import { CollapsibleSection } from "@/components/editorial/collapsible-section";

const verdictConfig: Record<string, { dot: string; label: string }> = {
  SUPPORTED: { dot: "bg-success", label: "SUPPORTED" },
  CONTESTED: { dot: "bg-info", label: "CONTESTED" },
  UNSUPPORTED: { dot: "bg-destructive", label: "UNSUPPORTED" },
  UNCERTAIN: { dot: "bg-warning", label: "UNCERTAIN" },
};

interface ClaimCardProps {
  claim_text: string;
  verdict: string;
  confidence: number;
  evidence_text?: string | null;
  evidence_references: string[];
}

export function ClaimCard({
  claim_text,
  verdict,
  confidence,
  evidence_text,
  evidence_references,
}: ClaimCardProps) {
  const config = verdictConfig[verdict] ?? { dot: "bg-muted-foreground", label: verdict };
  const pct = Math.round(confidence * 100);

  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-3">
      <p className="text-sm leading-relaxed italic text-foreground">
        &ldquo;{claim_text}&rdquo;
      </p>

      <div className="flex flex-wrap items-center gap-3">
        <span className={`inline-flex items-center gap-1.5 text-xs font-semibold ${config.dot === "bg-success" ? "text-success" : config.dot === "bg-info" ? "text-info" : config.dot === "bg-destructive" ? "text-destructive" : "text-warning"}`}>
          <span className={`inline-block h-2 w-2 rounded-full ${config.dot}`} />
          {config.label}
        </span>
        <div className="flex-1 h-1.5 rounded-full bg-muted max-w-32 overflow-hidden">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
        <span className="text-xs tabular-nums text-muted-foreground">
          {pct}%
        </span>
      </div>

      <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.05em] text-muted-foreground">
        <span className="font-medium">{config.label}</span>
        {evidence_references.length > 0 && (
          <span>
            &middot; sources: {evidence_references.length} chunk
            {evidence_references.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {evidence_text && (
        <CollapsibleSection label="Evidence">
          <p className="text-sm leading-relaxed text-muted-foreground">
            {evidence_text}
          </p>
        </CollapsibleSection>
      )}

      {evidence_references.length > 0 && (
        <CollapsibleSection label="Source Chunks">
          <div className="space-y-1 text-muted-foreground">
            {evidence_references.map((ref, i) => (
              <p key={i}>Chunk: {ref}</p>
            ))}
          </div>
        </CollapsibleSection>
      )}
    </div>
  );
}
