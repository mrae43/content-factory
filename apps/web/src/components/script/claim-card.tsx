"use client";

import { Badge } from "@/components/ui/badge";

const verdictStyles: Record<string, string> = {
  SUPPORTED: "bg-success/15 text-success",
  UNSUPPORTED: "bg-destructive/15 text-destructive",
  CONTESTED: "bg-warning/15 text-warning",
  UNCERTAIN: "bg-muted text-muted-foreground",
};

interface ClaimCardProps {
  claim_text: string;
  verdict: string;
  confidence: number;
  evidence_references: string[];
}

export function ClaimCard({
  claim_text,
  verdict,
  confidence,
  evidence_references,
}: ClaimCardProps) {
  return (
    <div className="rounded-lg border p-4 space-y-2">
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm">{claim_text}</p>
        <div className="flex items-center gap-2 shrink-0">
          <Badge className={verdictStyles[verdict] || "bg-muted text-muted-foreground"}>
            {verdict}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      {evidence_references.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {evidence_references.length} evidence reference(s)
        </p>
      )}
    </div>
  );
}
