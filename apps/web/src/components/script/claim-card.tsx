"use client";

import { Badge } from "@/components/ui/badge";

const verdictColors: Record<string, string> = {
  SUPPORTED: "bg-green-100 text-green-800",
  UNSUPPORTED: "bg-red-100 text-red-800",
  CONTESTED: "bg-yellow-100 text-yellow-800",
  UNCERTAIN: "bg-gray-100 text-gray-800",
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
          <Badge className={verdictColors[verdict] || "bg-gray-100 text-gray-800"}>
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
