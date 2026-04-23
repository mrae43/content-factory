"use client";

import { Badge } from "@/components/ui/badge";

const verdictColors: Record<string, string> = {
  SUPPORTED: "bg-green-100 text-green-800",
  UNSUPPORTED: "bg-red-100 text-red-800",
  CONTESTED: "bg-yellow-100 text-yellow-800",
  NOT_VERIFIABLE: "bg-gray-100 text-gray-800",
};

interface ClaimCardProps {
  claim_text: string;
  verdict: string;
  evidence: string;
  search_query: string;
}

export function ClaimCard({
  claim_text,
  verdict,
  evidence,
  search_query,
}: ClaimCardProps) {
  return (
    <div className="rounded-lg border p-4 space-y-2">
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm">{claim_text}</p>
        <Badge className={verdictColors[verdict] || "bg-gray-100 text-gray-800"}>
          {verdict}
        </Badge>
      </div>
      {evidence && (
        <p className="text-xs text-muted-foreground">{evidence}</p>
      )}
      {search_query && (
        <p className="text-xs text-muted-foreground italic">
          Query: {search_query}
        </p>
      )}
    </div>
  );
}
