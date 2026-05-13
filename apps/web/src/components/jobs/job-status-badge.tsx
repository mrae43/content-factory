import { Badge } from "@/components/ui/badge";

const statusColors: Record<string, string> = {
  PENDING: "bg-yellow-100 text-amber-900",
  RESEARCHING: "bg-blue-100 text-blue-800",
  FACT_CHECKING_RESEARCH: "bg-blue-100 text-blue-800",
  SCRIPTING: "bg-purple-100 text-purple-800",
  FACT_CHECKING_SCRIPT: "bg-orange-100 text-orange-800",
  FORMATTING: "bg-teal-100 text-teal-800",
  ASSET_GENERATION: "bg-indigo-100 text-indigo-800",
  COMPLETED: "bg-green-100 text-green-800",
  HUMAN_REVIEW_NEEDED: "bg-amber-100 text-amber-800",
  FAILED: "bg-red-100 text-red-800",
};

const statusLabels: Record<string, string> = {
  PENDING: "Queued",
  RESEARCHING: "Researching",
  FACT_CHECKING_RESEARCH: "Verifying Research",
  SCRIPTING: "Writing Script",
  FACT_CHECKING_SCRIPT: "Checking Script",
  FORMATTING: "Formatting",
  ASSET_GENERATION: "Generating Assets",
  COMPLETED: "Completed",
  HUMAN_REVIEW_NEEDED: "Needs Your Review",
  FAILED: "Failed",
};

interface JobStatusBadgeProps {
  status: string;
}

export function JobStatusBadge({ status }: JobStatusBadgeProps) {
  return (
    <Badge
      className={statusColors[status] || "bg-gray-100 text-gray-800"}
      aria-label={`Status: ${statusLabels[status] || status}`}
    >
      {statusLabels[status] || status.replace(/_/g, " ")}
    </Badge>
  );
}
