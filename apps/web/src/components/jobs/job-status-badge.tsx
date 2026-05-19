import { Badge } from "@/components/ui/badge";

const statusStyles: Record<string, string> = {
  PENDING: "bg-warning/15 text-warning",
  RESEARCHING: "bg-info/15 text-info",
  RETRIEVAL: "bg-info/15 text-info",
  SCRIPTING: "bg-accent-purple/15 text-accent-purple",
  FACT_CHECKING_SCRIPT: "bg-accent-orange/15 text-accent-orange",
  FORMATTING: "bg-accent-teal/15 text-accent-teal",
  ASSET_GENERATION: "bg-accent-indigo/15 text-accent-indigo",
  COMPLETED: "bg-success/15 text-success",
  HUMAN_REVIEW_NEEDED: "bg-warning/15 text-warning",
  FAILED: "bg-destructive/15 text-destructive",
};

const statusLabels: Record<string, string> = {
  PENDING: "Queued",
  RESEARCHING: "Researching",
  RETRIEVAL: "Retrieving Evidence",
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
      className={statusStyles[status] || "bg-muted text-muted-foreground"}
      aria-label={`Status: ${statusLabels[status] || status}`}
    >
      {statusLabels[status] || status.replace(/_/g, " ")}
    </Badge>
  );
}
