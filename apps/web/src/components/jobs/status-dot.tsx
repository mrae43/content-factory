interface StatusDotProps {
  status: string;
  className?: string;
}

const statusConfig: Record<string, { color: string; label: string }> = {
  PENDING: { color: "bg-muted-foreground", label: "Queued" },
  RESEARCHING: { color: "bg-warning", label: "Research Desk" },
  FACT_CHECKING_RESEARCH: { color: "bg-warning", label: "Source Verification" },
  SCRIPTING: { color: "bg-info", label: "Writer's Desk" },
  FACT_CHECKING_SCRIPT: { color: "bg-info", label: "Fact-Check Desk" },
  FORMATTING: { color: "bg-accent-purple", label: "Layout Desk" },
  ASSET_GENERATION: { color: "bg-accent-teal", label: "Production" },
  COMPLETED: { color: "bg-success", label: "Published" },
  FAILED: { color: "bg-destructive", label: "Killed" },
  HUMAN_REVIEW_NEEDED: { color: "bg-warning", label: "Your Review" },
};

export function StatusDot({ status, className = "" }: StatusDotProps) {
  const config = statusConfig[status] ?? {
    color: "bg-muted-foreground",
    label: status.replace(/_/g, " "),
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 ${className}`}
      aria-label={`Status: ${config.label}`}
    >
    <span
      className={`inline-block h-2 w-2 rounded-full ${config.color}`}
    />
      <span className="text-[0.6875rem] font-medium text-muted-foreground">
        {config.label}
      </span>
    </span>
  );
}

export function getStatusLabel(status: string): string {
  return statusConfig[status]?.label ?? status.replace(/_/g, " ");
}
