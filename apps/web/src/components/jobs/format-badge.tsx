interface FormatBadgeProps {
  formatType: string | null;
}

export function FormatBadge({ formatType }: FormatBadgeProps) {
  if (!formatType) return null;

  const label = formatType.charAt(0).toUpperCase() + formatType.slice(1).toLowerCase();

  return (
    <span className="inline-flex items-center rounded-[4px] bg-primary/10 px-1.5 py-0.5 text-[0.6875rem] font-semibold uppercase tracking-[0.05em] text-primary">
      {label}
    </span>
  );
}

export function FormatBadges({ formatType, platform }: { formatType: string | null; platform?: string | null }) {
  return (
    <span className="inline-flex items-center gap-1">
      <FormatBadge formatType={formatType} />
      {platform && (
        <span className="inline-flex items-center rounded-[4px] bg-muted px-1.5 py-0.5 text-[0.6875rem] font-medium uppercase tracking-[0.05em] text-muted-foreground">
          {platform}
        </span>
      )}
    </span>
  );
}
