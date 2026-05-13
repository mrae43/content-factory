import { Badge } from "@/components/ui/badge";

const formatBadgeStyles: Record<string, string> = {
  all: "bg-muted text-muted-foreground",
  video: "bg-destructive/15 text-destructive",
  blog: "bg-info/15 text-info",
  carousel: "bg-accent-purple/15 text-accent-purple",
};

export { formatBadgeStyles };

export function FormatBadge({ formatType }: { formatType: string | null }) {
  if (!formatType) return null;
  const color =
    formatBadgeStyles[formatType.toLowerCase()] || "bg-muted text-muted-foreground";
  return (
    <Badge className={color}>
      {formatType.charAt(0).toUpperCase() + formatType.slice(1).toLowerCase()}
    </Badge>
  );
}
