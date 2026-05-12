import { Badge } from "@/components/ui/badge";

const formatBadgeColors: Record<string, string> = {
  all: "bg-gray-100 text-gray-800",
  video: "bg-red-100 text-red-800",
  blog: "bg-blue-100 text-blue-800",
  carousel: "bg-purple-100 text-purple-800",
};

export { formatBadgeColors };

export function FormatBadge({ formatType }: { formatType: string | null }) {
  if (!formatType) return null;
  const color =
    formatBadgeColors[formatType.toLowerCase()] || "bg-gray-100 text-gray-800";
  return (
    <Badge className={color}>
      {formatType.charAt(0).toUpperCase() + formatType.slice(1).toLowerCase()}
    </Badge>
  );
}
