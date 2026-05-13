import { Skeleton } from "@/components/ui/skeleton";

export function JobCardSkeleton() {
  return (
    <div className="flex items-center justify-between p-4 rounded-lg border">
      <div className="flex items-center gap-3">
        <div className="space-y-2">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-3 w-24" />
        </div>
        <Skeleton className="h-5 w-16 rounded-full" />
      </div>
      <Skeleton className="h-5 w-24 rounded-full" />
    </div>
  );
}
