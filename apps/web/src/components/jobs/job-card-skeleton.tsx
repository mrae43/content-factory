import { Skeleton } from "@/components/ui/skeleton";

export function JobCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4 shadow-[0_1px_2px_rgba(31,28,24,0.04)]">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <Skeleton className="h-5 w-56" />
          <div className="mt-2 flex items-center gap-2">
            <Skeleton className="h-2 w-2 rounded-full" />
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-3 w-16" />
          </div>
          <div className="mt-2.5 flex gap-1.5">
            {[1, 2, 3, 4, 5, 6, 7].map((d) => (
              <Skeleton key={d} className="h-2 w-2 rounded-full" />
            ))}
          </div>
        </div>
        <Skeleton className="h-4 w-14 rounded-[4px]" />
      </div>
    </div>
  );
}
