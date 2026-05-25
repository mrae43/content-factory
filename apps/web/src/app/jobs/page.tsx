"use client";

import { useJobs } from "@/hooks/use-jobs";
import { FormatBadge } from "@/components/jobs/format-badge";
import { StatusDot } from "@/components/jobs/status-dot";
import { MiniPipeline } from "@/components/jobs/mini-pipeline";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { useUIStore } from "@/stores/ui-store";
import Link from "next/link";
import { pipelineStages, deskLabels } from "@/lib/constants/pipeline";

type EditorialFilter = "all" | "active" | "published" | "review" | "killed";

const editorialFilters: {
  key: EditorialFilter;
  label: string;
  statuses: string[];
}[] = [
  {
    key: "all",
    label: "All",
    statuses: [],
  },
  {
    key: "active",
    label: "Active",
    statuses: [
      "PENDING",
      "RESEARCHING",
      "RETRIEVAL",
      "SCRIPTING",
      "FACT_CHECKING_SCRIPT",
      "FORMATTING",
      "ASSET_GENERATION",
    ],
  },
  {
    key: "published",
    label: "Published",
    statuses: ["COMPLETED"],
  },
  {
    key: "review",
    label: "Review",
    statuses: ["HUMAN_REVIEW_NEEDED"],
  },
  {
    key: "killed",
    label: "Killed",
    statuses: ["FAILED"],
  },
];

function relativeTime(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diff = now - then;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} hour${hours > 1 ? "s" : ""} ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} day${days > 1 ? "s" : ""} ago`;
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function StoryListSkeleton() {
  return (
    <div className="space-y-1.5 sm:space-y-2">
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          className="rounded-lg border border-border bg-card p-3 sm:p-4"
        >
          <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3">
            <div className="min-w-0 flex-1">
              <Skeleton className="h-5 w-40 sm:w-56" />
              <div className="mt-2 flex items-center gap-2">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-16" />
              </div>
              <div className="mt-2.5 flex gap-1.5">
                {[1, 2, 3, 4, 5, 6, 7].map((d) => (
                  <Skeleton key={d} className="h-2 w-2 rounded-full" />
                ))}
              </div>
            </div>
            <Skeleton className="h-4 w-14 shrink-0 rounded-[4px]" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function JobsPage() {
  const { data: jobs, isLoading, isError, error } = useJobs();
  const { selectedJobFilter, setJobFilter } = useUIStore();

  const filterKey = (selectedJobFilter === "all"
    ? "all"
    : editorialFilters.find((f) => f.statuses.includes(selectedJobFilter))?.key ?? "all") as EditorialFilter;

  const setEditorialFilter = (key: EditorialFilter) => {
    const filter = editorialFilters.find((f) => f.key === key);
    if (!filter) return;
    setJobFilter(filter.key === "all" ? "all" : filter.statuses[0]);
  };

  const filtered =
    filterKey === "all"
      ? jobs
      : jobs?.filter((j) => {
          const filter = editorialFilters.find((f) => f.key === filterKey);
          return filter ? filter.statuses.includes(j.status) : true;
        });

  const counts: Record<EditorialFilter, number> = {
    all: jobs?.length ?? 0,
    active:
      jobs?.filter((j) =>
        [
          "PENDING",
          "RESEARCHING",
          "RETRIEVAL",
          "SCRIPTING",
          "FACT_CHECKING_SCRIPT",
          "FORMATTING",
          "ASSET_GENERATION",
        ].includes(j.status)
      ).length ?? 0,
    published:
      jobs?.filter((j) => j.status === "COMPLETED").length ?? 0,
    review:
      jobs?.filter((j) => j.status === "HUMAN_REVIEW_NEEDED").length ?? 0,
    killed: jobs?.filter((j) => j.status === "FAILED").length ?? 0,
  };

  return (
    <div className="space-y-4 sm:space-y-6">
      <div
        className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1 sm:mx-0 sm:flex-wrap sm:overflow-visible sm:px-0 sm:pb-0"
        role="group"
        aria-label="Filter stories"
      >
        {editorialFilters.map((filter) => (
          <button
            key={filter.key}
            onClick={() => setEditorialFilter(filter.key)}
            aria-pressed={filterKey === filter.key}
            className={`shrink-0 rounded-[4px] px-3 py-2 text-fluid-xs font-medium transition-colors sm:py-1.5 ${
              filterKey === filter.key
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            }`}
          >
            {filter.label}{" "}
            <span className="ml-0.5 opacity-70">
              {counts[filter.key]}
            </span>
          </button>
        ))}
      </div>

      {isLoading ? (
        <StoryListSkeleton />
      ) : isError ? (
        <p className="text-fluid-sm text-destructive">
          Failed to load jobs: {error?.message}
        </p>
      ) : (
        <div className="space-y-1.5 sm:space-y-2">
          {filtered?.map((job) => {
            const isActive = ![
              "COMPLETED",
              "FAILED",
              "HUMAN_REVIEW_NEEDED",
            ].includes(job.status);

            return (
              <Link key={job.id} href={`/jobs/${job.id}`} className="block min-h-[44px]">
                <Card className="border border-border bg-card shadow-[0_1px_2px_rgba(31,28,24,0.04)] transition-colors hover:bg-accent">
                  <CardContent className="p-3 sm:p-4">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3">
                      <div className="min-w-0 flex-1">
                        <p className="font-heading text-fluid-base font-semibold leading-snug text-foreground">
                          {job.title}
                        </p>
                        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5">
                          <StatusDot status={job.status} />
                          <span className="text-fluid-xs text-muted-foreground">
                            {relativeTime(job.updated_at)}
                          </span>
                        </div>
                        {isActive && (
                          <div className="mt-2.5 sm:mt-3">
                            <Tooltip>
                              <TooltipTrigger>
                                <span>
                                  <MiniPipeline
                                    currentStatus={job.status}
                                    formatType={job.format_type}
                                  />
                                </span>
                              </TooltipTrigger>
                              <TooltipContent side="top" align="center">
                                {(() => {
                                  const idx = pipelineStages.indexOf(job.status as typeof pipelineStages[number]);
                                  const label = idx >= 0 ? deskLabels[job.status] : job.status;
                                  return `${label} \u00B7 Step ${idx + 1} of ${pipelineStages.length}`;
                                })()}
                              </TooltipContent>
                            </Tooltip>
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-2 sm:flex-col sm:items-end sm:gap-1.5">
                        <FormatBadge formatType={job.format_type} />
                        {job.platform && (
                          <span className="text-fluid-xs font-medium uppercase tracking-[0.05em] text-muted-foreground">
                            {job.platform}
                          </span>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
          {filtered?.length === 0 && (
            <Card className="border border-border bg-card shadow-[0_1px_2px_rgba(31,28,24,0.04)]">
              <CardContent className="flex flex-col items-center px-6 py-10 text-center sm:py-12">
                <p className="font-heading text-fluid-sm font-semibold text-foreground">
                  No stories match this filter.
                </p>
                <p className="mt-1 text-fluid-xs text-muted-foreground">
                  Try a different filter or commission a new story.
                </p>
                {filterKey !== "all" && (
                  <Button
                    variant="link"
                    onClick={() => setEditorialFilter("all")}
                    className="mt-3 min-h-[44px] text-primary"
                  >
                    Clear filter
                  </Button>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
