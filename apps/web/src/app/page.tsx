"use client";

import { useJobs } from "@/hooks/use-jobs";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { FormatBadge } from "@/components/jobs/format-badge";
import { StatusDot } from "@/components/jobs/status-dot";
import { MiniPipeline } from "@/components/jobs/mini-pipeline";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { useState } from "react";

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

function SectionHeading({
  children,
  count,
}: {
  children: React.ReactNode;
  count?: number;
}) {
  return (
    <div className="flex items-baseline gap-2">
      <h3 className="font-heading text-[1.5rem] font-semibold text-foreground">
        {children}
      </h3>
      {count !== undefined && (
        <span className="text-[0.8125rem] font-medium text-muted-foreground">
          ({count})
        </span>
      )}
    </div>
  );
}

function StoryCard({
  job,
}: {
  job: {
    id: string;
    topic: string;
    status: string;
    format_type: string | null;
    platform?: string | null;
    created_at: string;
    updated_at: string;
  };
}) {
  return (
    <Link href={`/jobs/${job.id}`}>
      <Card className="border border-border bg-card shadow-[0_1px_2px_rgba(31,28,24,0.04)] transition-colors hover:bg-accent">
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="font-heading text-[1.0625rem] font-semibold leading-snug text-foreground">
                {job.topic}
              </p>
              <div className="mt-1.5 flex items-center gap-2">
                <StatusDot status={job.status} />
                <span className="text-[0.75rem] text-muted-foreground">
                  {relativeTime(job.updated_at)}
                </span>
              </div>
              <div className="mt-2.5">
                <MiniPipeline
                  currentStatus={job.status}
                  formatType={job.format_type}
                />
              </div>
            </div>
            <div className="flex shrink-0 flex-col items-end gap-1.5">
              <FormatBadge formatType={job.format_type} />
              {job.platform && (
                <span className="text-[0.625rem] font-medium uppercase tracking-[0.05em] text-muted-foreground">
                  {job.platform}
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

function PublishedCard({
  job,
}: {
  job: {
    id: string;
    topic: string;
    status: string;
    format_type: string | null;
    platform?: string | null;
    created_at: string;
    updated_at: string;
  };
}) {
  return (
    <Link href={`/jobs/${job.id}`}>
      <Card className="border border-border bg-card shadow-[0_1px_2px_rgba(31,28,24,0.04)] transition-colors hover:bg-accent">
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="font-heading text-[1.0625rem] font-semibold leading-snug text-foreground">
                {job.topic}
              </p>
              <div className="mt-1.5 flex items-center gap-2">
                <StatusDot status={job.status} />
                <span className="text-[0.75rem] text-muted-foreground">
                  {relativeTime(job.updated_at)}
                </span>
              </div>
            </div>
            <FormatBadge formatType={job.format_type} />
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

function KilledCard({
  job,
}: {
  job: {
    id: string;
    topic: string;
    status: string;
    format_type: string | null;
    created_at: string;
    updated_at: string;
    error_log: Record<string, unknown> | null;
  };
}) {
  const [expanded, setExpanded] = useState(false);
  const errorMsg =
    job.error_log &&
    typeof job.error_log === "object" &&
    "error" in job.error_log
      ? String(job.error_log.error)
      : null;

  return (
    <Card className="border border-border bg-card shadow-[0_1px_2px_rgba(31,28,24,0.04)]">
      <CardContent className="p-4">
        <Link href={`/jobs/${job.id}`}>
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="font-heading text-[1.0625rem] font-semibold leading-snug text-foreground">
                {job.topic}
              </p>
              <div className="mt-1.5 flex items-center gap-2">
                <StatusDot status={job.status} />
                <span className="text-[0.75rem] text-muted-foreground">
                  {relativeTime(job.updated_at)}
                </span>
              </div>
            </div>
            <FormatBadge formatType={job.format_type} />
          </div>
        </Link>
        {errorMsg && (
          <div className="mt-2">
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-[0.75rem] font-medium text-primary transition-colors hover:text-primary/80"
            >
              {expanded ? "Hide details" : "Show details"}
            </button>
            {expanded && (
              <p className="mt-1.5 rounded-md bg-muted p-2.5 font-mono text-[0.8125rem] text-destructive">
                {errorMsg}
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-10">
      <div className="space-y-3">
        <Skeleton className="h-6 w-36" />
        {[1, 2].map((i) => (
          <div
            key={i}
            className="rounded-lg border border-border bg-card p-4"
          >
            <Skeleton className="h-5 w-64" />
            <div className="mt-2 flex items-center gap-2">
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-3 w-16" />
            </div>
            <div className="mt-3 flex gap-1.5">
              {[1, 2, 3, 4, 5, 6, 7].map((d) => (
                <Skeleton key={d} className="h-2 w-2 rounded-full" />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="space-y-3">
        <Skeleton className="h-6 w-48" />
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="rounded-lg border border-border bg-card p-4"
          >
            <Skeleton className="h-5 w-56" />
            <div className="mt-2 flex items-center gap-2">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-3 w-24" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HomePage() {
  const [showKilled, setShowKilled] = useState(false);
  const { data: jobs, isLoading, isError, error } = useJobs();

  if (isLoading) {
    return <DashboardSkeleton />;
  }

  if (isError) {
    return (
      <div className="space-y-6">
        <h2 className="font-heading text-[2rem] font-bold text-foreground">
          Overview
        </h2>
        <p className="text-sm text-destructive">
          Failed to load jobs: {error?.message}
        </p>
      </div>
    );
  }

  const activeJobs = jobs?.filter(
    (j) =>
      !["COMPLETED", "FAILED", "HUMAN_REVIEW_NEEDED"].includes(j.status)
  ) ?? [];

  const reviewJobs =
    jobs?.filter((j) => j.status === "HUMAN_REVIEW_NEEDED") ?? [];

  const publishedJobs =
    jobs?.filter((j) => j.status === "COMPLETED") ?? [];

  const killedJobs = jobs?.filter((j) => j.status === "FAILED") ?? [];


  if (!jobs || jobs.length === 0) {
    return (
      <div className="flex items-center justify-center py-24">
        <Card className="max-w-sm border border-border bg-card shadow-[0_1px_2px_rgba(31,28,24,0.04)]">
          <CardContent className="flex flex-col items-center py-12 text-center">
            <p className="font-heading text-[1.125rem] font-semibold text-foreground">
              No stories yet.
            </p>
            <p className="mt-1.5 text-[0.875rem] text-muted-foreground">
              Commission your first piece.
            </p>
            <Link href="/jobs/new" className="mt-6">
              <Button className="bg-primary font-heading font-semibold text-primary-foreground hover:bg-primary/90">
                Commission
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-10">
      {(activeJobs.length > 0 || reviewJobs.length > 0) && (
        <section className="space-y-3">
          <SectionHeading count={activeJobs.length + reviewJobs.length}>
            Active Stories
          </SectionHeading>
          <div className="space-y-2">
            {reviewJobs.map((job) => (
              <StoryCard key={job.id} job={job} />
            ))}
            {activeJobs.map((job) => (
              <StoryCard key={job.id} job={job} />
            ))}
          </div>
        </section>
      )}

      {publishedJobs.length > 0 && (
        <section className="space-y-3">
          <SectionHeading count={publishedJobs.length}>
            Recently Published
          </SectionHeading>
          <div className="space-y-2">
            {publishedJobs.map((job) => (
              <PublishedCard key={job.id} job={job} />
            ))}
          </div>
        </section>
      )}

      {killedJobs.length > 0 && (
        <section className="space-y-3">
          <SectionHeading count={killedJobs.length}>
            Killed Stories
          </SectionHeading>
          {!showKilled ? (
            <button
              onClick={() => setShowKilled(true)}
              className="text-[0.8125rem] font-medium text-muted-foreground transition-colors hover:text-primary"
            >
              Show {killedJobs.length} killed{" "}
              {killedJobs.length === 1 ? "story" : "stories"}
            </button>
          ) : (
            <div className="space-y-2">
              {killedJobs.map((job) => (
                <KilledCard key={job.id} job={job} />
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
