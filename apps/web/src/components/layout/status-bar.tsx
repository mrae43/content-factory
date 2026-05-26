"use client";

import { useIsFetching, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

function formatTimeAgo(ms: number): string {
  const secs = Math.round(ms / 1000);
  if (secs < 5) return "just now";
  if (secs < 60) return `${secs}s ago`;
  return `${Math.round(secs / 60)}m ago`;
}

export function StatusBar() {
  const isFetching = useIsFetching();
  const queryClient = useQueryClient();
  const [lastUpdate, setLastUpdate] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
      if (event.type === "updated" && event.query.state.dataUpdatedAt) {
        setLastUpdate((prev) =>
          event.query.state.dataUpdatedAt > prev
            ? event.query.state.dataUpdatedAt
            : prev
        );
      }
      if (event.type === "updated") {
        if (event.query.state.status === "error") {
          setHasError(true);
        } else if (event.query.state.status === "success") {
          const anyError = queryClient
            .getQueryCache()
            .getAll()
            .some((q) => q.state.status === "error");
          setHasError(anyError);
        }
      }
    });
    return () => unsubscribe();
  }, [queryClient]);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const timeAgo = lastUpdate > 0 ? now - lastUpdate : 0;

  let status: "live" | "stalled" | "disconnected";
  if (hasError) {
    status = "disconnected";
  } else if (isFetching > 0 || timeAgo < 15000) {
    status = "live";
  } else {
    status = "stalled";
  }

  const statusConfig = {
    live: { dot: "bg-success animate-pulse", label: "Live" },
    stalled: { dot: "bg-warning", label: "Stalled" },
    disconnected: { dot: "bg-destructive", label: "Disconnected" },
  };

  const cfg = statusConfig[status];

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 flex h-8 items-center justify-between bg-muted px-4 text-[11px] leading-none text-muted-foreground">
      <span>
        {timeAgo > 0
          ? `Last updated: ${formatTimeAgo(timeAgo)}`
          : "No data"}
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className={cn("h-1.5 w-1.5 rounded-full", cfg.dot)} />
        {cfg.label}
      </span>
    </div>
  );
}
