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

  useEffect(() => {
    const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
      if (event.type === "updated" && event.query.state.dataUpdatedAt) {
        setLastUpdate((prev) =>
          event.query.state.dataUpdatedAt > prev
            ? event.query.state.dataUpdatedAt
            : prev
        );
      }
    });
    return () => unsubscribe();
  }, [queryClient]);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const isLive = isFetching > 0;
  const timeAgo = lastUpdate > 0 ? now - lastUpdate : 0;

  return (
    <div
      className={cn(
        "flex items-center gap-2 text-[11px] leading-none transition-colors",
        isLive ? "text-primary" : "text-muted-foreground"
      )}
    >
      <span className="inline-flex items-center gap-1">
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            isLive
              ? "bg-primary animate-pulse"
              : "bg-muted-foreground/40"
          )}
        />
        {isLive ? "Live" : "Stale"}
      </span>
      {timeAgo > 0 && (
        <span className="text-muted-foreground/60">
          &middot; {formatTimeAgo(timeAgo)}
        </span>
      )}
    </div>
  );
}
