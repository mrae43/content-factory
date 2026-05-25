"use client";

import type { ReactNode } from "react";
import type { JobStatusEnum } from "@content-factory/shared-types";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

function defaultTabForStatus(status: JobStatusEnum): string {
  if (status === "COMPLETED") return "output";
  if (status === "HUMAN_REVIEW_NEEDED") return "review";
  return "trail";
}

function TabBar({
  status,
  children,
}: {
  status: JobStatusEnum;
  children: ReactNode;
}) {
  const needsReview = status === "HUMAN_REVIEW_NEEDED";

  return (
    <Tabs defaultValue={defaultTabForStatus(status)}>
      <TabsList variant="line">
        <TabsTrigger value="output" className="font-heading text-base">
          Output
        </TabsTrigger>
        <TabsTrigger value="trail" className="font-heading text-base">
          Trail
        </TabsTrigger>
        <TabsTrigger
          value="review"
          className={`font-heading text-base ${needsReview ? "text-warning" : ""}`}
        >
          Review{" "}
          {needsReview ? (
            <span className="inline-flex items-center gap-1.5 text-warning">
              <span className="h-1.5 w-1.5 rounded-full bg-warning" />
              (1)
            </span>
          ) : (
            <span className="text-muted-foreground">(0)</span>
          )}
        </TabsTrigger>
      </TabsList>
      {children}
    </Tabs>
  );
}

function Output({ children }: { children: ReactNode }) {
  return <TabsContent value="output">{children}</TabsContent>;
}

function Trail({ children }: { children: ReactNode }) {
  return <TabsContent value="trail">{children}</TabsContent>;
}

function Review({ children }: { children: ReactNode }) {
  return <TabsContent value="review">{children}</TabsContent>;
}

TabBar.Output = Output;
TabBar.Trail = Trail;
TabBar.Review = Review;

export { TabBar };
