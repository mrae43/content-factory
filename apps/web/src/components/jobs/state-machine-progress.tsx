"use client";

const states = [
  "PENDING",
  "RESEARCHING",
  "FACT_CHECKING_RESEARCH",
  "SCRIPTING",
  "FACT_CHECKING_SCRIPT",
  "FORMATTING",
  "ASSET_GENERATION",
  "COMPLETED",
] as const;

interface StateMachineProgressProps {
  currentStatus: string;
  formatType?: string | null;
}

export function StateMachineProgress({
  currentStatus,
  formatType,
}: StateMachineProgressProps) {
  const currentIndex = states.indexOf(
    currentStatus as (typeof states)[number]
  );
  const fmt = (formatType ?? "all").toLowerCase();
  const skipAssetGeneration = fmt === "blog" || fmt === "carousel";

  return (
    <div className="flex items-center gap-1">
      {states.map((state, index) => {
        const isSkipped =
          state === "ASSET_GENERATION" &&
          skipAssetGeneration &&
          currentIndex >= states.indexOf("FORMATTING");

        return (
          <div key={state} className="flex items-center">
            <div
              className={`h-2 rounded-full ${
                isSkipped
                  ? "w-4 bg-muted-foreground/20 opacity-40"
                  : index <= currentIndex
                    ? "bg-primary w-8"
                    : "bg-muted w-8"
              }`}
              title={
                isSkipped
                  ? `${state.replace(/_/g, " ")} (skipped)`
                  : state.replace(/_/g, " ")
              }
            />
          </div>
        );
      })}
    </div>
  );
}
