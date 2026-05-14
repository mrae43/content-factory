interface MiniPipelineProps {
  currentStatus: string;
  formatType?: string | null;
}

const pipelineStages = [
  "PENDING",
  "RESEARCHING",
  "FACT_CHECKING_RESEARCH",
  "SCRIPTING",
  "FACT_CHECKING_SCRIPT",
  "FORMATTING",
  "ASSET_GENERATION",
] as const;

const deskLabels: Record<string, string> = {
  PENDING: "Queued",
  RESEARCHING: "Research",
  FACT_CHECKING_RESEARCH: "Verify",
  SCRIPTING: "Writing",
  FACT_CHECKING_SCRIPT: "Fact-Check",
  FORMATTING: "Layout",
  ASSET_GENERATION: "Assets",
};

export function MiniPipeline({ currentStatus, formatType }: MiniPipelineProps) {
  const fmt = (formatType ?? "all").toLowerCase();
  const skipAssetGeneration = fmt === "blog" || fmt === "carousel";
  const isTerminal =
    currentStatus === "COMPLETED" ||
    currentStatus === "FAILED" ||
    currentStatus === "HUMAN_REVIEW_NEEDED";

  const currentIndex = pipelineStages.indexOf(
    currentStatus as (typeof pipelineStages)[number]
  );

  const isCompleted = currentStatus === "COMPLETED";

  return (
    <div
      className="flex items-center gap-1.5"
      role="progressbar"
      aria-valuenow={currentIndex + 1}
      aria-valuemin={1}
      aria-valuemax={pipelineStages.length}
      aria-label={`Pipeline progress: step ${currentIndex + 1} of ${pipelineStages.length}`}
    >
      {pipelineStages.map((stage) => {
        const stageIndex = pipelineStages.indexOf(stage);
        const isSkipped =
          stage === "ASSET_GENERATION" &&
          skipAssetGeneration &&
          currentIndex >= pipelineStages.indexOf("FORMATTING");
        const isActive = stageIndex === currentIndex && !isSkipped && !isTerminal;
        const isDone = isCompleted || stageIndex < currentIndex;

        if (isSkipped) {
          return (
            <span
              key={stage}
              className="inline-block h-1.5 w-1.5 rounded-full bg-muted"
              title={`${deskLabels[stage]} (skipped)`}
            />
          );
        }

        return (
          <span
            key={stage}
            className={`inline-block rounded-full ${
              isDone
                ? "h-2 w-2 bg-success"
                : isActive
                  ? "h-2 w-2 bg-primary animate-pulse"
                  : "h-2 w-2 bg-border"
            }`}
            title={deskLabels[stage]}
          />
        );
      })}
    </div>
  );
}
