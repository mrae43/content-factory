export const pipelineStages = [
  "PENDING",
  "RESEARCHING",
  "RETRIEVAL",
  "SCRIPTING",
  "FACT_CHECKING_SCRIPT",
  "FORMATTING",
  "ASSET_GENERATION",
] as const;

export type PipelineStage = (typeof pipelineStages)[number];

export const deskLabels: Record<string, string> = {
  PENDING: "Queued",
  RESEARCHING: "Research",
  RETRIEVAL: "Retrieve",
  SCRIPTING: "Writing",
  FACT_CHECKING_SCRIPT: "Fact-Check",
  FORMATTING: "Layout",
  ASSET_GENERATION: "Assets",
};
