const states = [
  "PENDING",
  "RESEARCHING",
  "FACT_CHECKING_RESEARCH",
  "SCRIPTING",
  "FACT_CHECKING_SCRIPT",
  "ASSET_GENERATION",
  "COMPLETED",
] as const;

interface StateMachineProgressProps {
  currentStatus: string;
}

export function StateMachineProgress({
  currentStatus,
}: StateMachineProgressProps) {
  const currentIndex = states.indexOf(
    currentStatus as (typeof states)[number]
  );

  return (
    <div className="flex items-center gap-1">
      {states.map((state, index) => (
        <div key={state} className="flex items-center">
          <div
            className={`h-2 w-8 rounded-full ${
              index <= currentIndex
                ? "bg-primary"
                : "bg-muted"
            }`}
            title={state.replace(/_/g, " ")}
          />
        </div>
      ))}
    </div>
  );
}
