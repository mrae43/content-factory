# Pipeline Status Transitions — eval contract (7.1 Happy Path)

## What this eval guards
A successful story must traverse the full state machine in canonical order.
No valid story should skip a non-terminal stage.

## Assertions
- For each successful story, the status sequence is a valid prefix or
  completion of: PENDING → RESEARCHING → RETRIEVAL → SCRIPTING →
  FACT_CHECKING_SCRIPT → FORMATTING → ASSET_GENERATION → COMPLETED
- No valid story skips a non-terminal stage

## Failure modes covered
F-P1

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 7.1
