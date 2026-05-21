# End-to-End Integration — eval contract (8.1 Story Lifecycle)

## What this eval guards
The full pipeline must complete successfully for a diverse set of canonical
topics within a reasonable timeout, with no stories stalling indefinitely.

## Assertions
- Run 10 canonical topics (diverse domains, edge-case inputs) through full
  pipeline
- All stories reach a terminal state
- No stories stall indefinitely (timeout: 10 min per story)
- COMPLETED rate on clean inputs: ≥ 80%

## Failure modes covered
F-P1, F-P2, F-L1 (indirectly — stalled loop causes timeout)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 8.1
