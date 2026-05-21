# Pipeline Status Transitions — eval contract (7.3 Status Idempotency)

## What this eval guards
Re-processing a story from any non-terminal status must produce the same
terminal result (within acceptable LLM variance). No story should flip
between terminal states on re-run.

## Assertions
- Re-processing from any non-terminal status produces the same terminal
  result (within acceptable stochastic variance)
- No story flips between terminal states on re-run

## Failure modes covered
F-P2

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 7.3
