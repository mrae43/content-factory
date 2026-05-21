# Pipeline Status Transitions — eval contract (7.2 Terminal States)

## What this eval guards
The pipeline must reach the correct terminal status for each scenario:
COMPLETED on success, FAILED on unrecoverable error, HUMAN_REVIEW_NEEDED
when the Fact-Check Loop exhausts or High strictness demands it.

## Assertions
- All desks succeed → COMPLETED
- Unrecoverable error at any desk → FAILED (test each desk independently)
- Fact-Check Loop exhausted → HUMAN_REVIEW_NEEDED
- High strictness + all-SUPPORTED → HUMAN_REVIEW_NEEDED
- Human resolves review → resumes from correct next desk

## Failure modes covered
F-P2

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 7.2
