# Fact-Check Desk — eval contract (4.4 Evidence Traceability)

## What this eval guards
Every non-UNCERTAIN verdict must be backed by at least one evidence reference
to a valid Research Chunk. Without traceable evidence, fact-check verdicts
are unverifiable.

## Assertions
- Every non-UNCERTAIN verdict carries ≥ 1 `evidence_references` chunk ID (100%)
- All `evidence_references` IDs exist in pgvector (100%)
- Agent retrieves from Citation Index rather than re-querying pgvector
  (verified via trace)
- UNCERTAIN claims carry hedged `evidence_text` (not empty) (100%)

## Failure modes covered
F-V2 (broken citation chain propagates to blind fact-checking), F-F1
(unverifiable SUPPORTED)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 4.4
