# End-to-End Integration — eval contract (8.4 Latency Budget)

## What this eval guards
Each pipeline stage must complete within its latency budget. Slow stages
create a poor user experience and may indicate performance regressions.

## Assertions
- Research Desk: P50 < 15s, P95 < 45s
- Retrieval Desk: P50 < 20s, P95 < 60s
- Writer's Desk: P50 < 30s, P95 < 90s
- Fact-Check Desk (per cycle): P50 < 20s, P95 < 60s
- Layout Desk: P50 < 15s, P95 < 45s
- Full pipeline (no Visual Assets): P50 < 3 min, P95 < 8 min

## Failure modes covered
None directly in the taxonomy — performance monitoring

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 8.4
