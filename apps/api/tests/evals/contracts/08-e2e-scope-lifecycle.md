# End-to-End Integration — eval contract (8.3 Scope Lifecycle)

## What this eval guards
After a story completes, LOCAL-scope chunks (web search results) must be
cleaned up to avoid polluting future research. User-provided RAW-CONTEXT
chunks must be retained if the retention policy requires it.

## Assertions
- After COMPLETED: all `scope=LOCAL` chunks for that story cleaned up
  (deleted or marked inactive)
- `scope=RAW-CONTEXT` chunks from user-provided material retained if
  specified by Story's retention policy

## Failure modes covered
None directly in the taxonomy — data lifecycle hygiene

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 8.3
