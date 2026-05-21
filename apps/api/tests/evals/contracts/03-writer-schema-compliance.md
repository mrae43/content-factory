# Writer's Desk — eval contract (3.2 Schema Compliance)

## What this eval guards
The script schema must be correctly initialised: role, version, and content
must meet the expected contract on first draft.

## Assertions
- Role set correctly: `script.role == "master"` for first draft (100%)
- Version initialised: `script.version == 1` for initial draft (100%)
- Script not empty: `len(script.content) > 0` (100%)
- Directives consumed: Story Directives fields referenced in the scripting
  agent's prompt (verified via prompt template audit)

## Failure modes covered
F-P2 (wrong terminal state from schema corruption)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 3.2
