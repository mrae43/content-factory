# Fact-Check Loop — eval contract (5.3 Guardrail Strictness Compliance)

## What this eval guards
Each guardrail strictness level (Low/Medium/High) must apply the correct
similarity threshold, UNCERTAIN handling, and terminal routing. Passing
at Medium does not guarantee Low or High behaviour.

## Assertions
- Low (threshold 0.65): UNCERTAIN passes, all-SUPPORTED auto-advances,
  max_cycles=2
- Medium (threshold 0.72): UNCERTAIN passes, all-SUPPORTED auto-advances,
  max_cycles=3
- High (threshold 0.75): UNCERTAIN soft-fails → revision, all-SUPPORTED
  routes to human review, max_cycles=3
- Each level tested independently with a controlled script

## Failure modes covered
F-L3

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 5.3
