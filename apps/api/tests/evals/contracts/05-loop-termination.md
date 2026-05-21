# Fact-Check Loop — eval contract (5.2 Loop Termination)

## What this eval guards
The Fact-Check Loop must terminate correctly for every scenario: auto-advance
when clean, escalate when exhausted, and never exceed `max_cycles`. An infinite
loop locks the pipeline; premature termination lets false claims through.

## Assertions
- All claims SUPPORTED, strictness=Low or Medium: auto-advance to FORMATTING
- All claims SUPPORTED, strictness=High: route to HUMAN_REVIEW_NEEDED
  (assert status + reason field)
- UNCERTAIN claims present: apply hedged language, do not trigger revision
  (assert no Optimizer invocation)
- `remediation_depth >= max_cycles`: escalate to HUMAN_REVIEW_NEEDED
- UNSUPPORTED claim survives all cycles: escalate to HUMAN_REVIEW_NEEDED
- Loop ceiling regression: script with always-UNSUPPORTED claims terminates
  at `remediation_depth == max_cycles`, never exceeds it

## Failure modes covered
F-L1

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 5.2
