# Fact-Check Loop — eval contract (5.1 Remediation Effectiveness)

## What this eval guards
The Script Optimizer must surgically patch only the claims that failed
fact-checking, without collateral damage to supported claims, while
preserving narrative flow.

## Assertions
- Targeted patching: only claims with UNSUPPORTED or CONTESTED verdicts
  are modified — 0 unintended edits to SUPPORTED claims
- Fix rate per cycle: ≥ 70% of patched claims pass re-evaluation
- Regression rate: ≤ 5% of previously SUPPORTED claims newly fail after
  patching
- `remediation_depth` incremented by 1 per cycle (100%)
- LLM judge for surgical precision: ≥ 90% Yes/Yes on "only failed claim
  addressed?" and "minimum necessary change?"; criterion 3 (narrative flow)
  ≥ 4.0/5

## Failure modes covered
F-L2

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 5.1
