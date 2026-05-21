# Fact-Check Desk — eval contract (4.2 Verdict Accuracy)

## What this eval guards
The Fact-Check Desk must not pass false claims as SUPPORTED, and must not
reject true claims as UNSUPPORTED at a rate that causes loop exhaustion
on clean scripts. This is the highest-stakes eval surface.

## Assertions
- A claim with no supporting chunk must never receive SUPPORTED
- Planted false claims must be detected (UNSUPPORTED or CONTESTED) at
  ≥ 90% recall; false pass rate (SUPPORTED on planted false claim): 0%
- Overall verdict accuracy on 100 (claim, evidence_chunks) pairs: ≥ 0.82
- SUPPORTED precision: ≥ 0.88 (false SUPPORTED is the most dangerous error)
- UNSUPPORTED recall: ≥ 0.80 (missing an unsupported claim lets it through)
- UNCERTAIN vs CONTESTED accuracy: ≥ 0.70 (this distinction is genuinely hard)
- Adversarial: inject 5 known-false verifiable claims into 20 test scripts

## Failure modes covered
F-F1, F-F2

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 4.2
