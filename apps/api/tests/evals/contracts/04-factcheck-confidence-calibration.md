# Fact-Check Desk — eval contract (4.3 Confidence Calibration)

## What this eval guards
The confidence scores attached to fact-check verdicts must be well-calibrated.
Over-confident false SUPPORTED verdicts are dangerous; under-confident true
verdicts cause unnecessary loop iterations.

## Assertions
- For 200 (claim, verdict, confidence) triples with known ground truth:
- Expected calibration error (ECE): ≤ 0.10
- High-confidence accuracy (confidence ≥ 0.85): ≥ 0.90
- Low-confidence accuracy (confidence ≤ 0.50): may be < 0.65 (high
  uncertainty is the correct behaviour)

## Failure modes covered
F-F1 (over-confidence masking a false SUPPORTED), F-F2 (over-rejection
from under-confidence)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 4.3
