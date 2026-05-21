# Research Desk — eval contract (1.3 topic_relevance Calibration)

## What this eval guards
The pipeline's auto-assigned `topic_relevance` labels must agree with
human judgement. Mis-calibrated HIGH labels cause false positives in
synthesis quality; missed LOW labels pollute the context.

## Assertions
- Held-out set of 200 chunks with human-labelled relevance (HIGH/MEDIUM/LOW)
- Accuracy vs human labels: ≥ 0.80
- HIGH precision: ≥ 0.85 (false positives hurt synthesis quality)
- LOW recall: ≥ 0.75 (missed low-quality chunks pollute context)

## Failure modes covered
F-R1 (synthesis quality degradation from mis-labelled chunks)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 1.3
