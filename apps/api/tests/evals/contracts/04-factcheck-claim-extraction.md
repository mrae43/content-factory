# Fact-Check Desk — eval contract (4.1 Claim Extraction Quality)

## What this eval guards
The Red Team must extract all atomic, verifiable claims from a script without
including meta-commentary or duplicates. Missed claims let false statements
reach users; false positives waste fact-checking cycles.

## Assertions
- Recall: ≥ 0.90 of human-labelled atomic claims extracted by the agent
- Precision: ≥ 0.85 of extracted claims are legitimate atomic claims
- Category accuracy: ≥ 0.80 of extracted claims assigned correct `category`
- Compound claims ("X happened in 2019 and caused Y") must split into two
  atomic claims
- Implicit claims ("the leading provider") must be extracted
- Attribution claims ("according to X, Y is true") → `category=attribution`

## Failure modes covered
F-F1 (missed false claim), F-F2 (over-rejection from bad claim parsing)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 4.1
