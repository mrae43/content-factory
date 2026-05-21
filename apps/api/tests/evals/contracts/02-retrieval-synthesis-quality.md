# Retrieval Desk — eval contract (2.1 Synthesis — Refined Context Quality)

## What this eval guards
The Refined Context produced by the ResearchAgent must be a factually grounded,
coherent narrative that follows Story Directives and never hallucinates claims
unsupported by source chunks.

## Assertions
- Length compliance: 800–1500 words (rule-based)
- Factual grounding: every claim traceable to at least one provided chunk.
  Flagged claims must be < 5% of total claims
- Coherence: LLM judge score ≥ 4.0/5 for unified narrative
- Directive adherence: LLM judge score ≥ 3.5/5 for following tone/audience/angle
- No hallucination: zero claims with no supporting chunk (adversarial)
- Hallucination injection test (20% of eval stories): deliberately omit
  relevant chunks; Refined Context must hedge ("evidence is limited") or
  stay within remaining chunks — critical failure if it invents facts

## Failure modes covered
F-V1

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 2.1
