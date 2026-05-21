# Retrieval Desk — eval contract (2.3 Context Assembly Quality)

## What this eval guards
The AssembledContext produced by the Context Builder must contain all three
structural sections (narrative summary, evidence sections, raw chunk payloads),
with populated similarity scores and topic_relevance values. It must also
survive the Fact-Check Loop invariant (not rebuilt during revisions).

## Assertions
- Structure compliance: AssembledContext contains narrative summary, evidence
  sections, and raw chunk payloads
- `similarity_score` populated: 100% of retrieved chunks carry non-null score
- `topic_relevance` populated: 100% of retrieved chunks carry valid enum value
- Persisted on RenderJob: `assembled_context` field is non-null after RETRIEVAL
- Query composition: LLM judge ≥ 4.0/5 for query targeting correctness
- Evidence section relevance: LLM judge ≥ 4.0/5 that top-N evidence sections
  are the most pertinent
- Reuse invariant: after ≥ 1 Fact-Check Loop iteration, AssembledContext is
  identical to the version persisted after initial RETRIEVAL

## Failure modes covered
F-V2 (indirectly — broken context assembly breaks citation chains)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 2.3
