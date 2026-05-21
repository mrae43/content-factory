# Retrieval Desk — eval contract (2.2 Citation Index Completeness)

## What this eval guards
Every factual passage in the Refined Context must be traceable to a specific
research chunk via the Citation Index. Broken citations make the Fact-Check
Desk blind.

## Assertions
- Every passage has a citation: sample 10 random passages, ≥ 9/10 traceable
  to Citation Index
- Citation Index references valid chunk IDs: 100% of cited Research Chunk IDs
  exist in pgvector
- No orphaned citations: 100% of Citation Index entries map to a passage in
  the Refined Context
- Citation Index persisted on RenderJob: `citation_index` field is non-null
  after RETRIEVAL

## Failure modes covered
F-V2

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 2.2
