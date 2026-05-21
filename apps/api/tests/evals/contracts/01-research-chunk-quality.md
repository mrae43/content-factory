# Research Desk — eval contract (1.2 Chunk Quality)

## What this eval guards
Research chunks must be topically relevant, information-dense, and coherent.
Low-quality chunks waste embedding storage and degrade the Retrieval Desk's
synthesis.

## Assertions
- LLM judge scores each chunk on relevance (1–5), information density (1–5),
  and coherence (1–5)
- Mean score per dimension across all chunks for a Story: ≥ 3.5
- Flag stories where > 20% of chunks score < 3 on relevance

## Failure modes covered
F-R1 (sparse research indirectly — thin content scores low on density)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 1.2
