# Research Desk — eval contract (1.1 Coverage)

## What this eval guards
The Research Desk must produce enough diverse, unique chunks from web search
results to support downstream synthesis. Insufficient or duplicated research
poisons every desk that follows.

## Assertions
- Minimum chunk count per story: ≥ 5 chunks with `source_type=WEB_SEARCH`
- Source diversity: ≥ 3 distinct root domains across chunks
- No duplicate chunks: cosine similarity between any two chunk embeddings
  must not exceed 0.97
- `scope` correctly set: 100% of new chunks carry `scope=LOCAL`
- `source_type` correctly set: 100% of Tavily-originated chunks carry
  `source_type=WEB_SEARCH`

## Failure modes covered
F-R1, F-R2

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 1.1
