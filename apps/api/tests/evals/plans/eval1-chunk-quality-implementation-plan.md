# Eval 1.2 Implementation Plan — Chunk Quality

## Design decisions (resolved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunk scope | `WEB_SEARCH` only (Tavily LOCAL chunks) | Matches prompt template in evals-criteria.md; keeps scope tight |
| Scoring scale | Dedicated 1–5 per dimension | Separate from `judge.py` (0.0/0.5/1.0); matches contract exactly |
| Test data | Frozen chunks from live pipeline run against canonical topics | Deterministic golden mode; real-world chunk texts |
| Fixture structure | Desk-level `eval1_research.json` with sections (`coverage_cases`, `quality_corpus`, `relevance_cases`) | Single fixture file per desk, extensible for 1.3 |
| Case ID convention | Prefixed with sub-eval slug; failure-mode suffix when targeted | Self-locating IDs (e.g. `quality-brics`, `quality-sparse-FR1`) |
| Golden mode | Load `cached_responses` from fixture; no LLM calls | Deterministic, fast, no API costs |
| Live mode | Real LLM judge; compare entry-level means vs cached; flag drift > 0.5 | Regression detection without per-chunk noise |
| Cache update | `--update-cache` reads `baselines.json`, writes to fixture; separate from corpus capture | Never conflates cache refresh with corpus construction |
| Fixture validation | Pydantic constraints at load time (range 1–5, array length match) | Separate code path from eval assertions |
| Golden dataset | Not used — `quality_corpus` loads from its own desk-level fixture | No coupling to golden_dataset.json schema |

## Files to create

### 1. `tests/evals/chunk_quality_scorer.py`

New module: standalone LLM-as-judge for per-chunk 1–5 scoring.

```python
class ChunkQualityScorer:
    """Scores individual chunks on relevance (1–5), density (1–5), coherence (1–5).
    Separate from judge.py — own 1–5 scale, own prompt template."""

    async def score_chunk(self, topic: str, chunk_text: str) -> CachedChunkScore
    async def score_chunks(self, topic: str, chunks: list[SourceChunk]) -> list[CachedChunkScore]
    def compute_entry_means(self, scores: list[CachedChunkScore]) -> dict
    def check_thresholds(self, means: dict, scores: list[CachedChunkScore]) -> list[str]
    def detect_drift(self, cached: list[CachedChunkScore], live: list[CachedChunkScore]) -> list[str]
```

Uses the exact prompt template from `evals-criteria.md` §1.2:

```
You are evaluating a research chunk extracted from a web search result about the topic: "{topic}".

Chunk text:
---
{chunk_text}
---

Score the chunk on each dimension from 1–5:
1. Topical relevance: Is the chunk substantively about the topic?
2. Information density: Does it contain specific, usable facts (not boilerplate)?
3. Coherence: Is it a complete, readable unit (not mid-sentence truncation)?

Return JSON: {"relevance": N, "density": N, "coherence": N}
```

Uses `judge_llm.with_structured_output(CachedChunkScore)` — reuses the session-scoped LLM.

### 2. `tests/evals/test_eval1_chunk_quality.py`

```python
@pytest.mark.eval
class TestChunkQuality:

    QUALITY_ENTRY_IDS = [...]  # canonical topic slugs from quality_corpus

    @pytest.mark.parametrize("quality_entry_id", QUALITY_ENTRY_IDS, indirect=True)
    async def test_chunk_dimension_means(
        self, quality_entry_id, quality_corpus, chunk_quality_scorer,
        baseline_recorder, request
    ):
        entry = get_entry_by_id(quality_corpus, quality_entry_id)

        if golden_mode:
            scores = entry.cached_responses
        else:
            scores = await chunk_quality_scorer.score_chunks(entry.topic, entry.source_chunks)
            drift = chunk_quality_scorer.detect_drift(entry.cached_responses, scores)
            # record drift warnings to baseline, not hard failure

        means = chunk_quality_scorer.compute_entry_means(scores)

        # Assertion 1: Mean >= 3.5 per dimension
        violations = []
        for dim in ["relevance", "density", "coherence"]:
            if means[f"{dim}_mean"] < 3.5:
                violations.append(f"{dim}_mean {means[f'{dim}_mean']:.2f} < 3.5")

        # Assertion 2: Flag > 20% chunks with relevance < 3 (warning, not hard fail)
        low_rel_count = sum(1 for s in scores if s.relevance < 3)
        low_rel_pct = low_rel_count / len(scores) * 100
        if low_rel_pct > 20:
            violations.append(f"WARNING: {low_rel_pct:.0f}% chunks relevance < 3")

        if violations:
            pytest.fail(f"{quality_entry_id}: {'; '.join(violations)}")
```

| Case ID pattern | What it tests | Expectation |
|----------------|---------------|-------------|
| `quality-{topic-slug}` | Canonical topic with typical Tavily results | All dimensions mean >= 3.5 |
| `quality-sparse-FR1` | Topic returning very few/thin results | May fail density; tests F-R1 |
| `quality-boilerplate` | Results with boilerplate/low-information text | Fails density; tests detection |

### 3. `scripts/capture_corpus.py`

Standalone script (not pytest). One-shot corpus construction:

```
For each canonical topic:
  1. Call TavilySearchService.search(topic)
  2. Build source_chunks: [{content, source_url}, ...]
  3. Call LLM judge on each chunk -> cached_responses
  4. Write entry to quality_corpus section in eval1_research.json
  5. Append run record to capture_log.json
```

### 4. `tests/evals/fixtures/capture_log.json`

```jsonc
{
  "runs": [
    {
      "run_id": "<uuid>",
      "captured_at": "2026-05-21T12:00:00Z",
      "entries": [
        {"topic": "BRICS De-dollarization...", "chunk_count": 8, "status": "success"},
        ...
      ]
    }
  ]
}
```

Append-only, written only by `scripts/capture_corpus.py`. Never read by test code — traceability only.

## Files to modify

### 5. `tests/evals/fixtures/eval1_research_coverage.json`

**Rename** to `eval1_research.json` and restructure with sections:

```jsonc
{
  "eval_version": "1",
  "schema_version": "2",
  "coverage_cases": [
    // existing 6 cases, unchanged content
    // IDs renamed to carry failure-mode suffix where applicable:
    //   "coverage-sparse"       -> "coverage-sparse-FR1"
    //   "coverage-duplicates"   -> "coverage-dupe-FR2"
    //   others unchanged
  ],
  "quality_corpus": {
    "description": "Frozen Tavily chunks from live pipeline runs against canonical topics",
    "capture_run_id": "<uuid from capture_log.json>",
    "entries": [
      {
        "topic": "BRICS De-dollarization and the Shift Away from USD Dominance",
        "description": "Economics — high-density financial content",
        "source_chunks": [
          {"content": "...", "source_url": "https://..."}
        ],
        "cached_responses": [
          {"relevance": 5, "density": 4, "coherence": 5}
        ]
      }
    ]
  },
  "relevance_cases": []  // placeholder for Eval 1.3
}
```

### 6. `tests/evals/schemas.py` — Add schemas

```python
class CachedChunkScore(BaseModel):
    relevance: int = Field(..., ge=1, le=5)
    density: int = Field(..., ge=1, le=5)
    coherence: int = Field(..., ge=1, le=5)

class SourceChunk(BaseModel):
    content: str
    source_url: str

class QualityCorpusEntry(BaseModel):
    topic: str
    description: str
    source_chunks: list[SourceChunk]
    cached_responses: list[CachedChunkScore]

    @field_validator("cached_responses")
    @classmethod
    def arrays_match(cls, v, info):
        chunks = info.data.get("source_chunks", [])
        if len(v) != len(chunks):
            raise ValueError("cached_responses length must match source_chunks")
        return v

class QualityCorpus(BaseModel):
    description: str
    capture_run_id: str
    entries: list[QualityCorpusEntry]

class Eval1ResearchFixture(BaseModel):
    eval_version: str
    schema_version: str
    coverage_cases: list[ResearchingCase]
    quality_corpus: QualityCorpus
    relevance_cases: list[dict]  # placeholder
```

### 7. `tests/evals/conftest.py` — Add fixtures + CLI option

a) **CLI option**: `--update-cache`

```python
parser.addoption(
    "--update-cache", action="store_true", default=False,
    help="Update cached_judge_response in fixture from baselines.json",
)
```

b) **Path constant**: Rename `_RESEARCHING_FIXTURES_PATH` → `_EVAL1_RESEARCH_FIXTURES_PATH`

c) **Loader**: `_load_eval1_research()` returns structured dict with sections (backward-compat: `researching_case` fixture still extracts from `coverage_cases`).

d) **New fixture** `quality_corpus`:

```python
@pytest.fixture
def quality_corpus() -> QualityCorpus:
    data = _load_eval1_research()
    return data["quality_corpus"]
```

e) **New fixture** `quality_entry` (indirect parametrization by entry ID):

```python
@pytest.fixture
def quality_entry(request, quality_corpus) -> QualityCorpusEntry:
    entry_id = request.param
    for entry in quality_corpus.entries:
        slug = entry.topic.lower().replace(" ", "-")[:40]
        if slug == entry_id:
            return entry
    raise ValueError(f"Entry '{entry_id}' not found")
```

f) **New fixture** `chunk_quality_scorer`:

```python
@pytest.fixture
def chunk_quality_scorer(judge_llm) -> ChunkQualityScorer:
    return ChunkQualityScorer(judge_llm=judge_llm)
```

## Test flow

### Golden mode (default, CI)

```
For each quality_corpus entry:
  Load QualityCorpusEntry with cached_responses
  Validate via Pydantic (range 1-5, arrays match)  # fixture validation, not eval
  Compute per-dimension means across cached_responses
  Assertion 1: each dimension mean >= 3.5
  Assertion 2: count chunks with relevance < 3
               if > 20% of total -> WARNING (asserted as violation)
  Record to baseline_recorder
```

### Live mode (--live)

```
For each quality_corpus entry:
  Load source_chunks from fixture (unchanged)
  Call ChunkQualityScorer.score_chunks(topic, source_chunks)  # real LLM
  Compute per-dimension means
  Detect drift: compare live means vs cached means
    if any dimension mean drifts > 0.5 -> flag in baseline
  Same two assertions as golden mode, applied to live scores
  Record to baseline_recorder
```

### Cache update (--update-cache --run-id <id>)

```
Requires run_id that exists in baselines.json
  Load baselines.json, find run_id
  Read live scores for that run
  Overwrite cached_responses in eval1_research.json for matching entries
  Do NOT touch source_chunks, capture_run_id, or other fields
```

## What NOT to do (constraints)

- Do not modify legacy eval files (`test_eval1_research_coverage.py`, `test_outcome_research.py`, etc.)
- Do not implement 1.3 (Calibration) — leave `relevance_cases: []` as placeholder
- Do not modify `judge.py` or `rubrics.py` — separate scoring system
- Do not modify the golden dataset or its schema
- Do not introduce a pgvector dependency (frozen chunks are plain strings)
- Do not combine corpus capture and cache update in one operation

## Open infrastructure assumptions

1. **`capture_corpus.py` needs `TAVILY_API_KEY` and `TOGETHER_API_KEY` / `GEMINI_API_KEY`** in env, matching production requirements. The `eval_judge_model` setting controls which LLM scores chunk quality.

2. **Canonical topic definition** must be resolved before fixture construction. Audit golden cases (23) against coverage criteria, promote matching cases, fill gaps with new topics. This is a one-time sub-task of step 1.

3. **Baseline recording format**: The existing `BaselineRecorder` stores `weighted_average` and `dimensions` from LLM judge. For chunk quality, record `dimension_means` (relevance_mean, density_mean, coherence_mean), `low_relevance_pct`, and `drift_flags` under a new rubric key (`"chunk_quality"`).

4. **`--update-cache` reads `baselines.json`**, which is `tests/evals/baselines.json`. The run_id stored there must match a live-mode session. This assumes the existing baseline infrastructure is sufficient; no new persistence layer needed.

5. **Golden mode requires no external services** (no LLM, no Tavily, no DB). This matches the existing golden-mode contract for all evals.

6. **Live mode requires** `GEMINI_API_KEY` (for embeddings — if ChunkQualityScorer needs to be production-identical) and the `eval_judge_model` LLM endpoint. The LLM is already session-scoped via the `judge_llm` fixture.

## Execution order

1. Audit golden cases; define canonical topic set (6–8 topics)
2. Add schemas to `schemas.py` (CachedChunkScore, SourceChunk, QualityCorpusEntry, QualityCorpus, Eval1ResearchFixture)
3. Rename `eval1_research_coverage.json` → `eval1_research.json`; restructure with sections
4. Write `chunk_quality_scorer.py`
5. Add `--update-cache` CLI option, fixtures, and loader updates to `conftest.py`
6. Write `test_eval1_chunk_quality.py`
7. Run `scripts/capture_corpus.py` to populate `quality_corpus` + `capture_log.json`
8. Verify: `uv run pytest tests/evals/test_eval1_chunk_quality.py -m eval -v`
9. Verify: `uv run pytest tests/evals/test_eval1_research_coverage.py -m eval -v` (no regressions)
10. Verify: `nx lint api` and `nx test:unit api` no regressions
