# Eval 1.1 Implementation Plan — Research Coverage

## Design decisions (resolved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | New `EvalRunner.run_researching()` method | Matches existing EvalRunner pattern, isolates RESEARCHING from other states |
| Duplicate detection | TF-IDF cosine similarity | Deterministic, no API calls (golden mode) |
| Test data | Dedicated `fixtures/eval1_research_coverage.json` | Clean separation, extensible for 1.2/1.3 |
| Golden mode | Programmatic mock data → exact assertions | Deterministic, fast, no API calls |
| Live mode | Real Tavily → threshold assertions (≥1 chunk, ≥1 domain, metadata correct) | Tests real ingestion pipeline; baselines for trend |
| Assertion reporting | Collected per case (like `test_outcome_research.py`) | One test per case, all failures reported at end |
| Schema | Add to `schemas.py` | Consistent with existing `GoldenCase` etc. |

## Files to create

### 1. `tests/evals/fixtures/eval1_research_coverage.json`

6 cases, each with `id`, `topic`, `description`, `mock_web_results[]` (content + url), `expectations` (min_chunks, min_domains, max_similarity, scope_correct, source_type_correct), `should_pass`.

| Case ID | Mock data | Expectation |
|---------|-----------|------------|
| `coverage-happy` | 10 results, 5 domains | All pass |
| `coverage-sparse` | 3 results, 1 domain | Fails F-R1 (chunk count + diversity) |
| `coverage-duplicates` | 10 results, 3 domains, 6 near-duplicates | Fails F-R2 (duplicate check) |
| `coverage-single-domain` | 10 results, 1 domain | Fails diversity, passes count |
| `coverage-wrong-metadata` | 5 results, 3 domains, deliberately mutated scope/source_type | Fails metadata checks |
| `coverage-boundary` | Exactly 5 results, 3 domains | All pass (tests ≥ not >) |

**Note on `coverage-wrong-metadata`:** The mock `run_researching()` will accept a flag to intentionally set wrong scope/source_type, verifying the assertion helpers detect it. This is a test of the assertion harness, not the production ingestion path.

### 2. `tests/evals/test_eval1_research_coverage.py`

```python
@pytest.mark.eval
@pytest.mark.parametrize("researching_case", COVERAGE_CASE_IDS, indirect=True)
async def test_research_coverage(researching_case, researching_runner)
```

- Golden mode: Load mock data → `researching_runner.run_researching()` → capture chunks → check 5 assertions → verify all/fail match `should_pass`
- Live mode: Use topic from case → real Tavily search → `researching_runner.run_researching_live()` → same 5 checks but with softer thresholds (≥1) + baseline recording
- No LLM judge needed (rule-based)

## Files to modify

### 3. `tests/evals/schemas.py` — Add schemas

```python
class MockWebResult(BaseModel):
    content: str
    url: str

class ResearchingExpectations(BaseModel):
    min_chunks: Optional[int] = None
    min_domains: Optional[int] = None
    max_similarity: Optional[float] = None
    scope_correct: bool = True
    source_type_correct: bool = True

class ResearchingCase(BaseModel):
    id: str
    topic: str
    description: str
    mock_web_results: List[MockWebResult]
    expectations: ResearchingExpectations
    should_pass: bool
    inject_metadata_errors: bool = False  # for coverage-wrong-metadata case only
```

### 4. `tests/evals/conftest.py` — Add fixture + EvalRunner method

a) `researching_runner` fixture: Like `eval_runner` but with a capture-capable mock vector store.

b) `researching_case` fixture: Loads single case from JSON by ID (mirrors `golden_case` pattern).

c) `EvalRunner.run_researching(case)`: Takes `ResearchingCase`, simulates ingestion: each `MockWebResult` → one chunk with `meta={"scope": "LOCAL", "source_type": "WEB_SEARCH", "url": ...}`. Returns a dict with `{"chunks": [{"content": ..., "url": ..., "scope": ..., "source_type": ...}]}`.

d) `EvalRunner.run_researching_live(case)`: Uses real Tavily (via `TavilySearchService`), real ingestion capture, returns same dict format.

### 5. `tests/evals/assertions.py` — Add 5 check functions

Each returns `Optional[str]` (None = pass, string = error message):

| Function | What it checks |
|----------|---------------|
| `check_chunk_count(chunks, min_count)` | `len(chunks) >= min_count` |
| `check_domain_diversity(chunks, min_domains)` | Distinct root domains via `urlparse` |
| `check_no_duplicates(chunks, max_sim)` | TF-IDF cosine similarity between all pairs ≤ max |
| `check_scope(chunks, expected)` | All chunks carry `scope == expected` |
| `check_source_type(chunks, expected)` | All chunks carry `source_type == expected` |

Note: `check_no_duplicates` needs `sklearn.feature_extraction.text.TfidfVectorizer`. If `scikit-learn` is not already a test dependency, add it or roll a simple bag-of-words cosine similarity.

## Test flow

### Golden mode

```
For each case:
  load ResearchingCase from JSON fixture
  chunks = run_researching(case)  # mock ingestion with case's mock_web_results
  errors = []
  errors += check_chunk_count(chunks, case.expectations.min_chunks)
  errors += check_domain_diversity(chunks, case.expectations.min_domains)
  errors += check_no_duplicates(chunks, case.expectations.max_similarity)
  errors += check_scope(chunks, "LOCAL")
  errors += check_source_type(chunks, "WEB_SEARCH")
  if should_pass:
    assert errors == []  # "Expected all to pass, got: ..."
  else:
    assert errors != []  # "Expected at least one failure, all passed"
```

### Live mode

```
For each case:
  Use case.topic as the search query
  Run real TavilySearchService.search(topic)
  Run real ingestion via ContentFactoryVectorStore (or mock capture)
  captured_chunks = result from ingestion
  errors = []
  errors += check_chunk_count(captured_chunks, min_count=1)  # soft threshold
  errors += check_domain_diversity(captured_chunks, min_domains=1)
  errors += check_scope(captured_chunks, "LOCAL")
  errors += check_source_type(captured_chunks, "WEB_SEARCH")
  # Record actual metrics to baselines: chunk_count, domain_count, max_sim
  baseline_recorder.record_case_score(case.id, "research_coverage", {
    "chunk_count": len(captured_chunks),
    "domain_count": domain_count,
    "max_similarity": max_sim_value
  })
```

Golden mode omits duplicate check in live (no embeddings). Live mode duplicate check is optional (requires real embeddings).

## What NOT to do (constraints)

- Do not modify legacy eval files (`test_outcome_research.py`, etc.)
- Do not implement 1.2 (Chunk Quality) or 1.3 (Calibration)
- Do not modify the golden dataset or its schema
- Do not add an LLM judge to this eval (it's rule-based by design)

## Open infrastructure assumptions

1. **scikit-learn** may need to be added as a test dependency for TF-IDF vectorization. Check `apps/api/pyproject.toml` before implementation. If absent, implement a simple bag-of-words cosine similarity.

2. **`TavilySearchService` imports** in conftest or the test file may need a session-scoped mock dependency. The existing `mock_web_search_service` in root conftest is function-scoped; live mode would need a real service.

3. **Baseline recording format**: The existing `BaselineRecorder` stores `weighted_average` and `dimensions` from LLM judge. For rule-based metrics, we record raw numeric metrics instead. This doesn't require schema changes — just uses the existing dict structure under a new rubric key (`"research_coverage"`).

4. **Live mode requires** a real `TavilySearchService` with `TAVILY_API_KEY` and a working `ContentFactoryVectorStore` with DB connection. This matches the existing `--live` flag assumption.

## Execution order

1. Add schemas to `schemas.py`
2. Write `fixtures/eval1_research_coverage.json` with 6 cases
3. Add check functions to `assertions.py`
4. Add `researching_runner` fixture + `run_researching` methods to `conftest.py`
5. Write `test_eval1_research_coverage.py`
6. Verify: `uv run pytest tests/evals/test_eval1_research_coverage.py -m eval -v`
7. Verify: `nx lint api` and `nx test:unit api` no regressions
