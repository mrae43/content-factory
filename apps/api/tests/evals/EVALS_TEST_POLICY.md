# Evals + Test Policy — Findings & Implementation Plan

## 1. Current State Assessment

### 1.1 Test Infrastructure

| Layer | Files | Count | Status |
|:------|:------|:------|:-------|
| **Unit** | `tests/unit/` — crud, config, chunking, routes, queue_worker, vector_store | 6 files | Stable |
| **Agent** | `tests/agents/` — research, copywriter, red_team, asset_studio, optimizer | 5 files | Stable (P0 bugs fixed) |
| **Integration** | `tests/integration/` — orchestrator transitions | 1 file (35 tests) | Stable, comprehensive |
| **Eval** | `tests/evals/` | 0 files (empty `__init__.py`) | **Not started** |
| **Golden** | `tests/golden/golden_dataset.json` | `[]` (empty) | **Not started** |

### 1.2 Bugs Found in Existing Tests

| # | Severity | File | Issue | Status |
|:--|:---------|:-----|:------|:-------|
| 1 | **Medium** | `tests/agents/test_copywriter_agent.py:83` | Asserts `"No research chunks" in result.reasoning` but code now returns `"No refined research context available for scriptwriting."`. Test passes accidentally because the fixture omits `refined_context`, hitting the error path — but asserts wrong message. | **Fixed** — context now passes `refined_context`, assertion updated |
| 2 | **Medium** | `tests/agents/test_red_team_agent.py:16-36` | `chain_mock` patches `RunnableSequence.ainvoke` globally. Red Team now has 3 LLM passes (claim extraction, evidence retrieval, evaluation) but `test_returns_success_when_all_supported` returns `RedTeamVerdict` for ALL `ainvoke` calls — Pass 1 expects `ClaimExtractionResult`. Test passes accidentally because mock ignores schema type. | **Fixed** — `multi_chain_mock` with `side_effect` list; `ClaimExtractionResult` fixtures added |
| 3 | **Low** | `tests/agents/conftest.py:116-122` | `copywriter_context` fixture still passes `vector_store` and `job_id` but CopywriterAgent now reads `refined_context`. Fixture is dead code — tests bypass it by constructing their own context dicts. | **Fixed** — fixture now provides `refined_context` |
| 4 | **Low** | `app/workers/orchestrator.py:140-144` | Empty `refined_context` silently proceeds — defers failure to CopywriterAgent two hops later with a confusing error message. Guard should be at research transition. | **Fixed** — raises immediately if `refined_context` is empty |

### 1.3 Coverage Gaps

| What's Missing | Impact |
|:---------------|:-------|
| Trajectory evaluations | Tests only check final status, not reasoning paths or tool call sequences |
| Outcome evaluations | No LLM-as-Judge scoring, no benchmark dataset |
| System monitoring | No structured trace emission for production quality tracking |

---

## 2. Evaluation Framework Design

### 2.1 The Four Layers (from `testing-eval.md`)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4: SYSTEM (Production Monitoring)                │
│  → Structured traces, defect rate dashboards            │
│  → NOT in MVP scope                                      │
├─────────────────────────────────────────────────────────┤
│  Layer 3: OUTCOME (LLM-as-Judge)                        │
│  → Golden dataset + scoring rubrics                      │
│  → GemJudge (gemini-2.5-flash) evaluates agent outputs  │
│  → Scores tracked over time as distributions             │
├─────────────────────────────────────────────────────────┤
│  Layer 2: TRAJECTORY (Reasoning Path)                    │
│  → Did the agent take the right steps?                   │
│  → Verify tool call sequences, retrieval quality         │
│  → Check claim extraction completeness                   │
├─────────────────────────────────────────────────────────┤
│  Layer 1: COMPONENT (Deterministic)                      │
│  → Traditional unit tests (EXISTING)                     │
│  → Fix bugs #1-4 above                                  │
│  → Add optimizer tests                                   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Priority

| Phase | Layer | What | Effort |
|:------|:------|:-----|:-------|
| **P0** | Component | Fix stale tests (bugs #1-3) | Small |
| **P0** | Component | Add `ScriptOptimizerAgent` tests | Small |
| **P0** | Component | Add empty `refined_context` guard | Small |
| **P1** | Outcome | Golden dataset (20 cases) | Medium |
| **P1** | Outcome | LLM-as-Judge scoring harness | Medium |
| **P2** | Trajectory | Agent trace capture & validation | Large |
| **P3** | System | Structured logging + dashboards | Deferred |

---

## 3. Golden Dataset Schema

### 3.1 Dataset Entry Structure

```json
{
  "id": "case-001",
  "category": "factual_accuracy",
  "domain": "economics",
  "input": {
    "topic": "BRICS De-dollarization 2025",
    "pre_context": {
      "raw_text": "...",
      "source_urls": [],
      "target_audience": "Investors",
      "guardrail_strictness": "High"
    }
  },
  "expected_outcomes": {
    "research": {
      "must_include_facts": ["BRICS payment system", "GDP growth figures"],
      "must_avoid": ["fabricated statistics", "opinion as fact"],
      "min_chunks": 2,
      "min_confidence": 0.7
    },
    "script": {
      "must_include_topics": ["de-dollarization", "BRICS alternatives"],
      "must_avoid": ["unverified claims"],
      "scene_count_range": [3, 8],
      "word_count_range": [150, 500],
      "must_have_hook": true,
      "must_have_loop": true
    },
    "fact_check": {
      "expected_verdict": "SUPPORTED",
      "max_unsupported_claims": 0,
      "claims_with_known_verdicts": [
        {"claim_text": "...", "expected_verdict": "SUPPORTED"}
      ]
    }
  },
  "difficulty": "easy",
  "tags": ["economics", "geopolitics", "statistics"]
}
```

### 3.2 Category Distribution Target

| Category | Count | Description |
|:---------|:------|:------------|
| `factual_accuracy` | 6 | Well-documented topics with verifiable facts |
| `hallucination_trap` | 4 | Topics that commonly trigger fabricated statistics |
| `conflicting_evidence` | 4 | Topics where sources disagree |
| `edge_case_minimal` | 3 | Very short pre_context, sparse information |
| `edge_case_long` | 3 | Very long pre_context, information overload |
| **Total** | **20** | |

### 3.3 Domain Coverage

Economics, health/medicine, technology, history, climate science, geopolitics, space/exploration, social issues.

---

## 4. Scoring Rubrics

### 4.1 Research Quality Rubric

| Dimension | Weight | Score 0 | Score 0.5 | Score 1.0 |
|:----------|:-------|:--------|:----------|:----------|
| **Completeness** | 30% | Missing key facts | Partial coverage | All key facts captured |
| **Accuracy** | 30% | Contains fabrications | Minor inaccuracies | Fully grounded |
| **Synthesis** | 20% | Raw chunk dump | Partial integration | Coherent narrative |
| **Confidence Calibration** | 20% | Over/underconfident | Reasonable | Well-calibrated |

### 4.2 Script Quality Rubric

| Dimension | Weight | Score 0 | Score 0.5 | Score 1.0 |
|:----------|:-------|:--------|:----------|:----------|
| **Factual Grounding** | 35% | Hallucinations present | Minor stretches | Fully grounded in refined_context |
| **Narrative Structure** | 20% | No hook/loop | Partial structure | Hook-Value-Loop complete |
| **Engagement** | 15% | Dry/academic | Moderate | High-retention pacing |
| **Storyboard Quality** | 15% | Vague cues | Adequate | Precise visual/audio directives |
| **Length** | 15% | <60s or >240s | Borderline | 120-180s target |

### 4.3 Fact-Check Quality Rubric

| Dimension | Weight | Score 0 | Score 0.5 | Score 1.0 |
|:----------|:-------|:--------|:----------|:----------|
| **Claim Coverage** | 30% | Misses major claims | Partial extraction | All claims extracted |
| **Verdict Accuracy** | 40% | Wrong verdicts | Some wrong | All verdicts correct |
| **Evidence Quality** | 20% | No evidence cited | Weak evidence | Strong, specific evidence |
| **Confidence Calibration** | 10% | Mis-calibrated | Borderline | Well-calibrated |

### 4.4 Optimization Quality Rubric

| Dimension | Weight | Score 0 | Score 0.5 | Score 1.0 |
|:----------|:-------|:--------|:----------|:----------|
| **Patch Precision** | 35% | Rewrote whole script | Over-patched | Surgical patches only |
| **Narrative Preservation** | 25% | Flow broken | Minor disruptions | Seamless |
| **Grounding** | 25% | New hallucinations | Borderline claims | Fully grounded |
| **Claim Resolution** | 15% | Failed claims unresolved | Partial resolution | All claims resolved |

---

## 5. LLM-as-Judge Implementation

### 5.1 Architecture

```
Golden Dataset ──► Eval Runner ──► Agent Pipeline (mocked LLM or real) ──► Outputs
                                        │
                                        ▼
                                  Scoring Harness
                                        │
                              ┌─────────┴─────────┐
                              │                    │
                        Deterministic          LLM-as-Judge
                        Assertions            (gemini-2.5-flash)
                              │                    │
                              └─────────┬─────────┘
                                        ▼
                                  Score Aggregator
                                        │
                                        ▼
                                  JSON Report
                              (scores + distributions)
```

### 5.2 Eval Test Structure

```python
# tests/evals/test_outcome_research.py
@pytest.mark.eval
async def test_research_quality_brics_case(golden_case, judge_llm):
    """Outcome eval: research output scored against golden case."""
    result = await run_research_agent(golden_case.input)
    scores = await judge_llm.score(
        rubric=RESEARCH_RUBRIC,
        input=golden_case.input,
        output=result,
        reference=golden_case.expected_outcomes.research,
    )
    assert scores.weighted_average >= 0.7
    assert scores.dimension("accuracy") >= 0.8
```

### 5.3 Judge Prompt Template

```
You are an expert evaluator for AI-generated content. Score the following
agent output against the rubric below.

## Rubric
{rubric}

## Input Provided to Agent
{input}

## Agent Output
{output}

## Reference (Known-Good)
{reference}

## Instructions
1. Score each dimension independently (0.0, 0.5, or 1.0).
2. Provide specific evidence from the output for each score.
3. Calculate the weighted average.
4. Return JSON: {{"dimensions": {{...}}, "weighted_average": ..., "reasoning": "..."}}
```

---

## 6. Test Policy Rules

### 6.1 Marker Convention

| Marker | Purpose | Runs On | Timeout |
|:-------|:--------|:--------|:--------|
| `@pytest.mark.unit` | Deterministic component tests | Every commit | 5s per test |
| `@pytest.mark.agent` | Agent-level tests (mocked LLM) | Every commit | 10s per test |
| `@pytest.mark.integration` | Orchestrator transitions (mocked deps) | Every commit | 10s per test |
| `@pytest.mark.eval` | LLM-as-Judge outcome evals | CI nightly + on deploy | 60s per test |
| `@pytest.mark.golden` | Full pipeline against golden dataset | CI nightly | 300s per test |

### 6.2 CI Pipeline Gates

```
Every PR:
  pytest -m "unit or agent or integration" --timeout=30
  → Must pass 100%

Nightly (or on deploy):
  pytest -m "eval or golden" --timeout=600
  → Score regression check: weighted_average must not decrease > 5%
```

### 6.3 Regression Policy

| Metric | Threshold | Action |
|:-------|:----------|:-------|
| Component test pass rate | 100% | Block merge |
| Eval weighted score (any agent) | >= 0.7 | Block merge |
| Eval score regression (vs. last 7-day avg) | <= 5% drop | Alert + manual review |
| Golden dataset pass rate | >= 80% | Block deploy |

### 6.4 Adding New Eval Cases

1. Create a new entry in `tests/golden/golden_dataset.json`
2. Write a corresponding `@pytest.mark.eval` test in `tests/evals/`
3. Run the eval locally and record baseline scores
4. Add baseline to `tests/evals/baselines.json`
5. PR must include both the case and the baseline

---

## 7. Immediate Action Items

### Phase 0 — Fix Existing Tests (Do Now)

- [x] Fix `test_copywriter_agent.py:83` — update assertion to `"No refined research context"`
- [x] Fix `test_red_team_agent.py` — separate mocks for extraction vs. evaluation passes
- [x] Update `copywriter_context` fixture in `tests/agents/conftest.py` — add `refined_context`
- [x] Add guard for empty `refined_context` in `orchestrator.py:140`
- [x] Add `ScriptOptimizerAgent` unit tests in `tests/agents/test_optimizer_agent.py`
- [x] Remove dead `tools.py` or mark as future integration point
- [x] Fix `research_schema_output` fixture — add missing `refined_context` field

### Phase 1 — Golden Dataset + Outcome Evals

- [x] Populate `golden_dataset.json` with 20 seed cases (6 factual, 4 hallucination_trap, 4 conflicting, 6 edge)
- [x] Create `tests/evals/conftest.py` with judge LLM, eval runner, score aggregator
- [ ] Create `tests/evals/test_outcome_research.py`
- [ ] Create `tests/evals/test_outcome_script.py`
- [ ] Create `tests/evals/test_outcome_factcheck.py`
- [ ] Create `tests/evals/test_outcome_optimizer.py`
- [ ] Record baselines in `tests/evals/baselines.json`

### Phase 2 — Trajectory Evals

- [ ] Add structured trace emission to agents (intermediate reasoning, tool calls)
- [ ] Create `tests/evals/test_trajectory_research.py`
- [ ] Create `tests/evals/test_trajectory_factcheck.py`
- [ ] Verify claim extraction completeness per golden case

### Phase 3 — System (Deferred)

- [ ] Structured logging to Postgres/external service
- [ ] Defect rate dashboard
- [ ] Continuous improvement loop (annotate failures → new eval cases)

---

## 8. File Structure After Implementation

```
tests/
  evals/
    __init__.py
    EVALS_TEST_POLICY.md          ← this document
    conftest.py                    ← judge LLM, eval runner, score aggregator fixtures
    rubrics.py                     ← scoring rubric definitions
    test_outcome_research.py       ← research agent outcome evals
    test_outcome_script.py         ← copywriter outcome evals
    test_outcome_factcheck.py      ← red team outcome evals
    test_outcome_optimizer.py      ← optimizer outcome evals
    baselines.json                 ← recorded baseline scores per golden case
  golden/
    golden_dataset.json            ← 20+ curated test cases
  agents/
    conftest.py                    ← (updated: fix copywriter fixture, add optimizer + extraction fixtures, multi_chain_mock)
    test_optimizer_agent.py        ← NEW: ScriptOptimizerAgent tests (6 tests)
    test_copywriter_agent.py       ← (fixed: refined_context API, correct assertion)
    test_red_team_agent.py         ← (fixed: multi-pass mock strategy via multi_chain_mock)
    test_research_agent.py         ← (fixed: research_schema_output fixture with refined_context)
    test_asset_studio_agent.py
```
