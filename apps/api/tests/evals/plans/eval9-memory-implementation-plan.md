# Eval 9 Implementation Plan — Long-Term Memory (Cross-Job GLOBAL Scope)

## Design decisions (resolved in session)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Eval numbering | **Eval 9** (new top-level) | Cross-cutting concern parallel to Eval 5 (Loop); doesn't fit any existing stage |
| Sub-evals | 4 sub-evals (9.1–9.4) | Mirrors Eval 1 (coverage/quality/calibration) and Eval 4 (4 sub-evals) |
| 9.1 Promotion Quality | Golden + live mode | Golden: structural assertions (count, metadata, no dupes). Live: LLM judge on fact accuracy |
| 9.2 Retrieval Relevance | Golden only | Structural assertions (label integrity, dedup, non-empty). Live mode deferred |
| 9.3 Downstream Benefit | Golden + live mode | Golden: seeded fixture, assert facts in context/script/passing fact-check. Live: LLM judge `factual_grounding` |
| 9.4 Safety & Integrity | Golden only — A + C | A: hallucination propagation (seeded wrong fact → caught by Red Team). C: domain leakage (economics facts → health job → no leakage). B and D deferred |
| Test design 9.3 | **Synthetic paired cases** | Existing golden cases not designed for chaining. New Fed Policy → EM Debt pair gives controlled ground truth |
| Test harness 9.3 | Golden: **fixture-seeded GLOBAL store**. Live: **real two-job sequence** | Golden mode decoupled from orchestrator; live mode uses `--live` flag |
| Downstream metric | Golden: **assertion-based** (fact in context + script + passes fact-check). Live: **LLM judge `factual_grounding`** | Assertions are deterministic; LLM judge for semantic quality in live mode |
| Fixture format | `fixtures/global_memory_fixtures.json` | Mirrors `eval1_research.json` pattern |
| Golden cases prefix | **C-001 / C-002** (new `CROSS_JOB` trace type) | Avoids collision with existing H/R/E/F/N/M prefixes |
| Failure codes | **F-M1** (hallucination propagation, Critical) + **F-M3** (domain leakage, Medium) | F-M2 (suppression) and F-M4 (duplication) deferred |

## Files to create

### 1. `contracts/09-memory-promotion-quality.md`

Contract for 9.1. Asserts: GLOBAL facts must be atomic, compressed, non-redundant, carry complete metadata (`source_type=COMPRESSED_FACT`, `original_job_id`, `fact_category`, `confidence`, `ingested_at`).

### 2. `contracts/09-memory-retrieval-relevance.md`

Contract for 9.2. Asserts: Sequential Assembly must produce both `=== SYSTEM INTEL ===` and `=== CURRENT RUN RESEARCH ===` sections. GLOBAL facts labelled distinctly with no label leakage. Duplicate GLOBAL facts deduplicated. Not empty when relevant GLOBAL facts exist.

### 3. `contracts/09-memory-downstream-benefit.md`

Contract for 9.3. Asserts: GLOBAL facts from prior/completed jobs must be retrievable by new jobs on related topics. Script factual grounding must benefit from GLOBAL context. Known-verified facts from GLOBAL must not be re-flagged as UNSUPPORTED.

### 4. `contracts/09-memory-safety-integrity.md`

Contract for 9.4. Asserts: Intentionally distorted GLOBAL facts must receive UNSUPPORTED verdicts. GLOBAL facts from unrelated domains must not appear in script or context for a different-domain job.

### 5. `fixtures/global_memory_fixtures.json`

Structure:
```json
{
  "eval_version": "1",
  "schema_version": "1",
  "promotion_cases": [
    {
      "id": "promotion-happy",
      "script_content": "...",
      "supported_claims": [...],
      "expected_fact_count": 4,
      "expected_domains": ["economic"],
      "expectation": "should_pass"
    }
  ],
  "synthetic_pairs": [
    {
      "id": "memory-fed-to-em",
      "job_a": {
        "topic": "U.S. Federal Reserve Monetary Policy 2024-2025",
        "user_reference": "..."
      },
      "global_facts": [
        {
          "content": "The Federal Reserve raised the federal funds rate to 5.25-5.50% in July 2023 and held it through 2024.",
          "meta": {
            "scope": "GLOBAL", "version": "1.0",
            "source_type": "COMPRESSED_FACT",
            "fact_category": "monetary_policy",
            "claim_verdict": "SUPPORTED", "confidence": 0.95
          }
        },
        {
          "content": "Core PCE inflation fell from 5.4% in 2022 to 2.9% by Q4 2024.",
          "meta": { "fact_category": "economic" }
        },
        {
          "content": "The Fed began quantitative tightening in June 2022, reducing its balance sheet by approximately $1.5 trillion by end of 2024.",
          "meta": { "fact_category": "monetary_policy" }
        }
      ],
      "job_b": {
        "topic": "Impact of Federal Reserve Policy on Emerging Market Debt 2025",
        "user_reference": "...",
        "expected_outcomes": {
          "must_include_facts": [
            "federal funds rate 5.25-5.50%",
            "Core PCE inflation fell to 2.9%",
            "quantitative tightening balance sheet reduction"
          ],
          "must_avoid": [
            "Claiming the Fed cut rates before 2025",
            "Stating QE was still active in 2024"
          ],
          "expected_verdicts": {
            "The Federal Reserve held rates at 5.25-5.50% through 2024": "SUPPORTED",
            "Emerging market bond yields remained elevated due to tight US policy": null
          }
        },
        "should_pass": true
      }
    }
  ],
  "adversarial_cases": [
    {
      "id": "hallucination-propagation",
      "description": "Seeded GLOBAL fact has distorted number — Red Team must catch it",
      "global_facts": [
        {
          "content": "US GDP grew 3.8% in Q4 2025",
          "meta": { "fact_category": "economic" }
        }
      ],
      "job_b": {
        "topic": "US Economic Growth 2025-2026 Outlook",
        "user_reference": "The actual Q4 2025 GDP growth was 2.8%, not 3.8%..."
      },
      "expectation": "red_team_must_catch",
      "expected_verdict": "UNSUPPORTED"
    },
    {
      "id": "domain-leakage",
      "description": "Economics GLOBAL facts must not leak into health job context",
      "global_facts": [
        { "content": "The Fed raised rates to 5.25-5.50% in July 2023.", "meta": { "fact_category": "economic" } },
        { "content": "US core PCE inflation fell to 2.9% by Q4 2024.", "meta": { "fact_category": "economic" } }
      ],
      "job_b": {
        "topic": "mRNA Vaccine Technology and Clinical Trial Results",
        "user_reference": "mRNA vaccines use lipid nanoparticles..."
      },
      "expectation": "no_domain_leakage",
      "must_not_appear_in_context": ["5.25-5.50%", "PCE inflation", "federal funds rate"]
    }
  ]
}
```

### 6. `test_memory_promotion.py`

**Eval 9.1 — Promotion Quality**

```python
@pytest.mark.eval
@pytest.mark.parametrize("promotion_case", PROMOTION_CASE_IDS, indirect=True)
async def test_promotion_quality(promotion_case, promotion_runner, judge_llm, baseline_recorder):
```

- **Golden mode:** Feed script + claims into mock `_promote_to_global()`. Assert:
  - GLOBAL chunks created with `meta.scope == "GLOBAL"` 
  - `job_id is None` for each
  - `meta.source_type == "COMPRESSED_FACT"`
  - `meta.original_job_id` populated
  - `meta.fact_category` is valid domain string
  - Count within expected range
  - No fact-text duplicates
- **Live mode (`--live`):** Call real `_promote_to_global()` (requires mock job + vector store). LLM judge on:
  - `fact_accuracy` — is each compressed fact faithful to source?
  - `compression_fidelity` — is each fact atomic (not compound)?
  - `metadata_completeness` — are all required meta fields populated?
- **Baseline recording:** Same pattern as `test_outcome_factcheck.py` — record to `baseline_recorder`

### 7. `test_memory_retrieval.py`

**Eval 9.2 — Retrieval Relevance** (golden mode only)

```python
@pytest.mark.eval
@pytest.mark.parametrize("retrieval_case", RETRIEVAL_CASE_IDS, indirect=True)
async def test_retrieval_global_relevance(retrieval_case, retrieval_runner):
```

- Seed GLOBAL store from fixture + LOCAL chunks from fixture
- Run ContextBuilder-style Sequential Assembly (two `semantic_search` calls → merge)
- Assert:
  - `=== SYSTEM INTEL ===` section present and non-empty
  - `=== CURRENT RUN RESEARCH ===` section present and non-empty
  - No GLOBAL fact text appears in CURRENT RUN RESEARCH section (label integrity)
  - If GLOBAL facts duplicate LOCAL facts (exact text match), deduplicated — only one copy in merged output
  - Each GLOBAL fact in SYSTEM INTEL has `source_type=COMPRESSED_FACT` in its metadata

### 8. `test_outcome_memory.py`

**Eval 9.3 + 9.4 — Downstream Benefit and Safety/Integrity**

```python
@pytest.mark.eval
@pytest.mark.parametrize("memory_case", MEMORY_CASE_IDS, indirect=True)
async def test_downstream_benefit(memory_case, eval_runner, judge_llm, baseline_recorder):
```

**9.3 — Downstream benefit:**
- Golden mode: seed GLOBAL store from fixture's `global_facts`. Run Job B (copywriter + fact-check). Assert:
  - `AssembledContext` contains `=== SYSTEM INTEL ===` section
  - Each `must_include_fact` from fixture appears in script
  - Each `must_avoid` pattern does not appear in script
  - Expected known-verdict claims pass fact-check (SUPPORTED)
- Live mode (`--live`): Chain Job A → COMPLETED → Job B. LLM judge scores Job B script on `factual_grounding` dimension of `SCRIPT_RUBRIC`. Compare to baseline.

**9.4 — Safety and Integrity:**
- **Hallucination propagation (F-M1):** Seed distorted GLOBAL fact ("GDP grew 3.8%" vs actual "2.8%"). Run Job B. Assert the planted fact receives UNSUPPORTED verdict from Red Team.
- **Domain leakage (F-M3):** Seed economics GLOBAL facts. Run health-topic Job B. Assert no economics GLOBAL texts appear in `AssembledContext` or script.

**Parametrized cases:**

| Case ID | Sub-eval | Source | Expectation |
|---------|----------|--------|-------------|
| `M-001` (paired) | 9.3 | `global_memory_fixtures.json` synthetic_pairs | All assertions pass |
| `M-002` (paired) | 9.3 | Existing economics pair (H-001 → R-002) — future | All assertions pass |
| `M-A1` | 9.4 | adversarial_cases: hallucination-propagation | Red Team catches wrong fact |
| `M-A2` | 9.4 | adversarial_cases: domain-leakage | No economics facts in health output |

**Ruby — New MEMORY rubric (live mode only):**

```python
MEMORY_RUBRIC = _build_rubric(
    name="memory",
    dimensions=[
        ("fact_accuracy", "Are the GLOBAL-compressed facts faithful to their source claims? No distortion, no hallucination."),
        ("compression_fidelity", "Is each GLOBAL fact atomic (single statement)? Compound facts score 0."),
        ("metadata_completeness", "Are scope, source_type, original_job_id, fact_category, confidence, and ingested_at all populated?"),
        ("downstream_grounding", "Does the downstream script benefit from GLOBAL facts? Higher grounding score when GLOBAL context is used accurately."),
    ],
    weights={
        "fact_accuracy": 0.40,
        "compression_fidelity": 0.25,
        "metadata_completeness": 0.15,
        "downstream_grounding": 0.20,
    },
)
```

## Files to modify

### 9. `evals-criteria.md`

Add **Eval 9 — Long-Term Memory** section after Eval 8:

- **9.1 Promotion Quality** — criteria table with structural checks + LLM judge dimensions and thresholds
- **9.2 Retrieval Relevance** — criteria table for Sequential Assembly structural assertions
- **9.3 Downstream Benefit** — criteria table with golden-mode assertions + live LLM judge thresholds
- **9.4 Safety & Integrity** — criteria for hallucination detection and domain isolation

Add 2 new failure codes to taxonomy:

| Code | Name | Stage | Severity | Description |
|------|------|-------|----------|-------------|
| F-M1 | GLOBAL hallucination propagation | Memory | Critical | Compressed fact in GLOBAL store is inaccurate; propagates to future jobs |
| F-M3 | Domain leakage | Memory | Medium | GLOBAL facts from unrelated domains retrieved and used in wrong context |

### 10. `eval-contracts.md`

Add **Contract 10 — Cross-Job GLOBAL Memory Persistence**:

"GLOBAL-scoped facts promoted from completed jobs must survive garbage collection, be queryable by future jobs with `job_id=None, scopes=["GLOBAL"]`, and be clearly labelled as `=== SYSTEM INTEL ===` in assembled context."

| # | Test | Guards | Severity |
|---|---|---|---|
| 10 | `test_global_memory_persistence` | GLOBAL facts survive GC, queryable cross-job | High |

### 11. `contracts/_index.md`

Add:

## Eval 9 — Long-Term Memory

| File | Sub-eval | Criteria ref |
|------|----------|-------------|
| [`09-memory-promotion-quality.md`](09-memory-promotion-quality.md) | 9.1 Promotion Quality | Contract 10 |
| [`09-memory-retrieval-relevance.md`](09-memory-retrieval-relevance.md) | 9.2 Retrieval Relevance | Contract 10 |
| [`09-memory-downstream-benefit.md`](09-memory-downstream-benefit.md) | 9.3 Downstream Benefit | Contract 10 |
| [`09-memory-safety-integrity.md`](09-memory-safety-integrity.md) | 9.4 Safety & Integrity | Contract 10 |

### 12. `schemas.py`

Add to `RubricSet`:

```python
MEMORY = "memory"
```

Add to `TraceType`:

```python
CROSS_JOB = "cross_job"
```

### 13. `golden_dataset.json`

Add 2 new cases (prefix `C-` for cross-job):

- **C-001:** `trace_type: "cross_job"`, `domain: "economics"`, topic "U.S. Federal Reserve Monetary Policy 2024-2025"
- **C-002:** `trace_type: "cross_job"`, `domain: "economics"`, topic "Impact of Federal Reserve Policy on Emerging Market Debt 2025"

C-002 includes `reference_outputs` that assert the GLOBAL facts from C-001 are used correctly.

### 14. `rubrics.py`

Add `MEMORY_RUBRIC` to `RUBRICS` registry (as defined above).

## Execution order

| Step | File(s) | Depends on |
|------|---------|------------|
| 1 | `evals-criteria.md` (add Eval 9 section + failure codes) | Nothing — document-only |
| 2 | `eval-contracts.md` (add Contract 10) | Step 1 |
| 3 | `contracts/_index.md` (add Eval 9 entry) | Step 1 |
| 4 | 4 contract files: `09-memory-*` | Step 2 |
| 5 | `schemas.py` (RubricSet + TraceType additions) | Nothing |
| 6 | `rubrics.py` (MEMORY_RUBRIC + registry) | Step 5 |
| 7 | `fixtures/global_memory_fixtures.json` | Step 1 (criteria defined) |
| 8 | `golden_dataset.json` (C-001, C-002) | Step 7 (seed data) |
| 9 | `test_memory_promotion.py` | Steps 4, 6, 7 |
| 10 | `test_memory_retrieval.py` | Steps 4, 7 |
| 11 | `test_outcome_memory.py` | Steps 4, 6, 7, 8 |

Steps 1–8 can be partially parallelized. Steps 9–11 depend on fixture and schema being in place.
