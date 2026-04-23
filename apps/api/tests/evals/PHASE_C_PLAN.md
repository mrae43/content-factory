# Phase C — Dataset Population: Implementation Plan

> Generated: 2026-04-22
> Status: DONE (Batch 1 ✅, Batch 2 ✅, Batch 3 ✅, Batch 4 ✅, Batch 5 ✅, Batch 6 ✅)
> Depends on: Phase A (schemas.py, rubrics.py) ✅, Phase B (conftest.py) ✅

---

## 0. Current State

| Phase | Status | Evidence |
|-------|--------|----------|
| A — Data Layer | **DONE** | schemas.py (360 lines), golden_entry_schema.json (515 lines), rubrics.py (237 lines) |
| B — Infrastructure | **DONE** | conftest.py (454 lines), 8 fixtures, 4 helper classes |
| **C — Dataset Population** | **DONE** | golden_dataset.json = 23 cases (H-001..H-004, R-001..R-004, E-001..E-003, F-001..F-004, N-001..N-004, M-001..M-004) |
| D — Outcome Evals | Blocked on C | No test files yet |
| E — Trajectory Evals | Blocked on C | No test files yet |
| F — Negative Goldens | Blocked on C | No test files yet |
| G — Baselines | Blocked on D-F | baselines.json is empty shell |

### Batch Progress

| Batch | Category | Count | Status | Validated |
|-------|----------|-------|--------|-----------|
| 1 | Happy Path (H-001..H-004) | 4 | ✅ DONE | Pydantic pass |
| 2 | Revision Loop (R-001..R-004) | 4 | ✅ DONE | Pydantic pass |
| 3 | Escalation (E-001..E-003) | 3 | ✅ DONE | Pydantic pass |
| 4 | Fallback Chain (F-001..F-004) | 4 | ✅ DONE | Pydantic pass |
| 5 | Negative Golden (N-001..N-004) | 4 | ✅ DONE | Pydantic pass |
| 6 | Minimal Edge (M-001..M-004) | 4 | ✅ DONE | Pydantic pass |

---

## 1. Decisions Resolved

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ID format | 3-digit: `H-001`, `R-001` | Matches schemas.py regex `^[HRFENM]-\d{3}$` |
| Content source | Write realistic content from scratch | Cases need real raw_text, facts, claims for eval quality |
| Validation | Python one-liner after each batch | Catches enum mismatches immediately |
| Batch sizing | One tool call per category (4-4-3-4-4-4) | Manageable scope, category-aligned |
| Batch order | H → R → E → F → N → M | Complexity increases gradually |
| Validation method | Inline Python `GoldenDataset.model_validate_json()` | No extra files needed |

---

## 2. Batch Plan (6 Batches)

### Batch 1: Happy Path (H-001 to H-004) ✅ DONE

| ID | Domain | Topic | Difficulty | Category |
|----|--------|-------|------------|----------|
| H-001 | economics | BRICS De-dollarization | easy | factual_accuracy |
| H-002 | technology | EU AI Act Regulation | easy | factual_accuracy |
| H-003 | health | mRNA Vaccine Development | medium | hallucination_trap |
| H-004 | climate_science | Arctic Ice Melt Rates | medium | conflicting_evidence |

**Trace structure (all 4):** 5-state sequence, 3 agent calls, standard tool calls.

```
PENDING → RESEARCHING → FACT_CHECKING_RESEARCH → SCRIPTING → FACT_CHECKING_SCRIPT (approved)
```

**Validation command:**
```bash
python -c "import json; from tests.evals.schemas import GoldenDataset; data=json.load(open('tests/golden/golden_dataset.json')); GoldenDataset(cases=data); print(f'Valid: {len(data)} cases')"
```

### Batch 2: Revision Loop (R-001 to R-004) ✅ DONE

| ID | Domain | Topic | Difficulty | Revision Trigger |
|----|--------|-------|------------|-----------------|
| R-001 | geopolitics | Taiwan Semiconductor Monopoly | medium | 1 UNSUPPORTED (fabricated %) |
| R-002 | economics | Global Debt Crisis 2024 | hard | 2 CONTESTED (conflicting data) |
| R-003 | social_issues | Universal Basic Income Pilots | medium | String feedback (human reject) |
| R-004 | history | Library of Alexandria Destruction | hard | UNSUPPORTED causal claim |

**Trace structure:** 7-9 state sequences with feedback_history, optimization outcomes.

**Key complexity:**
- R-001: 7 states, 1 structured_claims feedback, optimizer patches 1 claim
- R-002: 9 states, 2 structured_claims feedback entries, 3 revision cycles
- R-003: 7 states, string feedback (NOT structured_claims), copywriter re-drafts
- R-004: 7 states, optimizer removes causal claim

### Batch 3: Escalation (E-001 to E-003)

| ID | Domain | Topic | Difficulty | Escalation Trigger |
|----|--------|-------|------------|-------------------|
| E-001 | technology | Quantum Computing Breakthroughs | medium | 3 failed revision cycles |
| E-002 | medicine | COVID Origins Lab Leak Theory | hard | Red Team LLM parse failure |
| E-003 | ai_ethics | AI Sentience Claims | hard | No verifiable evidence exists |

**Trace structure:** Terminal `HUMAN_REVIEW_NEEDED`.

**Key complexity:**
- E-001: Full 3-cycle revision loop, 3 feedback_history entries
- E-002: Direct ESCALATE from Red Team, no revision cycle
- E-003: No evidence found for any claim, ESCALATE

### Batch 4: Fallback Chain (F-001 to F-004) ✅ DONE

| ID | Domain | Topic | Difficulty | Tool A → Tool B |
|----|--------|-------|------------|-----------------|
| F-001 | cybersecurity | Zero-Day Exploit 2025 | medium | TavilySearch → [] → RAW-CONTEXT only |
| F-002 | history | Obscure Medieval Trade Route | hard | semantic_search → [] → ERROR → FAILED |
| F-003 | technology | Unreleased Product Features | hard | All evidence below threshold → ESCALATE |
| F-004 | economics | Contested GDP Figures | medium | Optimizer can't patch → best-effort |

**Key complexity:**
- F-001: `fallback_expectations` with TraceStep entries, pipeline still completes
- F-002: Short-circuits at RESEARCHING, final_status = `FAILED`
- F-003: Reaches FACT_CHECKING_SCRIPT then escalates
- F-004: `fallback_expectations` for optimizer partial patch

### Batch 5: Negative Golden (N-001 to N-004) ✅ DONE

| ID | Safety Constraint | Input Pattern |
|----|-------------------|---------------|
| N-001 | PII_MUST_NOT_PROPAGATE | SSN + full name + address in pre_context |
| N-002 | MEDICAL_DISCLAIMER_REQUIRED | Unverified cancer treatment topic |
| N-003 | REFUSE_HARMFUL_CONTENT | Weaponization instructions request |
| N-004 | DEFAMATION_PREVENTION | Unverified accusations against real person |

**Key complexity:**
- `safety_constraint` field with `pii_entities`, `check_stages`, `pass_criteria`
- `assertions` list with `NegativeAssertion` entries (stage + field + must_not_contain)
- `difficulty: adversarial`, `trace_type: negative_golden`

### Batch 6: Minimal Edge (M-001 to M-004) ✅ DONE

| ID | Edge Case | Input Pattern |
|----|-----------|---------------|
| M-001 | Empty pre_context | raw_text: "", source_urls: [] |
| M-002 | Very long pre_context | 50,000+ character raw_text |
| M-003 | Unicode/adversarial | Mixed CJK, RTL, emoji, zero-width chars |
| M-004 | Single-sentence topic | topic: "cats" (3 chars) |

**Key complexity:**
- M-001: final_status = `FAILED`, empty chunking
- M-002: Large raw_text content, pipeline completes but slow
- M-003: Unicode content preservation assertions
- M-004: Minimal input, generic output expected

---

## 3. Field-by-Field Reference

### Required fields on every GoldenCase entry:
```
id               — "H-001" format (regex: ^[HRFENM]-\d{3}$)
trace_type       — "happy_path" | "revision_loop" | "fallback_chain" | "negative_golden"
category         — "factual_accuracy" | "hallucination_trap" | "conflicting_evidence" | "edge_case_minimal" | "edge_case_long" | "safety_refusal" | "pii_protection"
domain           — free string (e.g. "economics", "geopolitics")
difficulty       — "easy" | "medium" | "hard" | "adversarial"
input.topic      — 1-500 chars
input.pre_context.raw_text      — Optional[str]
input.pre_context.source_urls   — List[str] (default [])
input.pre_context.target_audience — default "General"
input.pre_context.guardrail_strictness — default "High"
input.strict_compliance_mode    — default true
trace_spec.expected_state_sequence — List of JobStatus enum values
trace_spec.expected_agent_calls   — List[AgentCallSpec] (default [])
trace_spec.expected_tool_calls    — List[ToolCallSpec] (default [])
trace_spec.expected_feedback_history — List[FeedbackHistorySpec] (default [])
trace_spec.fallback_expectations   — List[TraceStep] (default [])
trace_spec.rejection_expectations  — List[TraceStep] (default [])
trace_spec.note                    — Optional[str]
expected_outcomes.final_status     — Required string
expected_outcomes.research         — Optional[ResearchOutcome]
expected_outcomes.script           — Optional[ScriptOutcome]
expected_outcomes.fact_check       — Optional[FactCheckOutcome]
expected_outcomes.optimization     — Optional[OptimizationOutcome]
expected_outcomes.assertions       — List[NegativeAssertion] (default [])
```

### Optional fields (can be null/omitted):
```
scoring                           — Optional[ScoringSpec]
safety_constraint                 — Optional[SafetyConstraint] (only for negative_golden)
metadata                          — defaults to CaseMetadata()
```

### JobStatus enum values (exact strings):
```
"PENDING", "RESEARCHING", "FACT_CHECKING_RESEARCH", "SCRIPTING",
"FACT_CHECKING_SCRIPT", "ASSET_GENERATION", "COMPLETED", "FAILED",
"HUMAN_REVIEW_NEEDED"
```

### AgentActionStatus values (for expected_verdict in AgentCallSpec):
```
"SUCCESS", "REVISION_NEEDED", "ESCALATE", "ERROR"
```

### EvalVerdict values (for claims_with_known_verdicts):
```
"SUPPORTED", "CONTESTED", "UNSUPPORTED", "UNCERTAIN"
```

---

## 4. Post-Steps

After all 6 batches are written and validated:

1. **Ruff format + lint:** `ruff format . && ruff check . --fix`
2. **Update GOLDEN_DATASET_FOUNDATION.md §7:** Mark Phase C as `✅ DONE`
3. **Run validation checklist from §9:** All 12 items should pass

---

## 5. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Enum string mismatch | Pydantic validation fails | Validate after each batch, fix immediately |
| Missing required field | Validation fails | Reference field list above per entry |
| raw_text too short | Research produces insufficient context | Write 200-800 words per case |
| State sequence wrong | Trajectory tests fail later | Match exact orchestrator state machine |
| Scoring rubric_set wrong | Outcome tests can't score | Match RubricSet enum: research/script/fact_check/optimizer |
