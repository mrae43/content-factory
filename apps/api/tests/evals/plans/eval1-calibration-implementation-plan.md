# Eval 1.3 Implementation Plan — `topic_relevance` Calibration

## Design decisions (resolved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dataset size | 50 chunks; expand to 200 if initial LOW recall < 0.75 | Balances annotation cost vs statistical meaning |
| Mode | Golden-only, no `--live` | Deterministic function (`_derive_topic_relevance`); live mode adds no value |
| Fixture location | `relevance` section in `eval1_research.json` | Placeholder already reserved; single fixture per desk |
| Score storage | Raw `similarity_score` + `human_label`; derive at test time | Tests actual production code path through `_derive_topic_relevance` |
| Construction | Engineered scores probing threshold boundaries (0.75, 0.50) | Targeted stress-testing of calibration |
| Threshold guard | `QualityCorpusValidator` at fixture load time | `pytest.exit(2)` on mismatch between stored and current thresholds |
| Test structure | Single function, one confusion matrix, all three assertions | Shared computation; reports all failures |
| Test file | New `test_eval1_calibration.py` | Matches pattern of 1.1 and 1.2 |
| `evals-criteria.md` | Fix Dataset Recommendations: 50, not 200 | Contract 4 compliance |

## Files to modify

### 1. `tests/evals/schemas.py` — Add schemas

Insert new section after `# 2c. CHUNK QUALITY EVAL SCHEMAS (Eval 1.2)`:

```python
# ==========================================
# 2d. CALIBRATION EVAL SCHEMAS (Eval 1.3)
# ==========================================


class RelevancyThresholds(BaseModel):
    high_threshold: float = Field(
        ..., description="Score threshold for HIGH relevance"
    )
    medium_threshold: float = Field(
        ..., description="Score threshold for MEDIUM relevance"
    )
    built_against: str = Field(
        ...,
        description="Canonical repr of thresholds at build time, "
        "e.g. '[(0.75, HIGH), (0.5, MEDIUM)]'",
    )


class RelevanceChunk(BaseModel):
    chunk_id: str
    chunk_text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    human_label: str = Field(..., pattern=r"^(HIGH|MEDIUM|LOW)$")


class RelevancyCalibrationSet(BaseModel):
    description: str
    threshold_config: RelevancyThresholds
    chunks: list[RelevanceChunk]

    @field_validator("chunks")
    @classmethod
    def validate_count(cls, v):
        if len(v) == 0:
            raise ValueError("Calibration set is empty")
        return v
```

Update `Eval1ResearchFixture`:

```python
class Eval1ResearchFixture(BaseModel):
    eval_version: str
    schema_version: str
    coverage_cases: list[ResearchingCase]
    quality_corpus: QualityCorpus
    relevance: RelevancyCalibrationSet  # was: list[dict]
```

### 2. `tests/evals/fixtures/eval1_research.json` — Populate `relevance` section

Replace `"relevance_cases": []` with a `"relevance"` object containing 50 chunks.

**Chunk distribution (50 chunks):**

| Human label | Count | Score range | Purpose |
|---|---|---|---|
| HIGH (above HIGH) | 14 | 0.80 – 0.98 | Easy HIGH — tests clear cases |
| HIGH (boundary) | 4 | 0.75 – 0.79 | Near HIGH/MEDIUM boundary — tests edge precision |
| MEDIUM (mid) | 6 | 0.55 – 0.74 | Well within MEDIUM range |
| MEDIUM (boundary) | 6 | 0.50 – 0.54 | Near MEDIUM/LOW boundary — tests edge calibration |
| LOW (boundary) | 6 | 0.45 – 0.49 | Near MEDIUM/LOW boundary — tests LOW recall |
| LOW (low) | 14 | 0.05 – 0.44 | Easy LOW — obvious rejection |

**Boundary design principle:** For edge cases near 0.75 and 0.50, human labels should sometimes disagree with the score-based prediction. For example:
- `score=0.73, human_label="HIGH"` → algorithm says MEDIUM, human says HIGH (tests LOW recall)
- `score=0.76, human_label="MEDIUM"` → algorithm says HIGH, human says MEDIUM (tests HIGH precision)
- `score=0.48, human_label="MEDIUM"` → algorithm says LOW, human says MEDIUM (tests LOW recall)
- `score=0.51, human_label="LOW"` → algorithm says MEDIUM, human says LOW (tests MEDIUM precision as false positive for LOW)

### 3. `tests/evals/conftest.py` — Add validator + fixtures

**a) Add `_validate_relevance_thresholds` function:**

```python
def _validate_relevance_thresholds(calibration_set: RelevancyCalibrationSet):
    from app.services.context_builder import TOPIC_RELEVANCE_THRESHOLDS

    current = sorted(
        [(t, l) for t, l in TOPIC_RELEVANCE_THRESHOLDS],
        key=lambda x: -x[0],
    )
    stored_cfg = calibration_set.threshold_config
    stored = sorted(
        [
            (stored_cfg.high_threshold, "HIGH"),
            (stored_cfg.medium_threshold, "MEDIUM"),
        ],
        key=lambda x: -x[0],
    )

    if current != stored:
        pytest.exit(
            f"Relevance calibration set built against thresholds {stored} "
            f"but current TOPIC_RELEVANCE_THRESHOLDS are {current}. "
            f"Regenerate the calibration fixture or reconcile thresholds.",
            returncode=2,
        )
```

**b) Update `_load_eval1_research()` to validate thresholds at load time:**

```python
def _load_eval1_research() -> dict:
    if not _EVAL1_RESEARCH_FIXTURES_PATH.exists():
        return {"coverage_cases": [], "quality_corpus": None, "relevance": None}
    data = json.loads(_EVAL1_RESEARCH_FIXTURES_PATH.read_text(encoding="utf-8"))
    if data.get("relevance"):
        from tests.evals.schemas import RelevancyCalibrationSet
        rel = RelevancyCalibrationSet(**data["relevance"])
        _validate_relevance_thresholds(rel)
        data["relevance"] = rel
    return data
```

**c) Add `calibration_set` fixture:**

```python
@pytest.fixture
def calibration_set() -> RelevancyCalibrationSet:
    data = _load_eval1_research()
    rel = data.get("relevance")
    if rel is None:
        pytest.exit(
            "relevance section missing from eval1_research.json", returncode=2
        )
    return rel
```

**d) Add `RelevancyCalibrationSet` to the existing conftest imports.**

### 4. `tests/evals/test_eval1_calibration.py` — New test file

```python
"""
Eval 1.3 — topic_relevance Calibration.

Reference-based eval comparing pipeline's auto-assigned topic_relevance
labels against human-labelled ground truth for a held-out set of 50 chunks.

Golden-only (no --live mode). Confusion matrix computed from
_derive_topic_relevance(similarity_score) vs human_label.
"""

import pytest

from app.services.context_builder import _derive_topic_relevance
from tests.evals.schemas import RelevancyCalibrationSet


@pytest.mark.eval
async def test_topic_relevance_calibration(
    calibration_set: RelevancyCalibrationSet,
    baseline_recorder,
):
    chunks = calibration_set.chunks

    # Build confusion matrix: predicted vs human
    tp_h = tn_h = fp_h = fn_h = 0  # HIGH positives
    tp_l = tn_l = fp_l = fn_l = 0  # LOW positives
    correct = 0

    for chunk in chunks:
        predicted = _derive_topic_relevance(chunk.similarity_score)
        actual = chunk.human_label

        if predicted == actual:
            correct += 1

        # HIGH confusion
        if predicted == "HIGH" and actual == "HIGH":
            tp_h += 1
        elif predicted == "HIGH" and actual != "HIGH":
            fp_h += 1
        elif predicted != "HIGH" and actual == "HIGH":
            fn_h += 1
        else:
            tn_h += 1

        # LOW confusion
        if predicted == "LOW" and actual == "LOW":
            tp_l += 1
        elif predicted == "LOW" and actual != "LOW":
            fp_l += 1
        elif predicted != "LOW" and actual == "LOW":
            fn_l += 1
        else:
            tn_l += 1

    n = len(chunks)
    accuracy = correct / n
    high_precision = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0.0
    low_recall = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0

    violations = []
    if accuracy < 0.80:
        violations.append(f"accuracy {accuracy:.2f} < 0.80")
    if high_precision < 0.85:
        violations.append(f"HIGH precision {high_precision:.2f} < 0.85")
    if low_recall < 0.75:
        violations.append(f"LOW recall {low_recall:.2f} < 0.75")

    baseline_recorder.record_case_score(
        "topic_relevance_calibration",
        "calibration",
        {
            "accuracy": accuracy,
            "high_precision": high_precision,
            "low_recall": low_recall,
            "n": n,
            "confusion_matrix": {
                "high": {"tp": tp_h, "fp": fp_h, "fn": fn_h, "tn": tn_h},
                "low": {"tp": tp_l, "fp": fp_l, "fn": fn_l, "tn": tn_l},
            },
        },
    )

    if violations:
        pytest.fail(f"Calibration failed: {'; '.join(violations)}")
```

### 5. `tests/evals/contracts/01-research-topic-relevance-calibration.md` — Fix dataset size

Change line 9 from:

```
- Held-out set of 200 chunks with human-labelled relevance (HIGH/MEDIUM/LOW)
```

to:

```
- Held-out set of 50 chunks with human-labelled relevance (HIGH/MEDIUM/LOW)
```

### 6. `tests/evals/evals-criteria.md` — Fix dataset size

Change line 383 from:

```
| Research chunk labels | 200 chunks | Human-labelled `topic_relevance` | 1.3 calibration |
```

to:

```
| Research chunk labels | 50 chunks | Human-labelled `topic_relevance` | 1.3 calibration |
```

## Files NOT to modify

- `test_outcome_research.py`, `test_outcome_script.py`, etc. (legacy evals)
- `judge.py`, `rubrics.py` (separate scoring system)
- `golden_dataset.json` or `golden_entry_schema.json`
- `assertions.py` (1.3 has its own computation inline)
- `chunk_quality_scorer.py` (1.3 doesn't use it)

## Open infrastructure assumptions

1. **`_derive_topic_relevance` is importable** from `app.services.context_builder` without triggering a chain of production imports. Already the case — it's a pure function with no side effects.

2. **No pgvector dependency** — the fixture stores pre-computed `similarity_score` values, avoiding the need for real embeddings or a running database. Matches golden-mode contract.

3. **Baseline recording format:** The existing `BaselineRecorder` stores arbitrary dicts per case. Records confusion matrix counts and metric values under rubric key `"calibration"`. No schema changes needed.

4. **`pytest.exit(2)`** is the correct mechanism for fixture corruption — it prevents test execution entirely rather than producing a misleading all-pass result.

## Execution order

1. Add schemas to `schemas.py`
2. Populate `relevance` section in `eval1_research.json` with 50 chunks
3. Add validator + fixture to `conftest.py`
4. Write `test_eval1_calibration.py`
5. Fix dataset size in `contracts/01-research-topic-relevance-calibration.md`
6. Fix dataset size in `evals-criteria.md`
7. Verify: `uv run pytest tests/evals/test_eval1_calibration.py -m eval -v`
8. Verify: `nx lint api && nx test:unit api` — no regressions
