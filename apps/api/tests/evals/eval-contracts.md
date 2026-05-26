# Eval Contracts

The eval suite must satisfy the following high-level contracts. Each contract
is a forward-facing "must" assertion — the individual files in [`contracts/`](contracts/)
derive their criteria from these.

---

## Contract 1 — `JobStatus` enum completeness

The eval `JobStatus` enum (`tests/evals/schemas.py:58`) and the
`golden_entry_schema.json` `JobStatus` enum MUST match the production
`JobStatusEnum` (`app/schemas/shorts.py:41`) — all 11 values: `PENDING`,
`RESEARCHING`, `RETRIEVAL`, `FACT_CHECKING_RESEARCH`, `SCRIPTING`,
`FACT_CHECKING_SCRIPT`, `FORMATTING`, `ASSET_GENERATION`, `COMPLETED`,
`FAILED`, `HUMAN_REVIEW_NEEDED`.

Any golden case referencing any of those states in `expected_state_sequence`
MUST pass Pydantic validation and JSON Schema validation at import time.

**References:** `evals-criteria.md` Eval 7.1, `app/schemas/shorts.py:41`

---

## Contract 2 — `storyboard_fields` default field name

`ScriptOutcome.storyboard_fields` (`tests/evals/schemas.py:210-213`) MUST
default to `["visual_description", "audio_cue"]`.

`assert_storyboard_fields()` (`tests/evals/assertions.py:71`) MUST look for
`visual_description`, not `visual_prompt`.

Existing golden dataset entries that hardcode `storyboard_fields` MUST use
`visual_description`.

**References:** `evals-criteria.md` Eval 6.2, `CONTEXT.md:142`

---

## Contract 3 — `CaseCategory` enum cross-check

The `CaseCategory` enum in `golden_entry_schema.json` (`$defs`) MUST contain
exactly the same values as `tests/evals/schemas.py:27-35`:
`FACTUAL_ACCURACY`, `HALLUCINATION_TRAP`, `CONFLICTING_EVIDENCE`,
`EDGE_CASE_MINIMAL`, `EDGE_CASE_LONG`, `SAFETY_REFUSAL`, `PII_PROTECTION`.

Any descrepancy between the two schemas counts as a schema drift failure.

**References:** `tests/evals/schemas.py:27-35`, `tests/golden/schemas/golden_entry_schema.json`

---

## Contract 4 — `topic_relevance` calibration dataset size

The `topic_relevance` calibration dataset size MUST be consistent across
the `evals-criteria.md` document: the same number SHALL appear in both
the Eval 1.3 criterion and the Dataset Recommendations table.

**References:** `evals-criteria.md` Eval 1.3 (line 52) and Dataset Recommendations
(line 383)

---

## Contract 5 — Platform Compliance YouTube coverage

Eval 6.3's Platform Compliance test table MUST include a row for YouTube
(with the corresponding carousel slide character limit check) alongside
Instagram, TikTok, Twitter/X, and LinkedIn.

**References:** `evals-criteria.md` Eval 6.3, `CONTEXT.md:133-135`,
`app/schemas/shorts.py:90-95`

---

## Contract 6 — `source_type` coverage for all four values

The eval suite MUST test `source_type` correctness for all four source
types: `WEB_SEARCH` (Tavily search), `URL_EXTRACT` (Tavily extract from
user-provided source URLs), `INFERRED` (Retrieval Desk synthesis fallback),
and `USER_PROVIDED` (user-supplied URLs/raw text).

`INFERRED` is the default applied by ContextBuilder when source_type is not
explicitly set in chunk metadata.

`USER_PROVIDED` chunks ingested from user-supplied raw text in the PENDING
stage MUST carry the correct `source_type`.

`URL_EXTRACT` chunks ingested from user-supplied source URLs in the
RESEARCHING stage MUST carry the correct `source_type`.

**References:** `evals-criteria.md` Eval 1.1, `CONTEXT.md:11`

---

## Contract 7 — Production Studio / Visual Asset type coverage

The eval suite MUST cover the `ASSET_GENERATION` stage and verify that
all six `AssetTypeEnum` values (`CAROUSEL_SLIDE`, `VISUAL_VEO`,
`AUDIO_LYRIA`, `VOICEOVER`, `SUBTITLE_JSON`, `DATA_CHART`) are produced
with correct metadata (prompt, timing, SynthID watermark via
`AssetRenderMeta`).

**References:** `app/schemas/shorts.py:74` `AssetTypeEnum`, `CONTEXT.md:104-112`

---

## Contract 8 — `FACT_CHECKING_RESEARCH` glossary documentation

The domain glossary (`CONTEXT.md`) MUST document the `FACT_CHECKING_RESEARCH`
pipeline state (describing it as "Source Verification" per the production
enum's editorial-name comment), OR the state MUST be removed from the
production `JobStatusEnum`, the eval `JobStatus` enum, and the glossary.

**References:** `CONTEXT.md:118-129`, `app/schemas/shorts.py:41`

---

## Contract 9 — `FORMATTING` transition in golden dataset

At least one golden dataset happy-path case MUST include `FORMATTING` and
`ASSET_GENERATION` in its `expected_state_sequence`, exercising the full
canonical sequence: `PENDING → RESEARCHING → RETRIEVAL → SCRIPTING →
FACT_CHECKING_SCRIPT → FORMATTING → ASSET_GENERATION → COMPLETED`.

The eval `JobStatus` enum MUST recognise `FORMATTING` (see Contract 1).

**References:** `evals-criteria.md` Eval 7.1, `tests/golden/golden_dataset.json`

---

## Summary

| # | Test | Guards | Severity |
|---|---|---|---|
| 1 | `test_jobstatus_enum_completeness` | Schema validation at import time | High |
| 2 | `test_storyboard_fields_default_name` | Correct field name in assertions | Medium |
| 3 | `test_casecategory_enum_crosscheck` | Schema drift between Pydantic and JSON | Medium |
| 4 | `test_calibration_dataset_size_consistent` | Internal document consistency | Low |
| 5 | `test_youtube_platform_compliance` | Missing platform coverage | Medium |
| 6 | `test_all_four_source_types` | WEB_SEARCH, URL_EXTRACT, INFERRED, USER_PROVIDED coverage | High |
| 7 | `test_asset_generation_stage_coverage` | Missing stage coverage | Medium |
| 8 | `test_fact_checking_research_glossary_documented` | Stale domain documentation | Low |
| 9 | `test_formatting_in_golden_dataset` | Happy-path sequence never exercised | Medium |
