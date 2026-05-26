# Content Factory — Eval Criteria

## Guiding Principles

1. **Eval at stage boundaries, not just end-to-end.** Each desk is a failure surface. A broken Retrieval Desk poisons every downstream component; catching it early is cheaper than debugging a bad script.
2. **Separate correctness from quality.** Correctness is binary and deterministic (did the status transition correctly?). Quality is graded and often requires an LLM judge (is the Refined Context coherent?).
3. **Match eval type to what can actually be verified.** Use rule-based evals for structure, format, and schema. Use LLM-as-judge for semantic quality. Reserve human evaluation for guardrail calibration and contested verdict ground truth.
4. **Test each Guardrail Strictness level independently.** Low/Medium/High encode fundamentally different acceptance criteria; a single passing eval at Medium does not guarantee Low or High behavior is correct.
5. **Plant adversarial cases.** The Fact-Check Desk is only valuable if it catches false claims. Seed known-false statements into scripts and measure recall.

---

## Eval 1 — Research Desk

**Pipeline stage:** `RESEARCHING`  
**Trigger:** After Tavily search results are chunked, embedded, and ingested into pgvector.

### 1.1 Coverage (Rule-based)

| Criterion | Method | Pass condition |
|---|---|---|
| Minimum chunk count | Count `source_type=WEB_SEARCH` chunks in pgvector for the Story | ≥ 5 chunks per story |
| Source diversity | Count distinct root domains across chunks | ≥ 3 unique domains |
| No duplicate chunks | Cosine similarity between any two chunk embeddings | No pair with similarity > 0.97 |
| `scope` correctly set | Assert all new chunks carry `scope=LOCAL` | 100% |
| `source_type` correctly set | Assert all Tavily-originated chunks carry `source_type=WEB_SEARCH` | 100% |

### 1.2 Chunk Quality (LLM-as-judge)

**Prompt template:**

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

**Thresholds:** Mean score ≥ 3.5 per dimension across all chunks for a Story. Flag stories where > 20% of chunks score < 3 on relevance.

### 1.3 `topic_relevance` Calibration (Reference-based)

For a held-out set of 50 chunks with human-labelled relevance (`HIGH/MEDIUM/LOW`), measure the pipeline's auto-assigned `topic_relevance` against the labels.

| Metric | Target |
|---|---|
| Accuracy | ≥ 0.80 |
| HIGH precision | ≥ 0.85 (false positives hurt synthesis quality) |
| LOW recall | ≥ 0.75 (missed low-quality chunks pollute context) |

---

## Eval 2 — Retrieval Desk

**Pipeline stage:** `RETRIEVAL`  
**Three sub-evals map to the three sequential steps.**

### 2.1 Synthesis — Refined Context Quality (LLM-as-judge)

After the ResearchAgent produces the Refined Context (800–1500 words):

| Criterion | Prompt focus | Pass condition |
|---|---|---|
| Length compliance | Rule-based word count | 800–1500 words |
| Factual grounding | "Every claim in this narrative is traceable to at least one of the provided chunks. True/False + explanation." | True, or flagged claims < 5% of total |
| Coherence | "Rate 1–5: does this read as a unified narrative rather than a stitched summary?" | ≥ 4.0 mean |
| Directive adherence | "Given these Story Directives (tone: {tone}, audience: {audience}, angle: {angle}), rate 1–5 how well the narrative follows them." | ≥ 3.5 mean |
| No hallucination | Adversarial: compare claims to source chunks; flag any claim with no supporting chunk | Zero ungrounded claims |

**Adversarial test — hallucination injection:**  
For 20% of eval stories, deliberately omit relevant chunks from the store before synthesis. Verify the Refined Context either correctly hedges ("evidence is limited") or stays within what the remaining chunks support. A Refined Context that invents facts when evidence is thin is a critical failure.

### 2.2 Citation Index Completeness (Rule-based)

| Criterion | Method | Pass condition |
|---|---|---|
| Every passage has a citation | Sample 10 random passages from the Refined Context; look up each in the Citation Index | ≥ 9/10 traceable |
| Citation Index references valid chunk IDs | Assert all cited Research Chunk IDs exist in pgvector | 100% |
| No orphaned citations | Assert all Citation Index entries map to a passage in the Refined Context | 100% |
| Persisted on RenderJob | Assert `citation_index` field is non-null after RETRIEVAL | 100% |

### 2.3 Context Assembly Quality (Rule-based + LLM-as-judge)

After the Context Builder produces the `AssembledContext`:

| Criterion | Method | Pass condition |
|---|---|---|
| Structure compliance | Assert `AssembledContext` contains: narrative summary, evidence sections, raw chunk payloads | All three present |
| `similarity_score` populated | Assert all retrieved chunks carry a non-null `similarity_score` | 100% |
| `topic_relevance` populated | Assert all retrieved chunks carry a valid `topic_relevance` enum value | 100% |
| Persisted on RenderJob | Assert `assembled_context` field is non-null after RETRIEVAL | 100% |
| Query composition | LLM judge: "Does this composite query (topic + Story Directives) target the right information?" | ≥ 4.0/5 |
| Evidence section relevance | LLM judge: "Are the top-N evidence sections the most pertinent to the topic, given the available chunks?" | ≥ 4.0/5 |

**Reuse invariant (regression test):** Trigger the Fact-Check Loop (≥ 1 optimizer iteration) and assert the `AssembledContext` has not been rebuilt. It must be identical to the version persisted after initial RETRIEVAL.

---

## Eval 3 — Writer's Desk

**Pipeline stage:** `SCRIPTING`

### 3.1 Master Script Quality (LLM-as-judge)

**Judge prompt (condensed):**

```
Story topic: {topic}
Story Directives: tone={tone}, audience={audience}, angle={angle}
AssembledContext (summary): {context_summary}

Evaluate the following master script on five dimensions (1–5 each):
1. Narrative coherence — does it read as a single, well-structured piece?
2. Directive fidelity — does it match the specified tone, audience, and angle?
3. Evidence grounding — are factual claims anchored to the provided context?
4. Claim density — does it avoid both thin content and over-claiming?
5. Audience fit — would the target audience find it accessible and engaging?

Return JSON: {"coherence": N, "fidelity": N, "grounding": N, "density": N, "audience_fit": N, "notes": "..."}
```

**Thresholds:** All five dimensions ≥ 3.5. Any single dimension < 3.0 is a hard failure.

### 3.2 Schema Compliance (Rule-based)

| Criterion | Check | Pass condition |
|---|---|---|
| Role set correctly | `script.role == "master"` for first draft | 100% |
| Version initialized | `script.version == 1` for initial draft | 100% |
| Script not empty | `len(script.content) > 0` | 100% |
| Directives consumed | Story Directives fields are referenced in the scripting agent's prompt | Verified via prompt template audit |

### 3.3 Format Script Derivation (LLM-as-judge, Layout Desk gate)

When a format script is derived from the master:

| Criterion | Check | Pass condition |
|---|---|---|
| Content fidelity | "Does the format script faithfully represent the master's key points?" | ≥ 4.0/5 |
| Format adaptation | "Is the format script appropriately adapted for its target format (e.g., blog)?" | ≥ 4.0/5 |
| Role set correctly | `script.role == "format"` | 100% |
| Version incremented | `script.version > master_script.version` | 100% |

---

## Eval 4 — Fact-Check Desk (Red Team)

**Pipeline stage:** `FACT_CHECKING_SCRIPT`  
This is the highest-stakes eval surface. An over-permissive Fact-Check Desk lets false claims reach users; an over-restrictive one creates infinite loops.

### 4.1 Claim Extraction Quality

**Ground truth construction:** For a held-out set of 30 scripts, have a human annotator enumerate all atomic, verifiable claims.

| Metric | Definition | Target |
|---|---|---|
| Recall | Fraction of human-labelled claims extracted by the agent | ≥ 0.90 |
| Precision | Fraction of extracted claims that are legitimate atomic claims (not meta-commentary, not duplicates) | ≥ 0.85 |
| Category accuracy | For extracted claims, fraction assigned the correct `category` enum | ≥ 0.80 |

**Edge cases to explicitly test:**
- Compound claims ("X happened in 2019 and caused Y") → should split into two atomic claims
- Implicit claims ("the leading provider" implies a ranking) → must be extracted
- Attribution claims ("according to X, Y is true") → `category=attribution`

### 4.2 Verdict Accuracy (Reference-based)

Build a **Verdict Ground Truth dataset**: 100 (claim, evidence_chunks) pairs, each human-labelled with the correct verdict.

| Metric | Target |
|---|---|
| Overall verdict accuracy | ≥ 0.82 |
| SUPPORTED precision | ≥ 0.88 (false SUPPORTED is the most dangerous error) |
| UNSUPPORTED recall | ≥ 0.80 (missing an unsupported claim lets it through) |
| UNCERTAIN vs CONTESTED accuracy | ≥ 0.70 (this distinction is genuinely hard) |

**Adversarial test — planted false claims:**  
Inject 5 known-false, verifiable claims into 20 test scripts. Measure:
- Detection rate (UNSUPPORTED or CONTESTED verdict) → target ≥ 90%
- False pass rate (SUPPORTED verdict on a planted false claim) → target 0%

### 4.3 Confidence Calibration (Statistical)

For 200 (claim, verdict, confidence) triples with known ground truth:

| Metric | Target |
|---|---|
| Expected calibration error (ECE) | ≤ 0.10 |
| High-confidence accuracy (confidence ≥ 0.85) | ≥ 0.90 |
| Low-confidence accuracy (confidence ≤ 0.50) | May be < 0.65 (high uncertainty is correct) |

### 4.4 Evidence Traceability (Rule-based)

| Criterion | Check | Pass condition |
|---|---|---|
| `evidence_references` populated | Every non-UNCERTAIN verdict has ≥ 1 chunk reference | 100% |
| Referenced chunks exist | All `evidence_references` IDs exist in pgvector | 100% |
| Citation Index used | Agent retrieves from Citation Index rather than re-querying pgvector | Verified via trace |
| UNCERTAIN handling | UNCERTAIN claims carry hedged `evidence_text` (not empty) | 100% |

---

## Eval 5 — Fact-Check Loop

**Cross-stage:** interaction between `FACT_CHECKING_SCRIPT` and `SCRIPTING`.

### 5.1 Remediation Effectiveness (Rule-based + LLM-as-judge)

After the Script Optimizer patches a script for failed claims:

| Criterion | Check | Pass condition |
|---|---|---|
| Targeted patching | Only claims with UNSUPPORTED or CONTESTED verdicts are modified | 0 unintended edits to SUPPORTED claims |
| Fix rate per cycle | Re-evaluate patched claims; measure fraction that now pass | ≥ 0.70 per cycle |
| Regression rate | Fraction of previously SUPPORTED claims that newly fail after patching | ≤ 0.05 |
| `remediation_depth` incremented | Assert counter increments by 1 per cycle | 100% |

**LLM judge for surgical precision:**

```
Original claim (failed): {claim_text}
Original sentence in script: {original_sentence}
Patched sentence in script: {patched_sentence}

1. Was only the failed claim addressed? (Yes/No)
2. Was the patch the minimum necessary change? (Yes/No)
3. Did the patch preserve the narrative flow? (1–5)
```

Target: Yes/Yes on criteria 1–2 for ≥ 90% of patches; criterion 3 ≥ 4.0.

### 5.2 Loop Termination (Rule-based)

| Scenario | Expected behavior | Test method |
|---|---|---|
| All claims SUPPORTED, strictness=Low or Medium | Auto-advance to FORMATTING | Assert status transition |
| All claims SUPPORTED, strictness=High | Route to `HUMAN_REVIEW_NEEDED` | Assert status + reason field |
| UNCERTAIN claims present | Apply hedged language, do not trigger revision | Assert no Optimizer invocation |
| `remediation_depth >= max_cycles` | Escalate to `HUMAN_REVIEW_NEEDED` | Force max_cycles failures, assert escalation |
| UNSUPPORTED claim survives all cycles | Escalate to `HUMAN_REVIEW_NEEDED` | Plant unfixable claim, assert escalation |

| CopywriterAgent returns `ESCALATE` (source contradiction) | Route to `HUMAN_REVIEW_NEEDED` in one pass, no retry | Assert status transition, assert no further SCRIPTING attempts |
| CopywriterAgent returns `ERROR` (evidence too thin) | Route to `HUMAN_REVIEW_NEEDED` (orchestrator cannot generate new evidence) | Assert status transition |
| ScriptOptimizerAgent returns `ESCALATE` (unresolved claim) | Route to `HUMAN_REVIEW_NEEDED` in one pass, no retry | Assert status transition, assert no further optimizer iterations |
| Formatter returns `ESCALATE` (correction_hint/plan conflict) | Route to `HUMAN_REVIEW_NEEDED` via harness short-circuit | Assert status transition |

**Regression test — loop ceiling:**  
Construct a script where the true answer is always UNSUPPORTED (claim contradicts all chunks). Verify the loop terminates at `remediation_depth == max_cycles` and never exceeds it.

### 5.3 Guardrail Strictness Compliance (Rule-based, per-level)

Test each strictness level independently with a controlled script.

| Level | similarity threshold | UNCERTAIN behavior | All-SUPPORTED outcome | Required `max_cycles` |
|---|---|---|---|---|
| Low | 0.65 | Passes | Auto-advance | 2 |
| Medium | 0.72 | Passes | Auto-advance | 3 |
| High | 0.75 | Soft fail → revision | Human review | 3 |

For each level: assert the agent applies the correct similarity threshold when querying evidence, handles UNCERTAIN as specified, and routes to the correct terminal state.

---

## Eval 6 — Layout Desk

**Pipeline stage:** `FORMATTING`

### 6.1 Format Output (Blog) Quality (Rule-based + LLM-as-judge)

| Criterion | Check | Pass condition |
|---|---|---|
| Sections present | Assert `format_output.sections` is non-empty | 100% |
| SEO metadata present | Assert `title`, `meta_description`, `tags` are all non-null | 100% |
| CTA present | Assert `call_to_action` is non-null and non-empty | 100% |
| Publish-readiness | LLM judge: "Is this blog post ready to publish without further editing?" | ≥ 4.0/5 |
| Master fidelity | LLM judge: "Does the blog faithfully represent the master script's key points?" | ≥ 4.0/5 |

### 6.2 Carousel Slide Deck (Rule-based + LLM-as-judge)

| Criterion | Check | Pass condition |
|---|---|---|
| Slide count | Assert deck has ≥ 3 and ≤ platform_max slides | 100% |
| `slide_number` ordinal integrity | Assert slides are numbered 1..N with no gaps | 100% |
| `text` within character limit | Assert each slide's text ≤ platform character limit | 100% |
| `visual_description` present | Assert non-null, non-empty on every slide | 100% |
| `hook_type` valid | Assert each slide's `hook_type` is a valid enum value | 100% |
| `sources_used` populated | Assert at least one chunk UUID cited per slide | ≥ 80% of slides |
| Hook diversity | Count distinct `hook_type` values across deck | ≥ 3 distinct types |
| `visual_description` quality | LLM judge: "Is this a useful, specific visual description for a designer?" | ≥ 3.5/5 |

**Do not call a Carousel a Format Output (schema test):**  
Assert that `carousel_slide_deck` objects are never stored in the `format_outputs` table and vice versa.

### 6.3 Platform Compliance (Rule-based)

For each supported platform, verify character limits are applied:

| Platform | Carousel slide char limit | Check |
|---|---|---|
| Instagram | Per platform spec | Assert `text` ≤ limit for `platform=INSTAGRAM` stories |
| TikTok | Per platform spec | Assert `text` ≤ limit for `platform=TIKTOK` stories |
| Twitter/X | Per platform spec | Assert `text` ≤ limit for `platform=TWITTER` stories |
| LinkedIn | Per platform spec | Assert `text` ≤ limit for `platform=LINKEDIN` stories |

---

## Eval 7 — Pipeline Status Transitions

**Cross-stage:** correctness of the state machine.

### 7.1 Happy Path (Rule-based)

For each successful story, assert the status sequence is a valid prefix or completion of:

```
PENDING → RESEARCHING → RETRIEVAL → SCRIPTING → FACT_CHECKING_SCRIPT → FORMATTING → ASSET_GENERATION → COMPLETED
```

No valid story should skip a non-terminal stage.

### 7.2 Terminal States (Rule-based)

| Scenario | Expected terminal status | Test |
|---|---|---|
| All desks succeed | `COMPLETED` | Happy path run |
| Unrecoverable error at any desk | `FAILED` | Inject fault at each desk |
| Fact-Check Loop exhausted | `HUMAN_REVIEW_NEEDED` | Force max_cycles failures |
| High strictness + all-SUPPORTED | `HUMAN_REVIEW_NEEDED` | Set strictness=High, clean script |
| CopywriterAgent `ESCALATE` (source contradiction) | `HUMAN_REVIEW_NEEDED` — no retry | Mock agent returns `ESCALATE` at SCRIPTING stage |
| CopywriterAgent `ERROR` (evidence too thin) | `HUMAN_REVIEW_NEEDED` — no retry | Mock agent returns `ERROR` at SCRIPTING stage |
| ScriptOptimizerAgent `ESCALATE` (unresolved claim) | `HUMAN_REVIEW_NEEDED` — no retry | Mock agent returns `ESCALATE` at SCRIPTING stage (optimizer path) |
| Formatter `ESCALATE` (correction_hint/plan conflict) | `HUMAN_REVIEW_NEEDED` — harness short-circuit | Mock harness returns `escalated=True` |
| Human resolves review | Resumes from correct next desk | Simulate human approval |

### 7.3 Status Idempotency (Rule-based)

Assert that re-processing a story from any non-terminal status produces the same terminal result (within acceptable stochastic variance for LLM components). No story should flip between terminal states on re-run.

---

## Eval 8 — End-to-End Integration

### 8.1 Story lifecycle (Smoke test)

Run a fixed set of 10 canonical topics (diverse domains, edge-case inputs) through the full pipeline. Assert:
- All stories reach a terminal state
- No stories stall indefinitely (timeout: 10 min per story)
- `COMPLETED` rate on clean inputs ≥ 80%

### 8.2 Directive Propagation (Rule-based)

Assert that Story Directives are present in the prompts of:
- Retrieval Desk (Synthesis step)
- Writer's Desk (Scripting step)
- Layout Desk (Formatting step)

Assert that Story Directives are **absent** from the Research Desk (Indexing) and Fact-Check Desk prompts (they should have no editorial awareness).

### 8.3 Scope Lifecycle (Rule-based)

After a story reaches `COMPLETED`:
- Assert all `scope=LOCAL` chunks for that story have been cleaned up (deleted or marked inactive)
- Assert `scope=RAW-CONTEXT` chunks from user-provided material are retained if specified by the Story's retention policy

### 8.4 Latency Budget (Monitoring)

| Stage | P50 target | P95 target |
|---|---|---|
| Research Desk | < 15s | < 45s |
| Retrieval Desk | < 20s | < 60s |
| Writer's Desk | < 30s | < 90s |
| Fact-Check Desk (per cycle) | < 20s | < 60s |
| Layout Desk | < 15s | < 45s |
| Full pipeline (no Visual Assets) | < 3 min | < 8 min |

---

## Eval Dataset Recommendations

| Dataset | Size | Construction method | Used by evals |
|---|---|---|---|
| Canonical topics | 10 stories | Hand-curated, diverse domains | 8.1 smoke test |
| Research chunk labels | 50 chunks | Human-labelled `topic_relevance`, threshold-probing at 0.75/0.50 boundaries | 1.3 calibration |
| Verdict ground truth | 100 (claim, chunks) pairs | Human-labelled verdicts | 4.2 verdict accuracy |
| Adversarial scripts | 20 scripts × 5 planted false claims | Constructed synthetic | 4.2 adversarial |
| Compound claim scripts | 15 scripts | Constructed synthetic | 4.1 edge cases |
| Loop exhaustion scripts | 10 scripts with unfixable claims | Constructed synthetic | 5.2 loop termination |
| Directive adherence matrix | 3 tones × 3 audiences × 3 angles | Combinatorial | 3.1, 2.1 |

---

## Failure Mode Taxonomy

| Code | Name | Stage | Severity | Description |
|---|---|---|---|---|
| F-R1 | Sparse research | Research | Medium | < 5 chunks ingested; synthesis will be thin |
| F-R2 | Duplicate chunks | Research | Low | Inflates apparent evidence, hurts diversity |
| F-V1 | Hallucinated synthesis | Retrieval | Critical | Refined Context contains claims not in any chunk |
| F-V2 | Broken Citation Index | Retrieval | High | Claims untraceable; Fact-Check Desk blind |
| F-W1 | Directive drift | Writer | Medium | Script ignores tone/audience/angle directives |
| F-F1 | Missed false claim | Fact-Check | Critical | Planted false claim receives SUPPORTED verdict |
| F-F2 | Over-rejection | Fact-Check | Medium | True claims consistently receive UNSUPPORTED |
| F-L1 | Infinite loop | Fact-Check Loop | Critical | `remediation_depth` exceeds `max_cycles` without escalation |
| F-L2 | Collateral edits | Fact-Check Loop | Medium | Optimizer modifies SUPPORTED claims |
| F-L3 | Wrong strictness | Fact-Check Loop | High | Guardrail level thresholds not applied correctly |
| F-O1 | Carousel as Format Output | Layout | Low | Type confusion; schema integrity violation |
| F-O2 | Character limit breach | Layout | High | Slide text exceeds platform limit |
| F-P1 | Status skip | Pipeline | High | Story skips a required non-terminal stage |
| F-P2 | Wrong terminal state | Pipeline | Critical | Story reaches wrong terminal (e.g., COMPLETED when should be HUMAN_REVIEW_NEEDED) |