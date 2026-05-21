# Eval Contracts — Index

Each contract file below derives its criteria from the high-level "MUST"
assertions in [`../eval-contracts.md`](../eval-contracts.md). Every eval
stage maps to a section in [`../evals-criteria.md`](../evals-criteria.md).

---

## Eval 1 — Research Desk (`RESEARCHING`)

| File | Sub-eval | Criteria ref |
|---|---|---|
| [`01-research-coverage.md`](01-research-coverage.md) | 1.1 Coverage | Contract 6 |
| [`01-research-chunk-quality.md`](01-research-chunk-quality.md) | 1.2 Chunk Quality | — |
| [`01-research-topic-relevance-calibration.md`](01-research-topic-relevance-calibration.md) | 1.3 Calibration | Contract 4 |

## Eval 2 — Retrieval Desk (`RETRIEVAL`)

| File | Sub-eval | Criteria ref |
|---|---|---|
| [`02-retrieval-synthesis-quality.md`](02-retrieval-synthesis-quality.md) | 2.1 Synthesis — Refined Context | — |
| [`02-retrieval-citation-index.md`](02-retrieval-citation-index.md) | 2.2 Citation Index | — |
| [`02-retrieval-context-assembly.md`](02-retrieval-context-assembly.md) | 2.3 Context Assembly | — |

## Eval 3 — Writer's Desk (`SCRIPTING`)

| File | Sub-eval | Criteria ref |
|---|---|---|
| [`03-writer-master-script-quality.md`](03-writer-master-script-quality.md) | 3.1 Master Script Quality | — |
| [`03-writer-schema-compliance.md`](03-writer-schema-compliance.md) | 3.2 Schema Compliance | — |
| [`03-writer-format-derivation.md`](03-writer-format-derivation.md) | 3.3 Format Script Derivation | — |

## Eval 4 — Fact-Check Desk (`FACT_CHECKING_SCRIPT`)

| File | Sub-eval | Criteria ref |
|---|---|---|
| [`04-factcheck-claim-extraction.md`](04-factcheck-claim-extraction.md) | 4.1 Claim Extraction | — |
| [`04-factcheck-verdict-accuracy.md`](04-factcheck-verdict-accuracy.md) | 4.2 Verdict Accuracy | — |
| [`04-factcheck-confidence-calibration.md`](04-factcheck-confidence-calibration.md) | 4.3 Confidence Calibration | — |
| [`04-factcheck-evidence-traceability.md`](04-factcheck-evidence-traceability.md) | 4.4 Evidence Traceability | — |

## Eval 5 — Fact-Check Loop

| File | Sub-eval | Criteria ref |
|---|---|---|
| [`05-loop-remediation-effectiveness.md`](05-loop-remediation-effectiveness.md) | 5.1 Remediation Effectiveness | — |
| [`05-loop-termination.md`](05-loop-termination.md) | 5.2 Loop Termination | — |
| [`05-loop-guardrail-strictness.md`](05-loop-guardrail-strictness.md) | 5.3 Guardrail Strictness | — |

## Eval 6 — Layout Desk (`FORMATTING`)

| File | Sub-eval | Criteria ref |
|---|---|---|
| [`06-layout-blog-quality.md`](06-layout-blog-quality.md) | 6.1 Blog Quality | Contract 2 |
| [`06-layout-carousel-slide-deck.md`](06-layout-carousel-slide-deck.md) | 6.2 Carousel Slide Deck | Contract 2 |
| [`06-layout-platform-compliance.md`](06-layout-platform-compliance.md) | 6.3 Platform Compliance | Contract 5 |

## Eval 7 — Pipeline Status Transitions

| File | Sub-eval | Criteria ref |
|---|---|---|
| [`07-pipeline-happy-path.md`](07-pipeline-happy-path.md) | 7.1 Happy Path | Contracts 1, 9 |
| [`07-pipeline-terminal-states.md`](07-pipeline-terminal-states.md) | 7.2 Terminal States | Contract 1 |
| [`07-pipeline-status-idempotency.md`](07-pipeline-status-idempotency.md) | 7.3 Status Idempotency | — |

## Eval 8 — End-to-End Integration

| File | Sub-eval | Criteria ref |
|---|---|---|
| [`08-e2e-story-lifecycle.md`](08-e2e-story-lifecycle.md) | 8.1 Story Lifecycle | — |
| [`08-e2e-directive-propagation.md`](08-e2e-directive-propagation.md) | 8.2 Directive Propagation | — |
| [`08-e2e-scope-lifecycle.md`](08-e2e-scope-lifecycle.md) | 8.3 Scope Lifecycle | — |
| [`08-e2e-latency-budget.md`](08-e2e-latency-budget.md) | 8.4 Latency Budget | — |

---

## Cross-cutting contracts (not covered by existing evals)

The following contracts in [`../eval-contracts.md`](../eval-contracts.md) address
gaps that lack a corresponding sub-eval in `evals-criteria.md`:

| Contract | What it guards | Covered by |
|---|---|---|
| [Contract 1](../eval-contracts.md#contract-1--jobstatus-enum-completeness) | `JobStatus` enum matches production | `07-pipeline-happy-path.md`, `07-pipeline-terminal-states.md` |
| [Contract 3](../eval-contracts.md#contract-3--casecategory-enum-cross-check) | `CaseCategory` enum consistency | — |
| [Contract 7](../eval-contracts.md#contract-7--production-studio--visual-asset-type-coverage) | `ASSET_GENERATION` stage coverage | — |
| [Contract 8](../eval-contracts.md#contract-8--fact_checking_research-glossary-documentation) | Glossary documents `FACT_CHECKING_RESEARCH` | — |
