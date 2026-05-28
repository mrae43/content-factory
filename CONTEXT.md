# Content Factory — Domain Glossary

## Image Storage

Generated carousel slide images are stored in SeaweedFS via its S3-compatible API, then served directly to the browser as plain HTTP URLs. The pipeline uses two identities in SeaweedFS's S3 config: an `anonymous` identity with `Read` access (so browsers can fetch images without authentication) and a `factory` identity with full `Admin/Read/Write` credentials used by the API container for uploads. S3 public URL format: `http://localhost:8333/{bucket}/{device_id}/{job_id}/slide_{NN}.png`.



## Story

The central unit of work flowing through the pipeline. Represented in the database as a `RenderJob`. A Story has a title, a status, and carries all content produced through the pipeline (research, scripts, claims, assets).

## Research

The first value-adding phase of the pipeline. Performed by the Research Desk. Web search only: TavilySearch on the Story's title. Results are chunked, embedded (Gemini 768-dim, L2-normalized — see ADR 0003), and ingested into the vector store as `LOCAL`-scope chunks with `source_type: WEB_SEARCH`. Produces no narrative output — synthesis is handled by the Retrieval Desk.

_Ancillary term:_ **Research Chunks** — the individual vectorized text fragments stored in pgvector, tagged with a **source_type** enum (`USER_PROVIDED` from user-supplied URLs/raw text, `WEB_SEARCH` from Tavily results, `INFERRED` from the Retrieval Desk's synthesis), and a **scope** (`RAW-CONTEXT` for user-provided material, `LOCAL` for web results and refined chunks). source_type controls epistemic weight during fact-checking; scope controls lifecycle (LOCAL chunks are cleaned up after completion).

Each chunk carries enrichment metadata: **similarity_score** (cosine distance from the retrieval query), **topic_relevance** (categorical: `HIGH | MEDIUM | LOW`, derived from the score), and optionally **source_authority** (a signal from domain reputation — deferred).

## Retrieval

The second phase, performed by the Retrieval Desk. Owns all pre-scripting evidence assembly. Two sequential steps:

1. **Context Assembly** — the Context Builder performs a semantic search against all chunk scopes (composite query from title + Story Directives), enriches retrieved chunks with `similarity_score` and `topic_relevance`, and produces an `AssembledContext` (narrative summary + formatted evidence sections + raw chunk payloads). The retrieved evidence is also used to produce a **Refined Context** (800–1500 word narrative).
2. **Persistence** — the `AssembledContext` and `refined_context` are persisted on the Story's `RenderJob` row for consumption by the Writer's Desk.

## Context Builder

A component that runs inside the RETRIEVAL phase. It performs a semantic search against all chunk scopes (composite query from title + Story Directives), enriches retrieved chunks with `similarity_score` and `topic_relevance`, and produces the `AssembledContext` that the CopywriterAgent receives (narrative summary + evidence sections + raw chunk payloads). The result is persisted on the Story's `RenderJob` row and reused across the Fact-Check Loop — it is not rebuilt on each optimizer iteration.

## Research Inputs

The retrievable material fed into the Indexer. Consists of two separate fields: **source_urls** (user-supplied URLs for web search and extraction) and **user_reference** (raw text providing editorial context). Consumed during Indexing and Retrieval — they do not travel downstream to the Writer's Desk directly.

_Avoid:_ Mixing Story Directives into Research Inputs. Directives shape synthesis; Inputs feed retrieval.

## User Reference

The editorial brief raw text provided by the user at commission time. Stored as its own column (`user_reference`) on the RenderJob row, separate from `source_urls`. Passed to the Context Builder during Retrieval to compose the composite query, and propagated into the Refined Context as background context. Not to be confused with the user-supplied raw text that was previously embedded in `pre_context`.

## Story Directives

The editorial metadata that shapes *how* research and scripting are framed. Includes target audience, tone, angle, and guardrail strictness. Stored as its own column on the RenderJob row. Unlike Research Inputs, Directives travel all the way to the Layout Desk (FORMATTING) — they are consumed by Synthesis and Scripting, not by Indexing.

## Script

The narrative body of a Story, written by the Writer's Desk. A script has a role (`master` for the general narrative, `format` for a format-specific adaptation), a version number, and optionally associated Claims from fact-checking. A master script is drafted first; format scripts are derived from it during the Layout Desk stage.

## Format Output

A text-only rendering of a Story in blog format. Produced by the Layout Desk from the master script. Contains sections, SEO metadata, tags, and a call to action. Format Outputs are publish-ready without further rendering — unlike Visual Assets, which require the Production Studio.

## Visual Asset

A media artifact that must be rendered from a blueprint. Two sub-types, both produced by a two-stage pipeline (Layout Desk plans the structure, Production Studio renders the media):

- **Carousel Slide Deck** — a sequence of composited slides (HTML layout + SVG graphics), targeted at Instagram/TikTok. The Layout Desk produces a slide outline with per-slide text and visual prompts; the Production Studio renders each slide into browser-displayable HTML (frontend-rendered from structured data — no image export in MVP).
- **Video** — a rendered motion picture with scenes, narration, and audio. The Layout Desk produces a scene outline; the Production Studio renders each scene and assembles the final video.

_Avoid_: Calling a Carousel a "Format Output" — it's a Visual Asset that gets rendered, not formatted.

## Visual Generation

The role of the AssetStudioAgent: translate video scenes and visual style into technical visual/audio prompts for production-grade AI models (Veo for video, Lyria for audio). Visual Generation explicitly does NOT generate text or typography — text captions are a separate concern handled by the text generation layer (CopywriterAgent, formatters). This is a hard domain separation boundary: the visual model renders the image; the text agent renders the caption. Rule 2 in AssetStudioAgent's prompt enforces this: "Do NOT include text or typography in visual prompts."

_Avoid_: Asking the Asset Studio to embed text into generated images. That is an integration failure, not a feature gap.

## Underspecified Scene

Input that lacks sufficient detail to produce a coherent visual/audio prompt within the 2-sentence hard limit. When a scene cannot be specified in 2 sentences of prompt, that is a signal the scene input is underspecified — the agent must return status=ERROR describing what is missing, not expand the prompt. This is enforced by AssetStudioAgent Rule 3.

## Red Team Report

The output of a fact-checking pass over a Script. Produced by the Fact-Check Desk (Red Team agent). Contains a list of Claim Verdicts plus an overall reasoning summary. Each claim is extracted, cross-referenced against evidence chunks, and assigned a verdict.

## Claim Verdict

The evaluation of a single atomic claim extracted from a script. Contains:
- **claim_text** — the exact statement under evaluation
- **verdict** — one of `SUPPORTED`, `UNSUPPORTED`, `CONTESTED`, `UNCERTAIN`
- **confidence** — 0.0–1.0 score
- **evidence_text** — human-readable evidence summary
- **evidence_references** — links to the Research Chunks that informed the verdict
- **category** — `statistic`, `attribution`, `chronological`, `causal`, `comparative`

## Evidence Grounding

The constraint that every claim verdict from the RedTeamAgent must trace to specific evidence chunks retrieved from the vector store. If no evidence is available for a claim, the verdict must be assigned as UNCERTAIN — the agent must not fabricate a SUPPORTED or REFUTED verdict to fill the gap. Enforced by the ## RULES section in `EVALUATION_SYSTEM` (agents.py). Also prohibits misrepresenting evidence to fit a preferred verdict: if the evidence genuinely conflicts, the correct outcome is UNCERTAIN with `conflicting_evidence=true`, not a forced SUPPORTED or REFUTED.

## Guardrail Strictness

The single knob that controls fact-checking rigor. Three levels, set per-Story:

| Level | Similarity | Claim Categories | Revisions | UNCERTAIN | All-SUPPORTED advances? |
|---|---|---|---|---|---|
| Low | 0.65 | statistic, attribution | 2 | Passes | Yes, auto |
| Medium | 0.72 | + chronological, causal | 3 | Passes | Yes, auto |
| High | 0.75 | + comparative | 3 | **Soft fail** | **No → Your Review** |

Formerly split across `guardrail_strictness` (the level) and `strict_compliance_mode` (the boolean), now collapsed into a single choice. The High profile absorbs the strict-compliance behavior: UNCERTAIN is treated as a soft fail (triggers revision), and even all-SUPPORTED scripts go to human review.

## Fact-Check Loop

The iterative cycle between the Fact-Check Desk and Writer's Desk when a script contains unsupported or contested claims. The Script Optimizer surgically patches the failed claims and returns the script for re-evaluation, up to a maximum of 3 cycles (the `remediation_depth`).

When `remediation_depth >= max_cycles`, the loop escalates: the Story moves to `Your Review` for human judgment instead of retrying.

In code, the bounded counter is named `remediation_depth` (0–3). In editorial/outward language, this pattern is called the Fact-Check Loop.

## Verdict

| Value | Meaning | UI Color | Behavior |
|---|---|---|---|
| SUPPORTED | Evidence confirms the claim | Green | Passes; no action needed |
| UNSUPPORTED | Evidence contradicts the claim | Red | Triggers Optimizer revision |
| CONTESTED | Evidence conflicts or is inconclusive | Amber | Triggers Optimizer revision |
| UNCERTAIN | Insufficient evidence to evaluate | Blue | Formatter applies hedged language |

## Asset

A generated media file associated with a Story. Produced by the Production Studio. Each asset carries generation metadata (prompt, timing, SynthID watermark). Types:

| Asset Type | Applies To | Description |
|---|---|---|
| `CAROUSEL_SLIDE` | Carousel | A single rendered slide image |
| `VISUAL_VEO` | Video | A rendered video clip |
| `AUDIO_LYRIA` | Video | Generated music or sound effect |
| `VOICEOVER` | Video | AI-generated spoken narration |
| `SUBTITLE_JSON` | Video | Timed subtitle track |
| `DATA_CHART` | Video | Animated data visualization

## Pipeline Statuses

Each Story passes through editorial desks in sequence. The canonical outward-facing names (used in UI, user communication) and their backing enum values:

| Desk (Outward) | Enum Value | Terminal? |
|---|---|---|---|
| Queued | `PENDING` | No |
| Research Desk | `RESEARCHING` | No |
| Retrieval Desk | `RETRIEVAL` | No |
| Writer's Desk | `SCRIPTING` | No |
| Fact-Check Desk | `FACT_CHECKING_SCRIPT` | No |
| Layout Desk | `FORMATTING` | No |
| Production Studio | `ASSET_GENERATION` | No | Renders Visual Assets (Carousel Slide Decks and Videos) |
| Published | `COMPLETED` | Yes |
| Killed | `FAILED` | Yes |
| Your Review | `HUMAN_REVIEW_NEEDED` | Yes (blocked on human)

## Platform

The target social media platform for a Story. Determines which formats are available and
per-character limits for carousel content. Supported platforms: Twitter/X, LinkedIn,
Instagram, TikTok, YouTube. Platform is required at creation time (platform-first policy).

## Carousel Slide

A single slide in a Carousel Slide Deck. Contains:
- **slide_number** — ordinal position in the deck
- **text** — the slide's body content
- **visual_description** — a text caption describing the visual element (icon/graphic) on the slide; displayed as a subdued subtitle. Replaces the earlier `visual_prompt` concept (which described a generated image — no longer applicable).
- **hook_type** — the rhetorical hook category (question, statistic, quote, visual, story, cta)
- **sources_used** — research chunk UUIDs cited on this slide

_Avoid_: Calling it a "Visual Asset" — the Carousel Slide Deck is the Asset; a single slide is a component of it.

## Working Memory

Ephemeral cross-stage state carried in the `working_memory` JSONB column on the
`RenderJob` row. Replaces the previous absence of agent-accessible memory between
pipeline transitions. Never explicitly cleared — retained on the job row as audit
trail. Three sub-sections: `copywriter_rationale`, `optimizer_phase`, and
`epistemic_ledger`. The orchestrator owns all reads and writes; agents remain
stateless and receive relevant sections via the context dict.

## Copywriter Rationale

Sub-section of Working Memory (`working_memory.copywriter_rationale`). A lightweight
metadata block produced by the CopywriterAgent alongside the script. Contains
`narrative_intent` (one-sentence strategy summary) and `claim_disambiguations`
(a list mapping script excerpts to their category — factual, stylistic, rhetorical,
interpretive — with intent and optional source reference). NOT a substitute for
the Red Team's claim extraction. Consumed by the Red Team to distinguish
intentional framing from factual assertions, reducing false-positive flags.

_LLM drift mitigation:_ Copywriter is instructed to match `script_excerpt` to
`script_content` word-for-word; Red Team falls back to fuzzy/semantic matching
on mismatch.

## Optimizer Phase

Sub-section of Working Memory (`working_memory.optimizer_phase`). Written after
each optimizer iteration. Contains the optimizer's `patch_summary` (free-text)
and `resolved_claims` (a list of `{claim_uuid, patch_intent, is_completely_resolved}`
entries). The `claim_uuid` is the ADR 0006 identity anchor — the orchestrator
translates the optimizer's text-keyed `original_claim_text` to the ledger UUID
after the optimizer returns, keeping raw UUIDs out of LLM prompts.

## Epistemic Ledger

Sub-section of Working Memory (`working_memory.epistemic_ledger`). Written after
each Red Team pass. Contains `weak_passes` — claims that the Red Team marked as
UNCERTAIN or CONTESTED, or that passed with low confidence (< 0.75). Each entry
includes the claim text, verdict, confidence, and a reason for the weakness.
Consumed by the Formatters alongside the flat `hedge_index` to produce
appropriately hedged language in the formatted output.

## Model Tier

The capability level of the LLM assigned to a pipeline agent. Two tiers (see ADR 0003 for embedding model — embedding is independent of agent LLM tier):
- **Premium**: agents requiring deep reasoning, synthesis, or precise fact-checking (CopywriterAgent, RedTeamAgent).
- **Standard**: agents performing constrained structured-output tasks or prompt enrichment (ScriptOptimizerAgent, AssetStudioAgent, BlogFormatterAgent, CarouselFormatterAgent, VideoFormatterAgent).

The specific model names are **env-var-driven defaults** (see `app/core/config.py`). The architecture is the tier assignment, not the particular model name. Swap models via env vars without code changes.

## Tool

A formal object wrapping a service capability with name, description, input/output schemas, workflow scope, and permissions. Tools serve two roles: **dependency injection** (DI tools, injected into agents at runtime by the harness) and **LLM function-calling** (LLM tools, registered with the model for dynamic invocation during execution). Defined alongside the backing service, registered in a central `ToolRegistry`. Agents declare tool dependencies explicitly via `di_tools` and `llm_tools` class attributes. Tool access is governed by symmetric permissions: both the tool and the agent must agree. See ADR 0004.

