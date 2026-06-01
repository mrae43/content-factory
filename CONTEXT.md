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

- **Carousel Slide Deck** — a sequence of composited slides with per-slide text and visual descriptions. The Layout Desk produces the carousel structure; the Production Studio renders each slide image via FLUX and uploads to S3.
- **Video** — a rendered motion picture with scenes and audio direction. The Layout Desk produces a structured scene outline plus a `unified_visual_prompt`; the Production Studio calls a model-agnostic video generation API (Together AI for testing, Seedance V2 for production) and stores the result in S3.

_Avoid_: Calling a Carousel a "Format Output" — it's a Visual Asset that gets rendered, not formatted.

## Visual Generation

The role of the Video Generator Agent: accept a structured video scene payload (produced by the Layout Desk's VideoFormatterAgent) and produce a playable video file by calling a model-agnostic video generation API (Together AI for testing, Seedance V2 for production). The agent orchestrates three deterministic steps: submit generation job, poll for completion, upload result to S3. Visual Generation explicitly does NOT generate text or typography — text captions are a separate concern handled by the text generation layer (CopywriterAgent, formatters). This is a hard domain separation boundary: the visual model renders the image; the text agent renders the caption.

_Avoid_: Asking the Video Generator Agent to embed text into generated videos. That is an integration failure, not a feature gap.

## Unified Visual Prompt

A single 1-3 sentence prompt (produced by the VideoFormatterAgent) that synthesises all video scenes into one API-ready description. Used as the input for single-shot video generation APIs (Together AI during testing). The `unified_visual_prompt` includes platform aspect ratio and narrative arc coverage. Per-scene `visual_prompt` fields are retained alongside it for future per-scene rendering (Seedance V2).

## Video Generator Agent

A `ServiceAgent` (no LLM) that orchestrates the video rendering pipeline. Injected DI tools: `generate_video`, `poll_video`, `upload_video`. Replaces the earlier `AssetStudioAgent`'s video track (which only produced prompts and a placeholder URL). Always runs wrapped in `AgentHarness`.

## Video Generation Provider

A provider-agnostic abstraction (`VideoGenProvider` ABC in `app/services/video_gen.py`) that mirrors the LLM provider registry pattern. Two implementations: `TogetherVideoGen` (testing, single-shot) and planned `SeedanceVideoGen` (production, per-scene). Selected via `video_gen_provider` env var.

## Audio Direction

Metadata describing the intended background music and sound effects for a video. Produced by the `VideoFormatterAgent` as a per-scene `audio_cue` and an overall `audio_direction` field. For v1, this is describe-only — no audio file is generated. A future audio generation agent could consume this metadata to call a music generation API.

## Underspecified Scene

Input that lacks sufficient detail to produce a coherent visual description. When a scene's `visual_prompt` or `narration_text` is generic, placeholder-level text, the formatter's stop conditions trigger status=ERROR describing what is missing. The video generation provider also enforces input validation: if the combined visual prompt is incoherent or contradictory, the agent escalates rather than hallucinating content.

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

A generated media file associated with a Story. Produced by the Production Studio via `ServiceAgent`-based generators (`CarouselImageAgent`, `VideoGeneratorAgent`). Each asset carries generation metadata (prompt, timing, SynthID watermark). Types:

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
- **Standard**: agents performing constrained structured-output tasks or prompt enrichment (ScriptOptimizerAgent, BlogFormatterAgent, CarouselFormatterAgent, VideoFormatterAgent).

The specific model names are **env-var-driven defaults** (see `app/core/config.py`). The architecture is the tier assignment, not the particular model name. Swap models via env vars without code changes.

## Tool

A formal object wrapping a service capability with name, description, input/output schemas, workflow scope, and permissions. Tools serve two roles: **dependency injection** (DI tools, injected into agents at runtime by the harness) and **LLM function-calling** (LLM tools, registered with the model for dynamic invocation during execution). Defined alongside the backing service, registered in a central `ToolRegistry`. Agents declare tool dependencies explicitly via `di_tools` and `llm_tools` class attributes. Tool access is governed by symmetric permissions: both the tool and the agent must agree. See ADR 0004.

## Discord Bot

A standalone process (`python -m app.discord_bot`) that manages the Script Content Pipeline for Discord-originated jobs. Runs inline in its own process — no QueueWorker involvement. Shares the `app/` package with the API via standard Python imports (no code duplication). Monorepo structure: two processes, one codebase.

## ScriptJob

A job that produces a completed script with verified claims, but does not produce formatted outputs or assets. Created by the Discord Bot in response to a user's `/script` command. Stored in the `factory.script_jobs` table with its own status enum (`script_job_status`). Managed exclusively by the Discord Bot — the QueueWorker never touches it. Produces a Script Content, a set of Claim Verdicts (denormalized as JSONB), and associated Working Memory. Does NOT produce platform-specific formatting or visual assets — those are handled by separate FormatJobs.

_Avoid:_ Calling it a "RenderJob" — ScriptJobs are a narrower concern with a different lifecycle. RenderJobs continue to exist for the API pipeline.

## FormatJob

A job that renders a completed Script into a platform-specific format. Created by the Discord Bot when a user clicks a format button (e.g. "Create Carousel") after a Script completes. Stored in the `factory.format_jobs` table with its own status enum (`format_job_status`). Managed by the QueueWorker via the same `FOR UPDATE SKIP LOCKED` pattern as RenderJobs. Always references a completed ScriptJob via `source_job_id`. Unique per `(source_job_id, platform, format_type)` — duplicate requests return the existing job.

## Script Content Pipeline

The text-only subset of the Content Factory pipeline: Research → Retrieval → Scripting → Fact-Check → Optimizer Loop. Produces a script and claims, no formatting or assets. Run inline by the Discord Bot process — not via the QueueWorker. A separate, decoupled concern from the Format & Asset Pipeline, which is managed by the QueueWorker. The split happens at the ADR 0010 boundary: text work is real-time and bot-managed; rendering work is queued and QueueWorker-managed.

## Snapshot Context Handoff

When a FormatJob is created from a completed ScriptJob, the necessary context (script content, claims, refined context, story directives, hedge index, epistemic ledger) is **copied** into the FormatJob row at creation time, not referenced via FK. This ensures FormatJobs are fully self-contained — they survive garbage collection of the source ScriptJob, and a failure in one FormatJob cannot corrupt shared data. The `source_job_id` FK is retained for traceability only, not for runtime data access.

_Avoid:_ Treating `source_job_id` as a live data source at format-runtime. All data needed for formatting lives in the FormatJob's snapshot columns.

