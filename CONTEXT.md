# Content Factory — Domain Glossary

## Image Storage

Generated carousel slide images are stored in SeaweedFS via its S3-compatible API, then served directly to the browser as plain HTTP URLs. The pipeline uses two identities in SeaweedFS's S3 config: an `anonymous` identity with `Read` access (so browsers can fetch images without authentication) and a `factory` identity with full `Admin/Read/Write` credentials used by the API container for uploads. S3 public URL format: `http://localhost:8333/{bucket}/{device_id}/{job_id}/slide_{NN}.png`.



## Story

The central unit of work flowing through the pipeline. Represented in the database as a `RenderJob`. A Story has a topic (headline), a status, and carries all content produced through the pipeline (research, scripts, claims, assets).

## Research

The first value-adding phase of the pipeline. Performed by the Research Desk. Web search only: TavilySearch on the Story's topic. Results are chunked, embedded (Gemini 768-dim, cosine), and ingested into the vector store as `LOCAL`-scope chunks with `source_type: WEB_SEARCH`. Produces no narrative output — synthesis is handled by the Retrieval Desk.

_Ancillary term:_ **Research Chunks** — the individual vectorized text fragments stored in pgvector, tagged with a **source_type** enum (`USER_PROVIDED` from user-supplied URLs/raw text, `WEB_SEARCH` from Tavily results, `INFERRED` from the Retrieval Desk's synthesis), and a **scope** (`RAW-CONTEXT` for user-provided material, `LOCAL` for web results and refined chunks). source_type controls epistemic weight during fact-checking; scope controls lifecycle (LOCAL chunks are cleaned up after completion).

Each chunk carries enrichment metadata: **similarity_score** (cosine distance from the retrieval query), **topic_relevance** (categorical: `HIGH | MEDIUM | LOW`, derived from the score), and optionally **source_authority** (a signal from domain reputation — deferred).

## Retrieval

The second phase, performed by the Retrieval Desk. Owns all pre-scripting evidence assembly. Three sequential steps:

1. **Synthesis** — the ResearchAgent queries the vector store (semantic search over all `RAW-CONTEXT` and `LOCAL` scopes) and feeds retrieved chunks to an LLM to produce a **Refined Context** (800–1500 word narrative), a **Citation Index**, and refined `INFERRED` chunks that are re-ingested into the vector store.
2. **Context Assembly** — the Context Builder performs a fresh semantic search against all scopes, enriches retrieved chunks with `similarity_score` and `topic_relevance`, and produces an `AssembledContext` (narrative summary + formatted evidence sections + raw chunk payloads).
3. **Persistence** — the `AssembledContext` and `Citation Index` are persisted on the Story's `RenderJob` row for consumption by the Writer's Desk.

## Context Builder

A component that runs inside the RETRIEVAL phase. It performs a fresh semantic search against all chunk scopes (composite query from topic + Story Directives), enriches retrieved chunks with `similarity_score` and `topic_relevance`, and produces the `AssembledContext` that the CopywriterAgent receives (narrative summary + evidence sections + raw chunk payloads). The result is persisted on the Story's `RenderJob` row and reused across the Fact-Check Loop — it is not rebuilt on each optimizer iteration.

## Citation Index

A structured sidecar attached to the Story alongside the Refined Context after the Retrieval Desk's Synthesis step. Maps claim fragments (or synthesis passages) to their source URLs and Research Chunk IDs. Enables the Fact-Check Loop to trace any claim to its origin — without re-searching the vector store. Not embedded in the Refined Context text itself.

## Research Inputs

The retrievable material fed into the Indexer. Includes user-provided URLs, raw text, and web search results. Consumed during Indexing and discarded after Synthesis — they do not travel downstream.

_Avoid:_ Mixing Story Directives into Research Inputs. Directives shape synthesis; Inputs feed retrieval.

## Story Directives

The editorial metadata that shapes *how* research and scripting are framed. Includes target audience, tone, angle, and guardrail strictness. Unlike Research Inputs, Directives travel all the way to the Writer's Desk — they are consumed by Synthesis and Scripting, not by Indexing.

## Script

The narrative body of a Story, written by the Writer's Desk. A script has a role (`master` for the general narrative, `format` for a format-specific adaptation), a version number, and optionally associated Claims from fact-checking. A master script is drafted first; format scripts are derived from it during the Layout Desk stage.

## Format Output

A text-only rendering of a Story in blog format. Produced by the Layout Desk from the master script. Contains sections, SEO metadata, tags, and a call to action. Format Outputs are publish-ready without further rendering — unlike Visual Assets, which require the Production Studio.

## Visual Asset

A media artifact that must be rendered from a blueprint. Two sub-types, both produced by a two-stage pipeline (Layout Desk plans the structure, Production Studio renders the media):

- **Carousel Slide Deck** — a sequence of composited slides (HTML layout + SVG graphics), targeted at Instagram/TikTok. The Layout Desk produces a slide outline with per-slide text and visual prompts; the Production Studio renders each slide into browser-displayable HTML (frontend-rendered from structured data — no image export in MVP).
- **Video** — a rendered motion picture with scenes, narration, and audio. The Layout Desk produces a scene outline; the Production Studio renders each scene and assembles the final video.

_Avoid_: Calling a Carousel a "Format Output" — it's a Visual Asset that gets rendered, not formatted.

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

