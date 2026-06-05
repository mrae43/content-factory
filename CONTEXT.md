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

The role of the Video Generator Agent: accept a structured video scene payload (produced by the Layout Desk's VideoFormatterAgent) and produce a playable video file by calling a model-agnostic video generation API (Kling AI for SHORT pipeline video clips, Together AI for legacy VIDEO pipeline, Seedance V2 planned for production). The agent orchestrates three deterministic steps: submit generation job, poll for completion, upload result to S3. Visual Generation explicitly does NOT generate text or typography — text captions are a separate concern handled by the text generation layer (CopywriterAgent, formatters). This is a hard domain separation boundary: the visual model renders the image; the text agent renders the caption.

_Avoid_: Asking the Video Generator Agent to embed text into generated videos. That is an integration failure, not a feature gap.

## Visual Style Theme

A user-facing constraint on the visual direction of a SHORT format video, selected from a fixed catalog (cinematic, minimalist, newsroom, documentary, dynamic) via a Discord dropdown at format-selection time. Injected into `story_directives` alongside the FormatJob creation and passed to the ShortFormatterAgent's plan prompt. The LLM generates the detailed `visual_style` string within bounds of the chosen theme — the theme is a creative constraint, not a hard schema. Catalog is hardcoded in the Discord bot's `ShortFormatSelectionView`; changing it requires a code deployment.

_Avoid:_ Letting the LLM decide the visual style entirely without user input — the output may not match the user's intent for the content's aesthetic.

## Unified Visual Prompt

A single 1-3 sentence prompt (produced by the VideoFormatterAgent) that synthesises all video scenes into one API-ready description. Used as the input for single-shot video generation APIs (Together AI during testing). The `unified_visual_prompt` includes platform aspect ratio and narrative arc coverage. Per-scene `visual_prompt` fields are retained alongside it for future per-scene rendering (Seedance V2).

## Video Generator Agent

A `ServiceAgent` (no LLM) that orchestrates the video rendering pipeline. Injected DI tools: `generate_video`, `poll_video`, `upload_video`. Replaces the earlier `AssetStudioAgent`'s video track (which only produced prompts and a placeholder URL). Always runs wrapped in `AgentHarness`.

## Video Generation Provider

A provider-agnostic abstraction (`VideoGenProvider` ABC in `app/services/video_gen.py`) that mirrors the LLM provider registry pattern. Three implementations: `KlingVideoGen` (SHORT pipeline video clips), `TogetherVideoGen` (legacy VIDEO pipeline, single-shot), and planned `SeedanceVideoGen` (production, per-scene). Selected via `video_gen_provider` env var. SHORT pipeline uses Kling AI because Together's `minimax/video-01-director` model has a 0% success rate in practice (always times out).

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

A job that renders a completed Script into a platform-specific format. Created by the Discord Bot when a user clicks a format button (e.g. "Create Carousel" or "Create Short") after a Script completes. Stored in the `factory.format_jobs` table with its own status enum (`format_job_status`). Managed by the QueueWorker via the same `FOR UPDATE SKIP LOCKED` pattern as RenderJobs. Always references a completed ScriptJob via `source_job_id`. Unique per `(source_job_id, platform, format_type)` — duplicate requests return the existing job.

Carries a `working_memory` JSONB column (mirroring ScriptJob) for Discord metadata: `discord_thread_id`, `discord_message_id`, and `final_embed_updated` flag. For SHORT format, all asset URLs are stored **inline** in `format_payload` JSONB rather than in the separate `Asset` table — see Inline Asset Storage.

## FormatJob Watcher

A background task in the Discord Bot process (not the QueueWorker) that polls `factory.format_jobs` every 5 seconds for rows containing a `discord_thread_id` in `working_memory`. On detecting a state change, it edits a single Living Embed message in the Discord thread to reflect the new state. Polling is scoped — only Discord-originated FormatJobs are watched; completed jobs drop out of scope immediately. On bot startup (`on_ready`), a one-shot Startup Re-Sync query catches FormatJobs that reached a terminal state while the bot was offline.

_Avoid:_ Calling it the "QueueWorker" — the FormatJob Watcher is read-only, non-locking, and lives entirely in the Discord Bot's event loop. The QueueWorker claims and mutates FormatJob rows; the Watcher observes them.

## Inline Asset Storage

The decision to store SHORT visual asset URLs (video clips, Ken Burns stills, voiceover, vocal alignment data) inside `FormatJob.format_payload` JSONB rather than inserting rows into the shared `Asset` table. Rationale (see ADR 0012): the `Asset` table has a hard FK to `render_jobs.id`, making it unsuitable for FormatJob-originated assets. A new `FormatAsset` table would create schema proliferation and synchronization risk. Since asset URLs are intermediate artifacts on the way to the final MP4 (not independent query targets), storing them inline in the same JSONB blob that carries the formatter output is the simplest correct approach. The `final_video_url` column on FormatJob is kept as a first-class column for direct access.

_Avoid:_ Expecting to find SHORT FormatJob assets via the `Asset` table. They live in `format_payload` — query the JSONB directly.

## Living Embed

A single Discord embed message posted when a FormatJob is created, then edited in-place as the job progresses through its states. Avoids spamming the thread with multiple progress messages. Fields: Status (emoji + label), Platform, Format, Duration (elapsed), Progress (step counter), and on completion the video URL + file upload. Updated by the FormatJob Watcher. When a SHORT FormatJob fails in the COMPOSITION phase, the embed shows a "Retry Composition" button.

_Avoid:_ Posting a new message for each state transition. Edit the embed — it's cleaner, and the Discord API supports it with no extra permissions.

## Composition Retry

A Discord button that appears on the Living Embed when a SHORT FormatJob fails in the COMPOSITION phase. Clicking it immediately defers the interaction (Discord 3-second requirement), then resets the FormatJob's status to `COMPOSITION` (not `PENDING`). The QueueWorker re-claims the job and re-runs only `_transition_composition()` — no re-formatting, no re-generating assets, no re-calling external APIs. This works because asset URLs are already persisted in `format_payload` from the prior successful `ASSET_GENERATION` pass, and `ShortComposerAgent` is a deterministic ServiceAgent.

_Avoid:_ Resetting to `PENDING` — that would require idempotency guards in every transition and waste API calls re-running successful phases.

## Startup Re-Sync

A one-time query run in the Discord Bot's `on_ready` hook. Scans `format_jobs` for rows where `status IN (COMPLETED, FAILED)` AND `working_memory` contains a `discord_message_id` AND `final_embed_updated` is `false`. For each match, posts the final result to the Discord thread and marks `final_embed_updated = true`. Prevents permanently stuck "PENDING" embeds after a bot container restart while the QueueWorker was processing a FormatJob.

## Two-Step Write Pattern

The sequence used to create a FormatJob with Discord metadata: (1) call `create_format_job()` with `working_memory={}` to get the DB row committed, (2) create the Discord thread and post the initial Living Embed, (3) update the FormatJob's `working_memory` with `discord_thread_id` and `discord_message_id`. This ordering guarantees the thread and embed exist before the working_memory references them. If thread creation fails, the FormatJob is already committed with an empty working_memory — it will still be processed by the QueueWorker, and the result can be recovered via Startup Re-Sync.

## Script Content Pipeline

The text-only subset of the Content Factory pipeline: Research → Retrieval → Scripting → Fact-Check → Optimizer Loop. Produces a script and claims, no formatting or assets. Run inline by the Discord Bot process — not via the QueueWorker. A separate, decoupled concern from the Format & Asset Pipeline, which is managed by the QueueWorker. The split happens at the ADR 0010 boundary: text work is real-time and bot-managed; rendering work is queued and QueueWorker-managed.

## Snapshot Context Handoff

When a FormatJob is created from a completed ScriptJob, the necessary context (script content, claims, refined context, story directives, hedge index, epistemic ledger) is **copied** into the FormatJob row at creation time, not referenced via FK. This ensures FormatJobs are fully self-contained — they survive garbage collection of the source ScriptJob, and a failure in one FormatJob cannot corrupt shared data. The `source_job_id` FK is retained for traceability only, not for runtime data access.

_Avoid:_ Treating `source_job_id` as a live data source at format-runtime. All data needed for formatting lives in the FormatJob's snapshot columns.

## Short

A format type (`SHORT`) for short-form vertical video (30–50s, up to 90s) targeting TikTok, Instagram Reels, and YouTube Shorts. Produces a composed MP4 via asset hybridization — per-scene video clips mixed with Ken Burns animated stills, overlaid with TTS voiceover, burned-in subtitles (karaoke-style per platform), and background music. Structurally distinct from `VIDEO` (which is a single-shot AI video with describe-only audio). Rendered by the Production Studio through a three-stage post-FORMATTING path: `ASSET_GENERATION` → `COMPOSITION` → `COMPLETED`.

Accessible via Discord through the two-step flow: `/script` → format selection → "Create Short" button → `ShortFormatSelectionView` (platform, visual_style_theme, loopable) → FormatJob → QueueWorker processes SHORT pipeline → FormatJob Watcher reports progress via Living Embed. See ADR 0012 for architecture.

_Avoid:_ Calling a Short a "video" — it follows a different pipeline path and produces a fundamentally different artifact.

## Short Format Payload

The structured output of the `ShortFormatterAgent`. A `ShortFormatPayload` containing: scenes (each with `asset_type` and optional `kb_motion`), `target_total_duration`, `visual_style`, `audio_direction`, `music_mood`, `voice_id`, `subtitle_preset`, and optional `loop_hook`. The `_format` discriminator is `"short"`. Stored as `Script.format_payload` with `Script.role = "format"` and `Script.format_type = "SHORT"`.

## Asset Hybridization

Per-scene visual generation strategy where the LLM tags each scene as either `video_clip` (generated via Kling AI) or `ken_burns` (a FLUX still animated with slow pan/zoom via FFmpeg `zoompan`). A validator enforces 1–2 `video_clip` scenes minimum per Short (auto-fixes violations). Cost-efficient and visually varied — avoids the monolithic single-shot approach of the `VIDEO` format.

## Ken Burns Motion

A preset camera movement applied to a static image to create the illusion of motion. Specified per scene as `kb_motion` with values: `pan_left`, `pan_right`, `zoom_in`, `zoom_out`, `static_zoom_in`. Required when `asset_type = "ken_burns"`, forbidden when `asset_type = "video_clip"`. Mapped to FFmpeg `zoompan` filter parameters in the Composition phase.

## TTS Provider

A provider-agnostic abstraction (`TTSProvider` ABC) for text-to-speech generation, mirroring the `VideoGenProvider` pattern. `ElevenLabsTTS` is the first implementation. Returns audio bytes plus `vocal_alignment_data` (word-level timestamps) for subtitle timing. Selected via config; swapping providers requires no pipeline changes.

## Vocal Alignment Data

Word-level timestamp JSON returned by the TTS provider alongside the voiceover audio. Used by the Composition phase to: (1) split the continuous voiceover into per-scene time ranges via string matching against each scene's `narration_text`, (2) generate `.ass` subtitle files with per-word karaoke highlighting. The source of truth for timing — scene `target_duration_seconds` is a pacing budget, not a hard constraint.

## Subtitle Preset

Platform-aware subtitle rendering template. Three presets: `CENTER_POP_YELLOW` (TikTok — bold karaoke with yellow highlight), `CLEAN_WHITE_LOWER` (YouTube — subtle bottom-center), `NEON_BOXED` (Instagram — boxed neon accent). Root-level on `ShortFormatPayload`, not per-scene. Mapped to hardcoded `.ass` style configurations in the Composition phase.

## Composition

A pipeline state (`COMPOSITION`) between `ASSET_GENERATION` and `COMPLETED` where the `ShortComposerAgent` assembles all raw assets (video clips, Ken Burns stills, voiceover, vocal alignment data, background music) into a final MP4 via FFmpeg. Deterministic, local, and fast — no external API calls. Pre-flight validation ensures all artifacts exist before FFmpeg runs. General-purpose: future formats (e.g., carousel PDF composition) can use this state.

_Avoid:_ Confusing Composition with ASSET_GENERATION. ASSET_GENERATION is all external I/O; Composition is local assembly.

## Short Composer Agent

A `ServiceAgent` (no LLM) that runs the 5-step composition pipeline: pre-flight validation → concurrent S3 download → .ass subtitle generation → atomic FFmpeg composition → S3 upload and cleanup. Receives asset URLs and alignment data from ASSET_GENERATION artifacts. Produces a single `SHORT_COMPOSED_VIDEO` asset.

## Short Visual Asset Agent

A `ServiceAgent` that generates per-scene visual assets for a Short. For scenes tagged `video_clip`, calls Kling AI's text-to-video API. For scenes tagged `ken_burns`, calls FLUX image generation. On video clip failure, retries once then falls back to Ken Burns, updating `asset_type` in-place. Ken Burns scenes are generated concurrently; video_clip scenes are sequential (submit→poll→download→upload per scene). All assets uploaded to S3 with appropriate `AssetType` (`SHORT_VIDEO_CLIP` or `SHORT_STILL_IMAGE`) and `render_meta.scene_number`.

## Short Voiceover Agent

A `ServiceAgent` that generates a continuous TTS voiceover track via a `TTSProvider` (ElevenLabs). Takes the concatenated scene narration texts and a `voice_id`, returns audio bytes plus `vocal_alignment_data`. Uploads a single `VOICEOVER` asset and a `VOCAL_ALIGNMENT` asset to S3.

## Loop Hook

A narrative bridge on `ShortFormatPayload` (populated when `story_directives.loopable = true`) describing how the final scene's narration connects back to the opening hook. Instructs the `ShortFormatterAgent` to write circular narrative structure. No FFmpeg stitch — the content itself loops by design.

## Kling Video Generation

The AI video generation service used by the SHORT pipeline for `video_clip` scenes. Kling AI produces 5-second or 10-second video clips from text prompts via the `text2video` API endpoint. Supports `9:16` aspect ratio natively (matching TikTok/YouTube Shorts). Authentication uses JWT (access key + secret key). Free tier provides 66 credits/day. Default model for testing: `kling-v1-6` with `std` mode (720p). Upgrade path: `kling-v2-6` or `kling-v3` with `pro` mode (1080p) via env var. See ADR 0013.

_Avoid:_ Using Kling for the legacy `VIDEO` format pipeline — that path still uses `TogetherVideoGen`.

## JWT Authentication (Kling)

The authentication scheme required by the Kling AI API. A JWT token is generated from an `access_key` + `secret_key` pair using `HS256` signing, valid for 30 minutes. The `KlingVideoGen` provider instance caches the token and refreshes it when within 5 minutes of expiry. This is stateful authentication (unlike Together's simple API key). Requires the `pyjwt` package.

## Duration Mapping

The rounding of a scene's `target_duration_seconds` to the nearest Kling-supported duration value. Kling's `text2video` endpoint accepts only `"5"` or `"10"` seconds (for v1/v2 models). Scenes targeting 3.0–7.0 seconds are rounded to `"5"`; scenes targeting 7.1–15.0 seconds are rounded to `"10"`. The generated video is slightly longer than the target, but the `ShortComposerAgent` already trims all scenes to their exact `target_duration_seconds` in FFmpeg, so no extra composition logic is needed.

## Scene Parallelism

The concurrent execution of Ken Burns scene generation within `ShortVisualAssetAgent`. Ken Burns scenes (FLUX image generation + upload) run in parallel via `asyncio.gather()`, while `video_clip` scenes remain sequential (each requires submit→poll→download→upload, which blocks on its own completion). The global FLUX rate limiter (`_min_gap`) still throttles concurrent requests, but the overhead between calls is parallelized. This is a performance optimization that reduces visual asset wall time for still-heavy shorts.

