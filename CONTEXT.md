# Content Factory — Domain Glossary

## Story

The central unit of work flowing through the pipeline. Represented in the database as a `RenderJob`. A Story has a topic (headline), a status, and carries all content produced through the pipeline (research, scripts, claims, assets).

## Script

The narrative body of a Story, written by the Writer's Desk. A script has a role (`master` for the general narrative, `format` for a format-specific adaptation), a version number, and optionally associated Claims from fact-checking. A master script is drafted first; format scripts are derived from it during the Layout Desk stage.

## Format Output

The structured, published-ready rendering of a Story in a specific format. Types: `BLOG`, `CAROUSEL`, `VIDEO`. Produced by the Layout Desk from the master script using a format-specific payload schema (sections for blog, slides for carousel, scenes for video). A format output may carry SEO metadata, hashtags, and source references.

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

A generated media file associated with a Story. Types: `VISUAL_VEO` (video clip), `AUDIO_LYRIA` (music/sound), `VOICEOVER`, `SUBTITLE_JSON`, `DATA_CHART`. Produced by the Production Studio (only for Stories targeting video formats). Each asset carries generation metadata (prompt, timing, SynthID watermark).

## Pipeline Statuses

Each Story passes through editorial desks in sequence. The canonical outward-facing names (used in UI, user communication) and their backing enum values:

| Desk (Outward) | Enum Value | Terminal? |
|---|---|---|
| Queued | `PENDING` | No |
| Research Desk | `RESEARCHING` | No |
| Source Verification | `FACT_CHECKING_RESEARCH` | No |
| Writer's Desk | `SCRIPTING` | No |
| Fact-Check Desk | `FACT_CHECKING_SCRIPT` | No |
| Layout Desk | `FORMATTING` | No |
| Production Studio | `ASSET_GENERATION` | No |
| Published | `COMPLETED` | Yes |
| Killed | `FAILED` | Yes |
| Your Review | `HUMAN_REVIEW_NEEDED` | Yes (blocked on human)

## Platform

The target social media platform for a Story. Determines which formats are available and
per-character limits for carousel content. Supported platforms: Twitter/X, LinkedIn,
Instagram, TikTok, YouTube. Platform is required at creation time (platform-first policy).

