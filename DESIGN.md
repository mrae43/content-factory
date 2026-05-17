# Content Factory — Design Source of Truth
> The visual, interaction, and architectural design document for the Content Factory UI.
> Every UI decision traces back to this file. When in doubt, read this first.
---
## 1. Design Philosophy
### Core Identity
Content Factory is a **personal editorial command center** — not a SaaS dashboard, not an admin panel, not an AI playground. It is the tool of an editor who codes: someone who commissions content, monitors the editorial process, reviews proofs, and publishes — but also wants full visibility into the multi-agent machinery underneath.
### Guiding Metaphor: The Editorial Office
The pipeline is not a conveyor belt. It is a newsroom with desks:
| Pipeline Stage | Editorial Desk | Visual Role |
|---|---|---|
| `PENDING` | Assignment Queue | Waiting to be picked up |
| `RESEARCHING` | Research Desk | Gathering sources and context |
| `FACT_CHECKING_RESEARCH` | Source Verification | Validating research (passthrough in MVP) |
| `SCRIPTING` | Writer's Desk | Drafting the narrative |
| `FACT_CHECKING_SCRIPT` | Fact-Check Desk | Three-pass claim evaluation |
| `FORMATTING` | Layout Desk | Multi-format typesetting |
| `ASSET_GENERATION` | Production Studio | Video/audio asset creation |
| `COMPLETED` | Published | Final output, ready to use |
| `FAILED` | Killed Story | Error details for post-mortem |
| `HUMAN_REVIEW_NEEDED` | Editor's Review | Awaiting your decision |
This metaphor shapes every visual and interaction decision. The UI should feel like opening the door to your private editorial office every time you load the app.
### Design Principles
1. **Output is hero.** The finished content (blog, carousel, video script) is rendered as if it could be published. Raw data is secondary, always accessible but never primary.
2. **Process as narrative.** The pipeline journey tells a story — research → writing → fact-checking → formatting. The UI reads like an editorial audit trail, not a database dump.
3. **Warmth over sterility.** Warm grays, serif headings, paper-like backgrounds. The UI should feel like a well-designed magazine, not a SaaS tool.
4. **Craft, not generated.** Every element should feel deliberately placed. No default spacings, no unstyled text, no raw JSON visible by default. If it looks like an AI made it, it's wrong.
5. **Full depth on demand.** Technical detail (agent traces, JSON payloads, revision diffs, claim evidence) is always one click away — in collapsible sections, tooltips, or expandable panels.
---
## 2. Visual Language
### Color Palette
**Primary palette — Stone & Copper:**
| Token | Light Mode | Usage |
|---|---|---|
| `background` | `oklch(0.965 0.006 84.6)` | Page background — warm stone, like aged paper |
| `foreground` | `oklch(0.228 0.009 75.2)` | Primary text — warm charcoal, not pure black |
| `card` | `oklch(0.994 0.006 84.6)` | Card surfaces — warm white, slightly cream |
| `card-foreground` | `oklch(0.228 0.009 75.2)` | Text on cards |
| `muted` | `oklch(0.920 0.010 81.8)` | Subtle backgrounds, inactive elements |
| `muted-foreground` | `oklch(0.553 0.012 58.1)` | Secondary text, labels, timestamps |
| `border` | `oklch(0.860 0.015 80.7)` | Borders, dividers — warm gray |
| `ring` | `oklch(0.599 0.140 37.4)` | Focus rings — copper accent |
**Accent — Burnt Sienna (copper):**
| Token | Light Mode | Usage |
|---|---|---|
| `primary` | `oklch(0.599 0.140 37.4)` | Primary actions, active states, accent highlights |
| `primary-foreground` | `oklch(0.994 0.006 84.6)` | Text on primary buttons/badges |
| `accent` | `oklch(0.931 0.014 57.6)` | Hover states, highlighted rows — copper tint |
| `accent-foreground` | `oklch(0.228 0.009 75.2)` | Text on accent surfaces |
**Semantic colors (status badges, indicators):**
| Token | Light Mode | Usage |
|---|---|---|
| `success` | `oklch(0.536 0.080 162.0)` | SUPPORTED claims, COMPLETED status |
| `warning` | `oklch(0.652 0.132 81.6)` | UNCERTAIN claims, active processing |
| `destructive` | `oklch(0.505 0.190 27.5)` | UNSUPPORTED claims, FAILED status, errors |
| `info` | `oklch(0.538 0.095 257.8)` | CONTESTED claims, informational states |
**Surface hierarchy:**
```
background (oklch(0.965 0.006 84.6))          ← page
  └── card (oklch(0.994 0.006 84.6))          ← elevated surface
       └── muted (oklch(0.920 0.010 81.8))    ← inset/depressed areas
            └── border (oklch(0.860 0.015 80.7)) ← dividers
```
### Typography
**Font families:**
| Role | Font | Rationale |
|---|---|---|
| **Display/Headings** | Playfair Display (serif) | Editorial authority, publication feel. Used for page titles, section headings, the masthead. |
| **Body/UI** | Inter (sans-serif) | Clean readability for data-dense content. Used for body text, labels, buttons, navigation. |
| **Mono** | JetBrains Mono | Code, technical detail, job IDs, timestamps. Used in collapsible raw-data sections. |
**Type scale:**
| Element | Font | Size | Weight | Tracking |
|---|---|---|---|---|
| Page title | Playfair Display | 2rem (32px) | 700 | -0.02em |
| Section heading | Playfair Display | 1.5rem (24px) | 600 | -0.01em |
| Card title | Playfair Display | 1.125rem (18px) | 600 | normal |
| Body text | Inter | 0.875rem (14px) | 400 | normal |
| Small/label | Inter | 0.75rem (12px) | 500 | 0.02em |
| Mono data | JetBrains Mono | 0.8125rem (13px) | 400 | normal |
**Editorial treatments:**
- **Drop caps** on blog section body text (first letter, Playfair Display, 3.5rem, copper color)
- **Pull quotes** for key claims or takeaways (Playfair Display italic, larger size, copper border-left)
- **Section dividers** as thin copper rules (`border-bottom: 1px solid oklch(0.599 0.140 37.4 / 0.3)`)
- **Masthead** "Content Factory" in Playfair Display, bold, with a subtle copper underline
### Spacing & Layout
**Base unit:** 4px. Spacing tokens at 4, 8, 12, 16, 24, 32, 48, 64.
**Content max-width:** 72rem (1152px) for the main content area. Editorial design breathes — generous margins, no full-bleed data.
**Card treatment:**
```
border-radius: 8px (subtle, not round)
border: 1px solid var(--border)
background: var(--card)
padding: 24px (section cards), 16px (inline cards)
box-shadow: 0 1px 2px oklch(0.228 0.009 75.2 / 0.04)
```
No heavy shadows. No ring borders. Cards are like printed cards on a desk — slight elevation, warm tone.
**Whitespace philosophy:** More than you think. Editorial design uses whitespace to separate, not borders. If two sections are far apart, they don't need a line between them. Let the type breathe.
---
## 3. App Shell
### Sidebar
```
┌──────────────────────────┐
│                          │
│  Content Factory         │  ← Masthead: Playfair Display, 1.25rem, bold
│  ───────────────         │     Copper underline (2px)
│                          │
│  Overview                │  ← Navigation items with editorial names
│  Stories                 │     Active: copper text + copper left border
│  Commission              │     Hover: accent background
│                          │     Inactive: muted-foreground
│                          │
│                          │
│                          │
│                          │  ← Generous empty space. Not packed.
│                          │
│                          │
│                          │
│                          │
│                          │
│  v1.0                    │  ← Version label, bottom, subtle
└──────────────────────────┘
```
- **Width:** 240px (w-60), fixed position, warm stone background (`var(--background)`)
- **Right border:** 1px solid `var(--border)` — no heavy separation
- **No icons.** Editorial design is text-driven. Navigation is pure text.
- **Collapsible:** collapses to 0 width on mobile/toggle. No slide animation needed.
### Header
```
┌─────────────────────────────────────────────────────────┐
│  Overview                                    [Commission]│
└─────────────────────────────────────────────────────────┘
```
- **Left:** Current page name in Playfair Display (matches sidebar active item)
- **Right:** Primary action button ("Commission" in copper)
- **Height:** h-14, border-bottom with `var(--border)`
- **Background:** `var(--card)` — slightly warmer than page bg
- **No hamburger menu** — sidebar toggle is a subtle icon if needed
---
## 4. Page Designs
### 4.1 Dashboard — "Overview"
**Concept: The editor's desk. What's happening right now.**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Overview                                       [Commission]│
│                                                             │
│  ── Active Stories ──────────────────────────────────────── │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  BRICS De-dollarization 2025                         │   │
│  │  Writer's Desk · Started 12 min ago                  │   │
│  │  [Research ✓] [Verification ✓] [Writing ●] [Fact-Ch]│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Quantum Computing Breakthroughs                     │   │
│  │  Fact-Check Desk · Started 28 min ago                │   │
│  │  [Research ✓] [Verification ✓] [Writing ✓] [Fact-Ch ●]│  │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ── Recently Published ────────────────────────────────── │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  AI Regulation in 2026                               │   │
│  │  Blog + Carousel · Published 2 hours ago             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Climate Tech Investment Trends                      │   │
│  │  All Formats · Published yesterday                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ── Killed Stories ────────────────────────────────────── │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Gene Therapy Ethics                                 │   │
│  │  Failed · 3 days ago                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
**Structure:**
1. **Active Stories** — Jobs in non-terminal states. Each card shows:
   - Topic as card title (serif)
   - Current desk name + elapsed time
   - Mini pipeline progress (dot-based, not the full bar)
   - Click navigates to job detail
2. **Recently Published** — Completed jobs. Cards show:
   - Topic + format badges
   - Relative timestamp
   - Click navigates to job detail
3. **Killed Stories** — Failed jobs. Collapsed by default, expandable.
   - Topic + "Failed" indicator
   - Error summary on expand
**Empty state:** Single centered card with editorial copy: "No stories yet. Commission your first piece." + copper "Commission" CTA.
**No stat cards.** Stats are small counters inline in section headers ("Active Stories (3)", "Recently Published (12)").
### 4.2 Jobs List — "Stories"
**Concept: The assignment board.**
Same card-list pattern as Dashboard, but with filtering and full history.
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Stories                                        [Commission]│
│                                                             │
│  (all) (active) (published) (review) (killed)              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  BRICS De-dollarization 2025          [Blog] [Active]│   │
│  │  Writer's Desk · 12 min ago                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  AI Regulation in 2026          [Blog] [Carousel] [✓]│   │
│  │  Published · 2 hours ago                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
**Filter pills:** Editorial language, not raw status names.
| Filter | Maps to Statuses |
|---|---|
| All | Everything |
| Active | PENDING, RESEARCHING, FACT_CHECKING_RESEARCH, SCRIPTING, FACT_CHECKING_SCRIPT, FORMATTING, ASSET_GENERATION |
| Published | COMPLETED |
| Review | HUMAN_REVIEW_NEEDED |
| Killed | FAILED |
**Card metadata:** Format badges as small copper-tinted pills. Status as a single word in the appropriate semantic color. Desk name + relative timestamp.
### 4.3 Create Job — "Commission Content"
**Concept: Giving the newsroom its assignment. One focused form, fast.**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Commission Content                            [Commission]│
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  Headline                                           │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ What's the story?                             │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  Editorial Brief                                    │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ Background, key points, angle, tone...       │   │   │
│  │  │                                               │   │   │
│  │  │                                               │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  Publication Target                                  │   │
│  │                                                     │   │
│  │  Format          Platform        Strictness          │   │
│  │  [All Formats ▾] [None ▾]        [High ▾]           │   │
│  │                                                     │   │
│  │  ── Research Materials ──────────────────────────    │   │
│  │  Source URLs                                         │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ https://...                                   │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  │  Reference Text                                     │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ Raw text, book excerpts, reports...           │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  [Commission This Story]                            │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
**Design rules:**
- Single card, max-w-2xl, centered
- Section labels in Playfair Display (Headline, Editorial Brief, Publication Target)
- "Research Materials" section is collapsible (collapsed by default — most jobs don't need it)
- Primary CTA: copper button, full width, serif label "Commission This Story"
- Confirmation step: warm muted background summary card with Confirm/Edit buttons
- Form fields: warm white background, warm border, no harsh outlines
- Error state: inline copper-tinted error card below the form
### 4.4 Job Detail — The Story
**Concept: Opening a story file. Output-first, then audit trail.**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  BRICS De-dollarization 2025                                │
│  Job #a3f2... · Commissioned Oct 13, 2026                   │
│  [Blog] [Carousel] [Active]                                 │
│                                                             │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 1: THE PUBLISHED PIECE                             │
│  ═══════════════════════════════════════════════════════════ │
│                                                             │
│  (For COMPLETED/PUBLISHED jobs: rendered format outputs)    │
│                                                             │
│  ┌─ Blog ──── Carousel ──── Video Script ──────┐           │
│  │                                              │           │
│  │  ┌──────────────────────────────────────┐   │           │
│  │  │  Drop cap  he rapid shift in global   │   │           │
│  │  │  economic dynamics has accelerated...  │   │           │
│  │  │                                       │   │           │
│  │  │  Key Takeaway                         │   │           │
│  │  │  ┃ "Central banks are diversifying     │   │           │
│  │  │  ┃  reserves at unprecedented rates"   │   │           │
│  │  │                                       │   │           │
│  │  │  Sources: 4 research chunks referenced │   │           │
│  │  └──────────────────────────────────────┘   │           │
│  │                                              │           │
│  │  ┌──────────────────────────────────────┐   │           │
│  │  │  Section 2: The Mechanics...          │   │           │
│  │  └──────────────────────────────────────┘   │           │
│  │                                              │           │
│  └──────────────────────────────────────────────┘           │
│                                                             │
│  SEO: "BRICS De-dollarization 2025: ..."                   │
│  Tags: economics, brics, currency, geopolitics              │
│                                                             │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 2: THE EDITORIAL TRAIL                             │
│  ═══════════════════════════════════════════════════════════ │
│                                                             │
│  ┌─ Research Desk ──────────────────────────────┐          │
│  │  ✓ Completed · 4 min · 12 chunks ingested     │          │
│  │                                                │          │
│  │  ┌──────────────────────────────────────┐     │          │
│  │  │  Research Summary                     │     │          │
│  │  │  <rendered markdown, not raw pre>     │     │          │
│  │  │  Confidence: 0.87                     │     │          │
│  │  └──────────────────────────────────────┘     │          │
│  │                                                │          │
│  │  [▼ Technical Details] collapsed               │          │
│  └────────────────────────────────────────────────┘          │
│                         │                                   │
│                         ▼                                   │
│  ┌─ Source Verification ────────────────────────┐          │
│  │  ✓ Passthrough · <1s                          │          │
│  └────────────────────────────────────────────────┘          │
│                         │                                   │
│                         ▼                                   │
│  ┌─ Writer's Desk ──────────────────────────────┐          │
│  │  ✓ Completed · 6 min · v2 (revised)           │          │
│  │                                                │          │
│  │  ┌──────────────────────────────────────┐     │          │
│  │  │  <rendered script with proper prose>  │     │          │
│  │  └──────────────────────────────────────┘     │          │
│  │                                                │          │
│  │  [▼ Technical Details] collapsed               │          │
│  └────────────────────────────────────────────────┘          │
│                         │                                   │
│                         ▼                                   │
│  ┌─ Fact-Check Desk ────────────────────────────┐          │
│  │  ✓ Approved · 3 min · 8 claims                │          │
│  │                                                │          │
│  │  ┌──────────────────────────────────────┐     │          │
│  │  │  Claim: "BRICS accounts for 35%..."  │     │          │
│  │  │  [SUPPORTED] 95% confidence           │     │          │
│  │  │  Sources: chunk #a3f2, chunk #b7c1    │     │          │
│  │  └──────────────────────────────────────┘     │          │
│  │  ┌──────────────────────────────────────┐     │          │
│  │  │  Claim: "China sold $50B in..."      │     │          │
│  │  │  [CONTESTED] 62% confidence           │     │          │
│  │  │  Sources: chunk #d2e9                 │     │          │
│  │  └──────────────────────────────────────┘     │          │
│  │  ...                                           │          │
│  │                                                │          │
│  │  [▼ Full Audit Trail] collapsed                │          │
│  └────────────────────────────────────────────────┘          │
│                         │                                   │
│                         ▼                                   │
│  ┌─ Layout Desk ────────────────────────────────┐          │
│  │  ✓ Blog + Carousel + Video · 8 min            │          │
│  │  [▼ Harness Details] collapsed                │          │
│  └────────────────────────────────────────────────┘          │
│                                                             │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 3: EDITORIAL REVIEW (only if HUMAN_REVIEW_NEEDED)  │
│  ═══════════════════════════════════════════════════════════ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  This story needs your review.                      │   │
│  │  3 revision cycles exhausted.                       │   │
│  │                                                     │   │
│  │  [Approve & Publish]  [Request Revision]             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
**Structure — three sections, strict order:**
**Section 1: The Published Piece (output-first)**
For completed jobs: fully rendered format outputs in tabs. Blog sections get drop caps, proper paragraphs, pull quotes for key takeaways. Carousel slides get numbered cards with visual prompts. Video scenes get a storyboard grid with narration, visual, and audio cues side by side.
For active jobs: this section shows the current desk's live output (what's being worked on right now), with a subtle pulsing copper indicator. The API returns partial data at each stage:

| Active Stage | What Renders in "Published Piece" | API Data Source |
|---|---|---|
| `PENDING` | Assignment summary card — topic, format, platform. No output yet. | `topic`, `format_type`, `platform` |
| `RESEARCHING` / `FACT_CHECKING_RESEARCH` | Research summary card — `refined_context` rendered as prose, with "Research in progress..." indicator if `refined_context` is still null | `refined_context` |
| `SCRIPTING` | Current script draft — latest `script.content` rendered as prose. If multiple revisions, show "Draft v{version}" label. | `scripts[latest].content`, `scripts[latest].version` |
| `FACT_CHECKING_SCRIPT` | Current script + live claim evaluation — script rendered as above, plus partial claim cards as they arrive via polling | `scripts[latest].content` + `scripts[latest].claims[]` |
| `FORMATTING` | Format outputs as they populate — tabs show "Generating..." placeholder until `format_payload` arrives | `scripts[]` where `format_type != "VIDEO"` → `format_payload` |
| `ASSET_GENERATION` | Video format output + asset cards showing generation status. Mock `s3://` URLs shown as "Pending render..." | `scripts[VIDEO].format_payload` + `assets[]` |

For failed jobs: this section is replaced by a "Killed Story" error card with structured error display and a "Technical Details" collapsible with the raw error_log JSON.
**Section 2: The Editorial Trail (process-as-narrative)**
Vertical timeline. Each node is a desk. Structure per node:
1. **Header:** Desk name (serif), completion status (semantic color icon), duration, summary stat
2. **Output preview:** The key output from that stage — rendered, not raw
3. **Collapsible "Technical Details":** Agent used, model, temperature, raw JSON payload, token usage. This is where the "editor who codes" digs in.

**Input context in the trail:** The Research Desk node includes a "Source Materials" sub-section showing:
- `pre_context.source_urls` as a list of clickable links (Inter 0.75rem, copper color)
- `pre_context.raw_text` in a collapsible "Reference Text" block (collapsed by default, shown as first ~100 chars + "...")
- `pre_context.target_audience` and `pre_context.guardrail_strictness` as small muted labels
**Revision history in the trail:** The Writer's Desk node shows revision cycles when `feedback_history` is non-empty:
- Each revision cycle is a nested card: "Revision 2 → Revision 3" with the `overall_reasoning` from `OptimizerFeedbackEntry`
- For structured feedback (`feedback_type === "structured_claims"`): show the list of `failed_claims` with their verdicts and evidence
- For legacy string feedback: show the raw feedback text in a muted blockquote
- The final script shown is always the latest version
**Assets in the trail:** The Production Studio node (only present for `format_type = "video"` or `"all"`) shows:
- Asset cards for each entry in `assets[]`: type badge (`VISUAL_VEO`, `AUDIO_LYRIA`, etc.), `url_or_path`, `render_meta` (start/end time, prompt used)
- Assets render as a grid of cards with the asset type as a small copper badge and the URL/path in monospace
- `render_meta.prompt_used` shown in a collapsible "Generation Prompt" section

Claim cards in the Fact-Check Desk node:
- Verdict badge (semantic color): SUPPORTED (green), CONTESTED (amber), UNSUPPORTED (red), UNCERTAIN (blue)
- Confidence percentage as a subtle bar
- Evidence sources linked to research chunks
- Category tag (statistic, attribution, chronological, causal, comparative)
**Section 3: Editorial Review (conditional)**
Only visible when `status === HUMAN_REVIEW_NEEDED`. A warm copper-bordered card with:
- Reason for escalation (revision count, Red Team escalation)
- Two buttons: "Approve & Publish" (copper primary) and "Request Revision" (outline)
### 4.5 Error Display
Errors are not red JSON dumps. They are structured:
```
┌─────────────────────────────────────────────────────┐
│  ⚠ Story Killed                                     │
│                                                     │
│  The fact-check desk encountered an error during     │
│  claim evaluation.                                   │
│                                                     │
│  Phase: FACT_CHECKING_SCRIPT                         │
│  Agent: RedTeamAgent                                 │
│  Time: Oct 13, 2026 2:34 PM                         │
│                                                     │
│  [▼ Technical Details]                               │
│  ┌──────────────────────────────────────────────┐   │
│  │  Traceback (most recent call last):            │   │
│  │  File "agents.py", line 412, in ...            │   │
│  │  ...                                            │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```
---
## 5. Component Specifications
### 5.1 Masthead
```
Content Factory
───────────────
```
- Font: Playfair Display, 1.25rem, 700
- Color: `var(--foreground)`
- Underline: 2px solid `var(--primary)` (copper), width auto to text
- Padding: 24px 20px
- No logo mark. The typography IS the logo.
### 5.2 Navigation Items
- Font: Inter, 0.875rem, 500
- Padding: 8px 20px
- Active state: `color: var(--primary)`, `border-left: 3px solid var(--primary)`, `background: var(--accent)`
- Hover state: `background: var(--accent)`
- Inactive state: `color: var(--muted-foreground)`
- Spacing between items: 4px
- No icons, no badges, no counts
### 5.3 Format Badges
Small pills indicating content format. Editorial treatment:
- Background: `var(--primary)` at 10% opacity
- Text: `var(--primary)` (copper)
- Font: Inter, 0.6875rem (11px), 600, uppercase, letter-spacing 0.05em
- Border-radius: 4px (not fully round — editorial, not tech)
- Spacing: 4px between badges
Values: `BLOG`, `CAROUSEL`, `VIDEO`, `ALL`
### 5.4 Status Indicators
Status is communicated through desk names, not raw enum values. A status badge uses:
- A small dot (8px) in the semantic color
- Desk name text in Inter 0.6875rem
- No background pill — just dot + text
| Status | Dot Color | Label |
|---|---|---|
| PENDING | `var(--muted-foreground)` | Queued |
| RESEARCHING | `var(--warning)` | Research Desk |
| FACT_CHECKING_RESEARCH | `var(--warning)` | Source Verification |
| SCRIPTING | `var(--info)` | Writer's Desk |
| FACT_CHECKING_SCRIPT | `var(--info)` | Fact-Check Desk |
| FORMATTING | `var(--accent-purple)` | Layout Desk |
| ASSET_GENERATION | `var(--accent-teal)` | Production |
| COMPLETED | `var(--success)` | Published |
| FAILED | `var(--destructive)` | Killed |
| HUMAN_REVIEW_NEEDED | `var(--warning)` | Your Review |
### 5.5 Claim Cards
Each fact-check claim rendered as:
```
┌─────────────────────────────────────────────────────┐
│  "BRICS nations now account for 35% of global       │
│   GDP measured by purchasing power parity."          │
│                                                      │
│  ● SUPPORTED                                    95%  │
│  Statistic · sources: 4 chunks                      │
│                                                      │
│  [▼ Evidence]                                        │
└─────────────────────────────────────────────────────┘
```
- Claim text in Inter, 0.875rem, italic (quoted)
- Verdict dot + label in semantic color
- Confidence as a subtle copper bar (width proportional to percentage)
- Category tag in small uppercase muted text
- Evidence section is collapsible, showing linked research chunk previews
### 5.6 Pipeline Progress (Mini)
For job cards in the Dashboard/Stories list:
```
[●][●][○][○][○][○][○]
```
- 7 dots for 7 meaningful stages (collapsing passthrough stages)
- Completed: filled dot in `var(--success)`
- Active: filled dot in `var(--primary)` with subtle pulse
- Future: hollow dot in `var(--border)`
- Skipped: tiny dot in `var(--muted)`
- No labels on the mini version — hover for tooltip with desk name
### 5.7 Timeline Node
For the editorial trail in job detail:
```
  ●  Research Desk
  │  ✓ Completed · 4 min · 12 chunks
  │
  │  <output preview card>
  │
  │  [▼ Technical Details]
  │
  ●  Source Verification
  │  ✓ Passthrough · <1s
  │
  ●  Writer's Desk
  ...
```
- Dot: 12px circle, semantic color for status
- Vertical connector: 2px line in `var(--border)`
- Desk name: Playfair Display, 1rem
- Duration + summary: Inter, 0.75rem, `var(--muted-foreground)`
- Output card: inset with `var(--muted)` background, proper typesetting
- Technical details: collapsible, monospace, muted background
### 5.8 Collapsible Sections
Used throughout for technical depth. Pattern:
- Trigger: `[▶ Technical Details]` — Inter 0.75rem, `var(--muted-foreground)`
- Expanded trigger: `[▼ Technical Details]` — `var(--primary)`
- Content: indented, `var(--muted)` background, `border-radius: 6px`, monospace for code/JSON
- Animation: none (editorial is instant, not bouncy)
---
## 6. Content Rendering Specifications
### 6.1 Blog Output
Each blog section rendered as:
- **Heading:** Playfair Display, 1.25rem, 600
- **Body:** Inter, 0.9375rem (15px), line-height 1.7, `var(--foreground)`
  - First paragraph: **drop cap** (first letter in Playfair Display, 3.5rem, `var(--primary)`, float left)
- **Key takeaway:** Pull quote style — left border 3px `var(--primary)`, Playfair Display italic, 1rem, `var(--muted-foreground)`, padding-left 16px
- **Sources used:** Small muted text "Sources: N research chunks" with expandable chunk previews
- **Word count:** Small muted label, right-aligned
SEO metadata and tags in a collapsed "SEO Details" section:
- Meta title, meta description as plain text
- Keywords as small copper pills (same style as format badges)
- Tags as larger editorial labels
### 6.2 Carousel Output
Each slide rendered as:
```
┌─────────────────────────────────────────┐
│  01                                      │
│                                          │
│  The slide text content rendered here    │
│  with proper line-height and spacing     │
│                                          │
│  Visual: A split-screen showing...       │  ← muted, smaller
│  Hook: question                          │  ← badge
│                                          │
│  127 / 280 characters                    │  ← char count
└─────────────────────────────────────────┘
```
- Slide number: Playfair Display, 2rem, `var(--primary)`, top-left
- Text: Inter, 0.9375rem, line-height 1.6
- Visual prompt: Inter, 0.75rem, `var(--muted-foreground)`, italic
- Hook type: small badge (same format badges style)
- Character count: muted, small, with warning color if near limit
Thread title at top. Hashtags as copper pills at bottom. CTA slide highlighted with copper border.
### 6.3 Video Script Output
Each scene rendered as a storyboard card:
```
┌──────────────────────────────────────────────────────┐
│  Scene 1                                    15s       │
│                                                       │
│  ┌─ Visual ──────────┐  ┌─ Narration ──────────────┐│
│  │ A wide aerial shot │  │ "In 2026, the global     ││
│  │ of a modern trading│  │  financial landscape is   ││
│  │ floor with screens │  │  undergoing a seismic..." ││
│  │ showing currency...│  │                           ││
│  └────────────────────┘  └───────────────────────────┘│
│                                                       │
│  Audio: Ambient trading floor sounds, low hum         │
│                                                       │
└──────────────────────────────────────────────────────┘
```
- Scene number + duration in header row
- Two-column layout: visual prompt (left), narration (right)
- Audio cue: full-width footer, Inter italic, `var(--muted-foreground)`
- Total duration shown at top of the storyboard section
- Visual style and audio direction as editorial notes above the grid
---
## 7. Interaction Patterns
### Transitions
- **Page navigation:** Instant. No animations. Editorial apps are crisp.
- **Collapsible expand/collapse:** Instant. No spring physics.
- **Hover states:** `transition-colors 150ms` on interactive elements only (cards, buttons, nav items).
- **Active pulse:** The mini-pipeline active dot uses a subtle `opacity` pulse (0.7 to 1.0, 2s cycle). Not distracting.
- **Loading states:** Skeleton components matching the exact layout of the content that will replace them. Warm stone color, not gray.
### Polling
- Job list: 5s interval when any job is active
- Job detail: 3s interval when job is in a non-terminal state
- Stop polling immediately on terminal states (COMPLETED, FAILED, HUMAN_REVIEW_NEEDED)
- No loading spinners during polling refreshes — seamless data updates
### Empty States
Every empty state is a warm card with editorial copy, not a sterile message:
| Context | Message | CTA |
|---|---|---|
| Dashboard, no jobs | "No stories yet. Commission your first piece." | [Commission] |
| Jobs list, no results | "No stories match this filter." | [Clear Filter] |
| Job detail, loading | Skeleton matching the full page layout | — |
### Error Handling
- **Form errors:** Inline copper-tinted alert card below the form field. Not red.
- **API errors:** Toast notification (sonner) with warm styling.
- **Job failures:** Structured "Killed Story" card in job detail (see 4.5).
---
## 8. Dark Mode (Secondary)
Dark mode is supported but not the primary identity. It exists for late-night sessions.
**Dark palette:**
| Token | Dark Mode |
|---|---|
| `background` | `oklch(0.210 0.008 84.6)` |
| `foreground` | `oklch(0.965 0.006 84.6)` |
| `card` | `oklch(0.255 0.006 56.1)` |
| `muted` | `oklch(0.299 0.008 75.3)` |
| `border` | `oklch(0.347 0.009 67.5)` |
| `primary` | `oklch(0.656 0.133 40.1)` (slightly lighter copper for contrast) |
| `accent` | `oklch(0.266 0.012 55.8)` |
Typography unchanged. Serif headings remain. The warm tones carry through — this is warm dark, not cold dark.
Toggle in sidebar footer, simple icon button. No system preference detection.
---
## 9. Implementation Notes
### What Changes From Current Code
| Component | Current | New |
|---|---|---|
| `globals.css` | Achromatic oklch tokens + `--font-heading: var(--font-sans)` | Stone & copper oklch palette + `--font-heading` wired to Playfair Display CSS variable |
| `layout.tsx` | Geist + Geist_Mono via `next/font/local` | Playfair Display + Inter + JetBrains Mono via `next/font/google`. Replace `localFont()` calls with `Google()` — adds 3 font downloads but Google Fonts caches aggressively. Wire font CSS variables: `--font-heading` → Playfair Display, `--font-sans` → Inter, `--font-mono` → JetBrains Mono. Remove Geist imports entirely. |
| `sidebar.tsx` | Plain text nav, no masthead | Masthead treatment, editorial labels |
| `header.tsx` | "Menu" + "Content Factory" | Page name + primary CTA |
| `page.tsx` (dashboard) | 3 stat cards + job list | Active Stories + Published + Killed sections |
| `jobs/page.tsx` | Raw status filter pills | Editorial filter labels (active/published/review/killed) |
| `jobs/new/page.tsx` | "Create New Job" form | "Commission Content" with editorial labels |
| `jobs/[id]/page.tsx` | Vertical card stack | Output-first + editorial timeline + review |
| `state-machine-progress.tsx` | Segmented pill bar — **DELETE, full replacement** (not restyle) | Split into two new components: `MiniPipeline` (dot row for cards) + `EditorialTimeline` (vertical timeline for detail). Current component exports nothing reused elsewhere. |
| `claim-card.tsx` | Card with badge | Styled claim card with dot verdict + confidence bar |
| `blog-viewer.tsx` | Structured but plain | Drop caps, pull quotes, editorial typesetting |
| `carousel-viewer.tsx` | Slide cards | Numbered slides with editorial treatment |
| `video-script-viewer.tsx` | Scene list | Storyboard grid layout |
| `format-badge.tsx` | Color-coded pills | Copper-tinted editorial labels |
| `job-status-badge.tsx` | Colored Badge pill (`bg-{color}/15 text-{color}`) | **DELETE, full replacement** with new `StatusDot` component (dot + desk name). Current pill style has no equivalent in the editorial system. |
### New Components Needed
| Component | Purpose |
|---|---|
| `EditorialTimeline` | Vertical timeline with desk nodes, connectors, collapsible details |
| `Masthead` | "Content Factory" serif treatment with copper underline |
| `DropCap` | Wrapper component for drop-cap first letter in paragraphs |
| `PullQuote` | Styled blockquote with copper left border |
| `CollapsibleSection` | Reusable `[▼ Technical Details]` pattern |
| `StatusDot` | 8px dot + desk name label |
| `MiniPipeline` | Dot-based progress for job cards |
### Files That Don't Change
- All API hooks (`use-jobs.ts`) — data layer unchanged
- API client (`api-client.ts`) — unchanged
- Zustand store (`ui-store.ts`) — add `theme` toggle, rest unchanged
- All backend code — this is purely a frontend redesign
### Migration Strategy
1. **Phase 1: Palette + Typography** — Update `globals.css` color tokens + `@theme inline` block. Replace Geist font imports in `layout.tsx` with `next/font/google` (Playfair Display, Inter, JetBrains Mono). Wire `--font-heading` to Playfair Display, `--font-sans` to Inter, `--font-mono` to JetBrains Mono. Instant visual shift across the entire app.
2. **Phase 2: Shell** — Redesign sidebar and header. Masthead treatment.
3. **Phase 3: Dashboard + Stories** — Restructure the list pages with editorial sections.
4. **Phase 4: Job Detail** — Output-first layout, editorial timeline, card-based content rendering.
5. **Phase 5: Commission Form** — Editorial labeling and refined layout.
6. **Phase 6: Polish** — Dark mode, empty states, micro-interactions, claim card styling.
---
## 10. Decision Log
Decisions made during the design process, recorded for traceability.
| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Doc scope | Combined design + UX + architecture | Pain point is disconnected UI from app essence; need full picture |
| 2 | Audience | Personal use, power user | Simplifies persona work; no non-technical users to accommodate |
| 3 | Feel/direction | Organic/Editorial | Chosen by owner; magazine-like warmth over industrial coldness |
| 4 | Machinery visibility | Editor who codes | Personal tool needs full depth; editorial feel doesn't mean hidden machinery |
| 5 | Typography | Serif headings + sans body | Strongest single change to kill AI-generated look; Playfair Display + Inter |
| 6 | Color palette | Stone & copper | Warmth differentiates from SaaS; terracotta accent bridges factory + editorial |
| 7 | Pipeline visualization | Vertical timeline | Best for showing full journey with expandable technical detail |
| 8 | Content rendering | Card-based with typesetting | Output is hero; drop caps, pull quotes, proper prose treatment |
| 9 | App shell | Redesigned sidebar with masthead | Editorial frame; minimal code change, maximum aesthetic impact |
| 10 | Dashboard | Newsroom overview | Active stories top, published below, killed collapsible |
| 11 | Job detail layout | Output-first | See the product first, the process second |
| 12 | Color mode | Light-first | Editorial = paper = light; dark mode secondary |
| 13 | Create job flow | Single refined form | Personal use = speed; editorial naming adds metaphor without adding steps |