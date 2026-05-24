# Content Factory — Design Source of Truth
> The visual, interaction, and architectural design document for the Content Factory UI.
> Every UI decision traces back to this file. When in doubt, read this first.
---
## 1. Design Philosophy
### Core Identity
Content Factory is a **personal editorial command center** — not a SaaS dashboard, not an admin panel, not an AI playground. It is the tool of an editor who codes: someone who commissions content, monitors the editorial process, reviews proofs, and publishes — but also wants full visibility into the multi-agent machinery underneath.

The UI serves two mental modes that the user switches between constantly:
- **Editor Mode** — "what does the output look like, is it good?" The editorial skin, rendered content, narrative flow.
- **Operator Mode** — "why did stage X fail, what did the agent actually do, what's the pipeline state?" The audit trail, raw state enums, polling heartbeat, claim evidence.

Both modes are always accessible, never hidden behind a toggle. The UI surfaces the right mode's information by default based on pipeline state, but one click flips to the other.

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
1. **State-aware defaults, not output-first.** The default view reacts to pipeline state: active jobs show the trail, `HUMAN_REVIEW_NEEDED` shows review, completed jobs show output. Raw data is a tab away, never hidden.
2. **Process as narrative.** The pipeline journey tells a story — research → writing → fact-checking → formatting. The UI reads like an editorial audit trail, not a database dump.
3. **Warmth over sterility.** Warm grays, serif headings, paper-like backgrounds. The UI should feel like a well-designed magazine, not a SaaS tool.
4. **Craft, not generated.** Every element should feel deliberately placed. No default spacings, no unstyled text, no raw JSON visible by default. If it looks like an AI made it, it's wrong.
5. **Full depth on demand.** Technical detail (agent traces, JSON payloads, revision diffs, claim evidence) is always one click away — in collapsible sections, tooltips, or expandable panels.
6. **Operator as first-class citizen.** The user may be debugging at midnight. Raw state names are a hover away, polling has a visible heartbeat, and the shell surfaces counts of things needing attention, not just page names.
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
│  Stories (12)            │     Active: copper text + copper left border
│  Commission              │     Hover: accent background
│                          │     Inactive: muted-foreground
│                          │     Badge count in muted-foreground parentheses
│                          │     Warning dot + count on Stories when review-needed > 0
│                          │
│                          │
│                          │  ← Generous empty space. Not packed.
│                          │
│                          │
│                          │
│                          │  [🌙]  ← Dark mode toggle, header area of sidebar
│                          │
│  v1.0                    │  ← Version label, bottom, subtle
└──────────────────────────┘
```
- **Width:** 240px (w-60), fixed position, warm stone background (`var(--background)`)
- **Right border:** 1px solid `var(--border)` — no heavy separation
- **No icons.** Editorial design is text-driven. Navigation is pure text.
- **Badge counts:** Story count in parentheses, muted-foreground. When `HUMAN_REVIEW_NEEDED > 0`, a warning-colored dot and count appear on the Stories item (e.g. "Stories (12) ⚑ 2").
- **Collapsible:** collapses to 0 width on mobile/toggle. No slide animation needed.

### Status Bar
```
┌─────────────────────────────────────────────────────────────┐
│  Last updated: 3s ago              ● Live                   │
└─────────────────────────────────────────────────────────────┘
```
A persistent operator feedback element fixed at the **bottom** of the viewport. Always visible, never competes with page content.

- **Height:** 32px, full width
- **Background:** `var(--muted)`, text `var(--muted-foreground)`
- **Font:** Inter 0.75rem, 500
- **Content:** "Last updated: {relative time}" on the left, a pulsing dot + "Live" on the right
- **Live indicator:** 8px dot in `var(--success)` that fades opacity (0.5 → 1.0, 2s cycle) while polling is active. Turns `var(--warning)` with label "Stalled" if no poll response in >15s. Turns `var(--destructive)` with label "Disconnected" if fetch fails.
- **Freshness scope:** Reflects the page currently in view (job list polling on Stories, job detail polling on a job). Not an app-wide aggregate.
- **Tooltip on hover:** Shows exact last-updated timestamp and next poll ETA.
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
**Concept: The editor's desk. What needs your attention right now.**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Overview                                       [Commission]│
│                                                             │
│  ┌─ Needs Attention ─(copper dashed border)──────────────┐ │
│  │                                                        │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  AI Regulation Enforcement Loopholes              │  │ │
│  │  │  Your Review · Waiting 45 min                     │  │ │
│  │  │  [Research ✓] [Writing ✓] [Fact-Ch ☒] [Your Rev ●]│  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │                                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ── Active Stories ──────────────────────────────────────── │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  BRICS De-dollarization 2025                         │   │
│  │  Writer's Desk · Started 12 min ago                  │   │
│  │  ● ● ○ ○ ○ ○ ○                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Quantum Computing Breakthroughs                     │   │
│  │  Fact-Check Desk · Started 28 min ago                │   │
│  │  ● ● ● ● ○ ○ ○                                      │   │
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
**Structure — four sections, strict order:**
1. **Needs Attention** — Jobs in `HUMAN_REVIEW_NEEDED`. Only renders when count > 0. No empty state. Dashed copper border (`border: 1px dashed var(--primary)`), section header in Playfair Display with `var(--warning)` text. Cards use the same layout as Active Stories.
2. **Active Stories** — Jobs in non-terminal states. Each card shows:
   - Topic as card title (serif)
   - Current desk name + elapsed time
   - Mini pipeline progress (dot-based, inline — see 5.6)
   - Click navigates to job detail
3. **Recently Published** — Completed jobs. Cards show:
   - Topic + format badges
   - Relative timestamp
   - Click navigates to job detail
4. **Killed Stories** — Failed jobs. Collapsed by default, expandable.
   - Topic + "Failed" indicator
   - Error summary on expand
**Empty state:** Single centered card with editorial copy: "No stories yet. Commission your first piece." + copper "Commission" CTA.
**No stat cards.** Stats are small counters inline in section headers ("Active Stories (3)", "Recently Published (12)").
### 4.2 Jobs List — "Stories"
**Concept: The assignment board.**
Same card-list pattern as Dashboard, but with filtering, full history, and higher density. Mini pipeline is **hover-only** (tooltip) rather than inline — keeps the list scannable at a glance.
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
**Concept: Opening a story file. Three tabs — Output, Trail, Review — always visible. Default tab is state-aware.**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  BRICS De-dollarization 2025                                │
│  Job #a3f2... · Commissioned Oct 13, 2026                   │
│  [Blog] [Carousel] [Active]                                 │
│  ┌─[Copy]────────────────────────────────────────┐          │  ← Copy-to-clipboard for output
│  │                                                 │          │
│  │  [Output] [Trail] [Review (0)]                  │          │  ← Three tabs, always visible
│  │                                                 │          │
│  │  ┌─ Blog ──── Carousel ──── Video Script ──┐  │          │
│  │  │                                           │  │          │
│  │  │  ┌───────────────────────────────────┐   │  │          │
│  │  │  │  Drop cap  he rapid shift in...    │   │  │          │
│  │  │  │                                   │   │  │          │
│  │  │  │  Key Takeaway                     │   │  │          │
│  │  │  │  ┃ "Central banks are...          │   │  │          │
│  │  │  └───────────────────────────────────┘   │  │          │
│  │  └───────────────────────────────────────────┘  │          │
│  └─────────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Tab bar** (top of content area, below job header):
- Three tabs: **Output** / **Trail** / **Review (N)**
- Playfair Display 1rem, `var(--muted-foreground)` inactive, `var(--foreground)` + copper 2px underline active
- Review tab always shows count: "Review (0)" in `var(--muted-foreground)` when none needed, "Review (N)" with `var(--warning)` dot when `HUMAN_REVIEW_NEEDED`
- When review is needed, the Review tab title uses `var(--warning)` text color

**State-aware default tab:**

| Job Status | Default Tab | Rationale |
|---|---|---|
| All active states (PENDING → ASSET_GENERATION) | **Trail** | You're monitoring progress — the trail shows where it is and what's happening |
| `HUMAN_REVIEW_NEEDED` | **Review** | Interrupt — action required immediately |
| `COMPLETED` | **Output** | You're here for the result |
| `FAILED` | **Trail** (showing error node) | You need to see what broke |

User can always switch tabs freely. Default is only on initial load of the job detail page.

**Output tab** — the rendered output content:
For completed jobs: fully rendered format outputs in sub-tabs (Blog / Carousel / Video Script). Blog sections get drop caps, proper paragraphs, pull quotes for key takeaways. Carousel slides get numbered cards with visual prompts. Video scenes get a storyboard grid with narration, visual, and audio cues side by side.
For active jobs: renders the current desk's live output with a subtle pulsing copper indicator. API data per stage same as original spec:

| Active Stage | What Renders in Output Tab | API Data Source |
|---|---|---|
| `PENDING` | Assignment summary card — topic, format, platform. No output yet. | `topic`, `format_type`, `platform` |
| `RESEARCHING` / `FACT_CHECKING_RESEARCH` | Research summary card — `refined_context` rendered as prose, with "Research in progress..." indicator if `refined_context` is still null | `refined_context` |
| `SCRIPTING` | Current script draft — latest `script.content` rendered as prose. If multiple revisions, show "Draft v{version}" label. | `scripts[latest].content`, `scripts[latest].version` |
| `FACT_CHECKING_SCRIPT` | Current script + live claim evaluation — script rendered as above, plus partial claim cards as they arrive via polling | `scripts[latest].content` + `scripts[latest].claims[]` |
| `FORMATTING` | Format outputs as they populate — sub-tabs show "Generating..." placeholder until `format_payload` arrives | `scripts[]` where `format_type != "VIDEO"` → `format_payload` |
| `ASSET_GENERATION` | Video format output + asset cards showing generation status. Mock `s3://` URLs shown as "Pending render..." | `scripts[VIDEO].format_payload` + `assets[]` |

For failed jobs: replaced by a "Killed Story" error card with structured error display and "Technical Details" collapsible.

**Copy-to-clipboard:** A copper copy icon in the top-right corner of each output section. Click copies the text content of that section:
- Blog: full article body (excluding SEO metadata)
- Carousel: per-slide copy button + "Copy all slides" at top
- Video: per-scene narration copy + "Copy full script" at top
- Brief toast "Copied" (1.5s, muted, non-intrusive)

**Trail tab** — the editorial timeline:
Vertical timeline. Each node is a desk. Structure per node:
1. **Header:** Desk name (Playfair Display, 1rem), with raw enum as muted parenthetical on hover (Inter 0.6875rem, `var(--muted-foreground)` — e.g. "Writer's Desk" with "(SCRIPTING)" appearing on hover over the header)
2. **Status line:** Completion icon + duration + summary stat
3. **Output preview:** The key output from that stage — rendered, not raw
4. **Collapsible "Technical Details":** Agent used, model, temperature, raw JSON payload, token usage

Timeline styling from 5.7 (unchanged: dot + vertical connector).

**Input context in the trail:** The Research Desk node includes a "Source Materials" sub-section showing:
- `pre_context.source_urls` as a list of clickable links (Inter 0.75rem, copper color)
- `pre_context.raw_text` in a collapsible "Reference Text" block (collapsed by default, shown as first ~100 chars + "...")
- `pre_context.target_audience` and `pre_context.guardrail_strictness` as small muted labels

**Revision history in the trail:** The Writer's Desk node shows revision cycles when `feedback_history` is non-empty. Fully automatic — no human stop per cycle (the evaluator-optimizer loop runs without intervention):
- Each revision cycle is a nested card: "Revision 2 → Revision 3" with the `overall_reasoning` from `OptimizerFeedbackEntry`
- For structured feedback (`feedback_type === "structured_claims"`): show the list of `failed_claims` with their verdicts and evidence
- For legacy string feedback: show the raw feedback text in a muted blockquote
- The final script shown is always the latest version

**Assets in the trail:** The Production Studio node (only present for `format_type = "video"` or `"all"`) shows:
- Asset cards for each entry in `assets[]`: type badge (`VISUAL_VEO`, `AUDIO_LYRIA`, etc.), `url_or_path`, `render_meta` (start/end time, prompt used)
- Assets render as a grid of cards with the asset type as a small copper badge and the URL/path in monospace
- `render_meta.prompt_used` shown in a collapsible "Generation Prompt" section

Claim cards in the Fact-Check Desk node (same as original):
- Verdict badge (semantic color): SUPPORTED (green), CONTESTED (amber), UNSUPPORTED (red), UNCERTAIN (blue)
- Confidence percentage as a subtle bar
- Evidence sources linked to research chunks
- Category tag (statistic, attribution, chronological, causal, comparative)

**Review tab** — human review interface:
Always visible. When no review needed: muted state showing "No action needed. The pipeline will proceed automatically." with a subtle checkmark.
When `HUMAN_REVIEW_NEEDED`: a warm copper-bordered card with:
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
- **Badge counts:** Story count in parentheses after label, `var(--muted-foreground)`. When `HUMAN_REVIEW_NEEDED > 0`, a warning-colored dot and count appear on the Stories item (Inter 0.6875rem, `var(--warning)`). E.g. "Stories (12)" normally, "Stories (12) ⚑ 2" when review items exist.
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
7 dots for 7 meaningful stages (collapsing passthrough stages):

| Dot state | Color | Visual |
|---|---|---|
| Completed | `var(--success)` | Filled circle ● |
| Active | `var(--primary)` | Filled circle with subtle opacity pulse ● |
| Future | `var(--border)` | Hollow circle ○ |
| Skipped | `var(--muted)` | Tiny dot (reduced size) |

**Hybrid display strategy:**
- **Dashboard (Overview) cards:** Inline, always visible below the desk name line. 7 dots in a row, no labels. Hover tooltip shows desk names.
- **Stories list cards:** Hidden. Only visible on hover as a tooltip. Keeps the list scannable at high density.

**Tooltip content:** On hover over the dot row (dashboard) or card (stories), a thin tooltip shows: desk name for each completed/active step, raw state enum in muted parentheses. No tooltip on future steps.
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
- **Visual heartbeat:** The `StatusBar` (see 3.3) provides live feedback — pulsing dot when polling is active, warning when stalled, destructive when disconnected

### Copy / Export
- **Copy-to-clipboard:** Copper icon button in top-right corner of each output section. Click copies text content only (no HTML formatting). Brief "Copied" toast (1.5s, muted background, Inter 0.75rem).
- **Scope:** Blog → full article body. Carousel → per-slide + "Copy all slides" button. Video → per-scene narration + "Copy full script" button.
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
## 8. Dark Mode (Default)
Dark mode is the **default and primary identity**. The app loads in dark mode. Light mode exists for users who prefer paper-like reading during daylight. The editorial identity (typography, copper palette, warm tones) carries through in both modes — this is warm dark, not cold blue-black.
### Default Behavior
- On first load, respect `prefers-color-scheme` system preference. If the OS is in dark mode, the app is dark. If the OS is in light mode, the app is light.
- User toggle overrides system preference for that session (stored in localStorage).
- No automatic re-sync with system preference after manual toggle.

### Dark Palette
| Token | Dark Mode |
|---|---|
| `background` | `oklch(0.210 0.008 84.6)` |
| `foreground` | `oklch(0.965 0.006 84.6)` |
| `card` | `oklch(0.255 0.006 56.1)` |
| `muted` | `oklch(0.299 0.008 75.3)` |
| `border` | `oklch(0.347 0.009 67.5)` |
| `primary` | `oklch(0.656 0.133 40.1)` (slightly lighter copper for contrast) |
| `accent` | `oklch(0.266 0.012 55.8)` |

Typography unchanged. Serif headings remain. The warm tones carry through — this is warm dark, not cold dark. Surface hierarchy same as light: `background < card < muted < border`.

### Light Palette (Secondary)
Same Stone & Copper palette from Section 2 (reproduced here for completeness):
| Token | Light Mode |
|---|---|
| `background` | `oklch(0.965 0.006 84.6)` |
| `foreground` | `oklch(0.228 0.009 75.2)` |
| `card` | `oklch(0.994 0.006 84.6)` |
| `muted` | `oklch(0.920 0.010 81.8)` |
| `border` | `oklch(0.860 0.015 80.7)` |
| `primary` | `oklch(0.599 0.140 37.4)` |
| `accent` | `oklch(0.931 0.014 57.6)` |

### Toggle
Toggle button in the sidebar header area, below the masthead. Simple moon/sun icon. No label text needed.
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
| `jobs/[id]/page.tsx` | Vertical card stack | Tabbed layout (Output / Trail / Review) with state-aware defaults, copy-to-clipboard |
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
| `MiniPipeline` | Dot-based progress for job cards (hybrid: inline on dashboard, hover on stories) |
| `StatusBar` | Persistent polling heartbeat bar fixed at bottom of viewport |
| `TabBar` | Editorial-styled tab bar with state-aware defaults and urgency dot |
| `CopyButton` | Copper copy-to-clipboard icon with "Copied" toast |
### Files That Don't Change
- All API hooks (`use-jobs.ts`) — data layer unchanged
- API client (`api-client.ts`) — unchanged
- Zustand store (`ui-store.ts`) — update `theme` toggle logic (dark default, system preference), rest unchanged
- All backend code — this is purely a frontend redesign
### Migration Strategy
1. **Phase 1: Palette + Typography** — Update `globals.css` color tokens + `@theme inline` block. Replace Geist font imports in `layout.tsx` with `next/font/google` (Playfair Display, Inter, JetBrains Mono). Wire `--font-heading` to Playfair Display, `--font-sans` to Inter, `--font-mono` to JetBrains Mono. Instant visual shift across the entire app.
2. **Phase 2: Shell** — Redesign sidebar with masthead + badge counts. Add `StatusBar` component at bottom. Implement dark-mode-as-default with system preference detection.
3. **Phase 3: Dashboard** — Add "Needs Attention" section above Active Stories. Hybrid mini pipeline (inline on dashboard, hover on stories).
4. **Phase 4: Job Detail** — Tabbed layout (Output / Trail / Review) with state-aware defaults. `TabBar` component. Copy-to-clipboard buttons. Enum on hover in trail timeline.
5. **Phase 5: Commission Form** — Editorial labeling and refined layout.
6. **Phase 6: Polish** — Empty states, micro-interactions, claim card styling.
---
## 10. Decision Log
Decisions made during the design process, recorded for traceability. Superseded entries are struck through with a reference to the replacement.

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Doc scope | Combined design + UX + architecture | Pain point is disconnected UI from app essence; need full picture |
| 2 | Audience | Personal use, power user | Simplifies persona work; no non-technical users to accommodate |
| 3 | Feel/direction | Organic/Editorial | Chosen by owner; magazine-like warmth over industrial coldness |
| 4 | Machinery visibility | Editor who codes | Personal tool needs full depth; editorial feel doesn't mean hidden machinery |
| 5 | Typography | Serif headings + sans body | Strongest single change to kill AI-generated look; Playfair Display + Inter |
| 6 | Color palette | Stone & copper | Warmth differentiates from SaaS; terracotta accent bridges factory + editorial |
| 7 | Pipeline visualization | Vertical timeline | Best for showing full journey with expandable technical detail |
| 8 | Content rendering | Card-based with typesetting | Published output gets proper editorial typesetting |
| 9 | App shell | Redesigned sidebar with masthead | Editorial frame; minimal code change, maximum aesthetic impact |
| 10 | Dashboard | Newsroom overview with Needs Attention section | Active stories top, published below, killed collapsible. Needs Attention pulls HUMAN_REVIEW_NEEDED to the top as an interrupt |
| ~~11~~ | ~~Job detail layout~~ | ~~Output-first~~ | ~~See the product first, the process second~~ → **Superseded by #15** |
| ~~12~~ | ~~Color mode~~ | ~~Light-first~~ | ~~Editorial = paper = light; dark mode secondary~~ → **Superseded by #16** |
| 13 | Create job flow | Single refined form | Personal use = speed; editorial naming adds metaphor without adding steps |
| 14 | Sidebar density | Badge counts on nav items | Operator needs glanceable counts of things needing attention |
| 15 | Job detail layout | Tabbed (Output / Trail / Review) with state-aware defaults | Operator mode jumps between output, trail, and review — tabs make all three first-class. Default tab reacts to pipeline state |
| 16 | Color mode | Dark-first with system preference detection | Developer's personal tool: most operators monitor in dark mode. Warm tones carry through in both. Light is secondary |
| 17 | Polling feedback | Persistent StatusBar at bottom of viewport | Page-level polling heartbeat — live, stalled, or disconnected at a glance |
| 18 | Mini pipeline | Hybrid: inline on Dashboard, hover-only on Stories | Dashboard has ≤5 active cards (dots make sense inline). Stories may have 20+ (hover avoids visual noise) |
| 19 | Raw state visibility | Enum in muted parenthetical on hover in trail | Operator debugging needs enum; editorial mode doesn't want clutter. Hover reveals it |
| 20 | Copy / export | Per-section copy-to-clipboard with toast | Output is the product — copying should be one click |
| 21 | Revision loop | Fully automatic, no human stop per cycle | The evaluator-optimizer loop is the architecture's core self-healing property. HUMAN_REVIEW_NEEDED is the pressure valve |