# Layout Desk — eval contract (6.3 Platform Compliance)

## What this eval guards
Each supported platform's character limits must be enforced on carousel
slide text. A story targeting YouTube must have its slide text limited
just as Instagram or TikTok stories do.

## Assertions
- Instagram: `text` ≤ platform char limit for `platform=INSTAGRAM` stories
- TikTok: `text` ≤ platform char limit for `platform=TIKTOK` stories
- Twitter/X: `text` ≤ platform char limit for `platform=TWITTER` stories
- LinkedIn: `text` ≤ platform char limit for `platform=LINKEDIN` stories
- YouTube: `text` ≤ platform char limit for `platform=YOUTUBE` stories

## Failure modes covered
F-O2

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 6.3
