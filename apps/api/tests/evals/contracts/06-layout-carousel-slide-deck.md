# Layout Desk — eval contract (6.2 Carousel Slide Deck)

## What this eval guards
Carousel slide decks must be structurally valid: correctly numbered slides
within platform limits, with visual descriptions, valid hook types, cited
sources, and diverse hooks. Carousel decks must never be stored as format
outputs.

## Assertions
- Slide count: ≥ 3 and ≤ platform_max slides (100%)
- `slide_number` ordinal integrity: numbered 1..N with no gaps (100%)
- `text` within character limit: each slide ≤ platform limit (100%)
- `visual_description` present: non-null, non-empty on every slide (100%)
- `hook_type` valid: each slide's hook_type is a valid enum value (100%)
- `sources_used` populated: ≥ 80% of slides cite ≥ 1 chunk UUID
- Hook diversity: ≥ 3 distinct hook_type values across deck
- `visual_description` quality: LLM judge ≥ 3.5/5 for useful specific
  visual description
- Schema isolation: carousel_slide_deck objects never stored in the
  `format_outputs` table and vice versa (100%)

## Failure modes covered
F-O1, F-O2

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 6.2
