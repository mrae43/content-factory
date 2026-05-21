# Layout Desk — eval contract (6.1 Format Output — Blog Quality)

## What this eval guards
Blog outputs must have complete structure (sections, SEO metadata, CTA)
and be publish-ready while faithfully representing the master script.

## Assertions
- Sections present: `format_output.sections` is non-empty (100%)
- SEO metadata present: `title`, `meta_description`, `tags` all non-null (100%)
- CTA present: `call_to_action` is non-null and non-empty (100%)
- Publish-readiness: LLM judge ≥ 4.0/5
- Master fidelity: LLM judge ≥ 4.0/5 for faithful representation of master
  script's key points

## Failure modes covered
F-O1 (type confusion — blog stored as carousel or vice versa)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 6.1
