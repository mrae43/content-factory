# End-to-End Integration — eval contract (8.2 Directive Propagation)

## What this eval guards
Story Directives (tone, audience, angle) must be present in the prompts
of desks that need editorial awareness (Retrieval, Writer, Layout) and
absent from desks that must remain neutral (Research, Fact-Check).

## Assertions
- Story Directives present in prompts of: Retrieval Desk (Synthesis step),
  Writer's Desk (Scripting step), Layout Desk (Formatting step)
- Story Directives absent from prompts of: Research Desk (Indexing),
  Fact-Check Desk (Red Team)

## Failure modes covered
F-W1 (directive drift from missing propagation), F-F1 (Fact-Check
receiving editorial bias)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 8.2
