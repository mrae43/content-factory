# Writer's Desk — eval contract (3.3 Format Script Derivation)

## What this eval guards
When a format script is derived from the master, it must faithfully represent
the master's key points while appropriately adapting to its target format
(e.g. blog). Schema fields must be correctly set.

## Assertions
- Content fidelity: LLM judge ≥ 4.0/5 that format script faithfully represents
  the master's key points
- Format adaptation: LLM judge ≥ 4.0/5 that script is appropriately adapted
  for its target format
- Role set correctly: `script.role == "format"` (100%)
- Version incremented: `script.version > master_script.version` (100%)

## Failure modes covered
F-W1 (directive drift in format adaptation), F-O1 (type confusion)

## Last reviewed
2026-05-21 — derived from evals-criteria.md Eval 3.3
