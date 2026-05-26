# Content Factory — Agent Guide

Multi-agent AI pipeline generating short/reel scripts, blog articles, and social carousels for high-stakes domains (politics, macro-economics, history). Async pipeline with 11-state state machine, Red Team revision loop, fact-check persistence to pgvector, guardrail-driven escalation, and human review endpoints.

## Toolchain

| Command | What |
|---------|------|
| `pnpm dev` | Nx parallel dev servers |
| `pnpm lint` | nx run-many -t lint |
| `nx test:unit api` | `uv run pytest -m unit` |
| `nx test:agent api` | `uv run pytest -m agent` |
| `nx test:integration api` | `uv run pytest -m integration` |
| `nx test api -- -m eval` | Eval with LLM-as-Judge (default: golden mode, no API calls) |
| `nx test api -- -m eval -- --live` | Eval with real LLM calls |
| `nx migrate api` | `alembic upgrade head` |
| `uv sync --extra test` | Install Python deps + test extras (from `apps/api/`) |
| `nx typecheck web` | `tsc --noEmit` |
| `apps/api/clean_code.ps1` | PowerShell ruff lint+format (line-length=88) |

**Quick checks:** `nx test:unit api && nx test:agent api && nx lint api && nx typecheck web`

## Architecture

- **Two apps:** `apps/api` (FastAPI) + `apps/web` (Next.js 16 App Router, React 19). One auto-generated types lib at `libs/shared-types/`.
- **Queue:** asyncio `QueueWorker` with `FOR UPDATE SKIP LOCKED` — no Celery/Redis. Runs inside FastAPI lifespan. Polls every 5s, locks for 15 min.
- **LLM routing:** Config-driven provider registry in `app/services/llm.py`. Model names use `provider:model` prefix convention (e.g. `together:meta-llama/...`). Bare names default to Together AI; bare names starting with `gemini` resolve to Google GenAI SDK for backward compatibility. New providers are registered in the `PROVIDERS` dict. Embeddings use `models/gemini-embedding-001` (768-dim, cosine, L2-normalized — see ADR 0003).
- **DB:** PostgreSQL 16 + pgvector, `factory` schema, pool size 20.
- **Migrations:** Alembic (sync, psycopg2). `env.py` auto-replaces `asyncpg` with `psycopg2` in URL. Auto-runs on container start via `entrypoint.sh`.
- **Generated code:** `libs/shared-types/src/types/api.ts` from Pydantic via OpenAPI. Run `nx generate-types api` after schema changes.
- **Nx targets** use `uv run` for all Python commands.

## Pipeline States

11 states defined in `app/db/models.py`:

| State | Editorial Desk | What happens |
|-------|---------------|--------------|
| `PENDING` | Assignment Queue | Chunk raw text → ingest to vector store (scope `RAW-CONTEXT`) |
| `RESEARCHING` | Research Desk | Tavily web search → ingest to vector store (scope `LOCAL`) |
| `RETRIEVAL` | Deep Research | `ResearchAgent` → `refined_context` + `citation_index` → `context_builder` → `AssembledContext` |
| `FACT_CHECKING_RESEARCH` | Source Verification | Legacy state, auto-forwarded to `SCRIPTING` |
| `SCRIPTING` | Writer's Desk | `CopywriterAgent` drafts script (or `ScriptOptimizerAgent` for revisions) |
| `FACT_CHECKING_SCRIPT` | Fact-Check Desk | `RedTeamAgent` extracts atomic claims → per-claim evidence search → structured verdict |
| `FORMATTING` | Layout Desk | `FormatterHarness` runs blog/carousel/video formatters in parallel |
| `ASSET_GENERATION` | Production Studio | `AssetStudioAgent` (video prompts) + `CarouselImageAgent` (images via FLUX) |
| `COMPLETED` | Published | Garbage-collect LOCAL-scoped chunks, finalize |
| `FAILED` | Killed Story | Orchestrator exception → logged |
| `HUMAN_REVIEW_NEEDED` | Editor's Review | High-strictness pass, max revisions exhausted, or escalation |

## Model Config (env vars)

All production agents default via Together AI. Two tiers:

**Premium tier** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`):

| Env var | Default | Temp | Purpose |
|---------|---------|------|---------|
| `copywriter_model` | Llama-3.3-70B | 0.7 | Script drafting |
| `evaluator_model` | Llama-3.3-70B | 0.0 | Red Team fact-check |

**Standard tier** (`openai/gpt-oss-20b`):

| Env var | Default | Temp | Purpose |
|---------|---------|------|---------|
| `optimizer_model` | gpt-oss-20b | 0.3 | Surgical script patching |
| `asset_model` | gpt-oss-20b | 0.5 | Asset studio (video/audio prompts) |
| `formatter_model` | gpt-oss-20b | 0.3 | Blog/carousel/video formatting |

**Image generation:** `black-forest-labs/FLUX.1-schnell` via Together AI. Platform-specific dimensions. Storage via `StorageAdapter` (default: `s3` → SeaweedFS, fallback `local` → `static/carousel_images/`).

**Eval models** (separate from production):

| Env var | Default | Temp |
|---------|---------|------|
| `eval_copywriter_model` | `MiniMaxAI/MiniMax-M2.7` | 0.7 |
| `eval_red_team_model` | `openai/gpt-oss-120b` | 0.0 |
| `eval_optimizer_model` | `openai/gpt-oss-20b` | 0.3 |
| `eval_judge_model` | `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` | 0.0 |

**Other settings:** `max_red_team_revisions=3`, `similarity_threshold=0.75`, `retrieval_retry_max=3`, `context_builder_top_k=10`, `synthid_watermark_enabled=True`.

## Agents

| Agent | File | Role |
|-------|------|------|
| `ResearchAgent` | `app/services/agents.py` | Deep research → structured context + citation index |
| `CopywriterAgent` | `app/services/agents.py` | Draft narrative script with hook/body/closer |
| `RedTeamAgent` | `app/services/agents.py` | 3-pass: claim extraction → evidence retrieval → structured verdict |
| `ScriptOptimizerAgent` | `app/services/optimizer.py` | Surgical patching of unsupported/contested claims |
| `BlogFormatterAgent` | `app/services/formatters.py` | Plan→Execute blog layout |
| `CarouselFormatterAgent` | `app/services/formatters.py` | Plan→Execute carousel with platform char limits |
| `VideoFormatterAgent` | `app/services/formatters.py` | Plan→Execute with scene structure |
| `AssetStudioAgent` | `app/services/agents.py` | Veo video prompts + Lyria audio prompts |
| `CarouselImageAgent` | `app/services/carousel_image_agent.py` | Generate carousel images via FLUX + Together AI |
| `FormatterHarness` | `app/services/harness.py` | Retry wrapper with validation and doom-loop detection |

**All agents** extend `BaseAgent` which provides tenacity retry (3 attempts, exponential backoff).

## Evaluator-Optimizer Pattern

1. **RedTeamAgent** (3-pass): extract atomic claims → per-claim `semantic_search` → structured `RedTeamVerdict` per claim
2. Verdicts persisted to `fact_check_claims` table (versioned per script)
3. On `REVISION_NEEDED` → feedback saved → `ScriptOptimizerAgent` patches only failed claims (preserves supported ones)
4. Max `max_red_team_revisions` loops → `HUMAN_REVIEW_NEEDED`

## Guardrails

Three profiles in `app/core/guardrails.py`:

| Profile | Threshold | Categories | Human Review |
|---------|-----------|------------|--------------|
| Low | 0.65 | statistic, attribution | No |
| Medium | 0.72 | + chronological, causal | No |
| High | 0.75 | + all categories | Yes (uncertain = soft fail) |

## Human Review

- `POST /api/v1/jobs/{id}/approve-script` → approve (proceeds to FORMATTING) or reject with feedback (back to SCRIPTING)
- `POST /api/v1/jobs/{id}/regenerate-assets` → regenerate carousel images
- Triggered by: High strictness pass, max red team revisions, or escalation (no evidence available)

## Testing

- **pytest markers** (from `pyproject.toml`): `unit`, `agent`, `eval`, `golden`, `integration`.
- **Default skips eval:** `addopts = "-m 'not eval'"`. Run with `-m eval`.
- **Eval golden mode** (default): uses pre-recorded `baselines.json` — deterministic, no API calls. `--live` flag for real LLM calls.
- **Agent tests** mock LLM and DB — no external services or pgvector needed.
- **Unit tests** need a running pgvector instance.
- **async def test** required for all tests (`asyncio_mode = "auto"`).

## Platform Support

| Platform | Format | Char limit | Carousel dimensions |
|----------|--------|-----------|-------------------|
| TikTok | video | 2200 | 1080×1920 |
| YouTube | video | 5000 | 1920×1080 |
| Instagram | carousel | 2200 | 1080×1350 |
| Twitter/X | carousel | 4000 | 1080×1620 |
| LinkedIn | carousel | 3000 | 1080×1350 |

Blog articles have no platform constraint.

## Provider Abstraction

The LLM layer in `app/services/llm.py` is provider-agnostic via a config-driven registry.

**Registering a new provider:**

1. Add an entry to the `PROVIDERS` dict:
   ```python
   "my_provider": {
       "class": MyLLMClass,
       "api_key_attr": "my_api_key",       # attr on settings
       "base_url": "https://api.example.com/v1",  # optional
   }
   ```
2. Add the corresponding env var to `Settings` in `app/core/config.py` (e.g. `my_api_key: str`).
3. Use the `provider:model` convention when setting model env vars, e.g. `MY_MODEL=my_provider:MyModel/Name`.

**Prefix convention:**

- `together:meta-llama/Llama-3.3-70B-Instruct-Turbo` → Together AI via ChatOpenAI
- `gemini-1.5-pro` → Google GenAI SDK (backward compat — no prefix needed for gemini)
- `meta-llama/Llama-3.3-70B-Instruct-Turbo` → defaults to Together AI (no prefix)

The current default model values are **ephemeral example config** — not hard-coded production decisions. Swap them via env vars without touching code.

**Embedding model** is independently configurable via `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` env vars. The default (`models/gemini-embedding-001`, 768-dim) is documented in ADR 0003.

## Conventions

- **Python:** Ruff lint/format via `apps/api/clean_code.ps1` (line-length 88). No `print()` in production.
- **TypeScript:** ESLint v9 with `@nx/enforce-module-boundaries` — apps can only import specific lib types.
- **API schema changes:** always run `nx generate-types api` to regenerate TS types.

## Gotchas

- **Postgres port:** Docker maps to `127.0.0.1:5433`, not 5432. pgAdmin connects to internal hostname `db:5432`.
- **`GEMINI_API_KEY` is mandatory** (used for embeddings even when runtime LLM is Together AI — see ADR 0003). `TOGETHER_API_KEY` is optional but needed if using default models.
- **Type generation** needs `GEMINI_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL` in env (skips gracefully if missing).
- **pnpm 11:** `shamefully-hoist=true`, `node-linker=hoisted`. Build allowlist in `pnpm-workspace.yaml`.
- **Hot reload:** `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d`.
- **Chunking:** `MarkdownTextSplitter` with 2000 char chunks, 400 overlap.
- **Design system:** See `DESIGN.md` for the UI design source of truth — editorial theme (Stone & Copper palette, Playfair Display + Inter + JetBrains Mono typography), output-first page layouts, and the complete component migration plan.
