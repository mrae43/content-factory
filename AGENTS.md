# Content Factory — Agent Guide

Multi-agent AI pipeline generating short/reel scripts, blog articles, and social carousels for high-stakes domains (politics, macro-economics, history). 9-step async pipeline with a Red Team revision loop, fact-check persistence to pgvector, and guardrail-driven escalation.

## Toolchain

| Command | What |
|---------|------|
| `pnpm dev` | Nx parallel dev servers |
| `pnpm lint` | nx run-many -t lint |
| `nx test:unit api` | `pytest -m unit` (fast, needs pgvector) |
| `nx test:agent api` | `pytest -m agent` (mocked LLM/DB — no external services) |
| `nx test:integration api` | `pytest -m integration` (CI-only) |
| `nx test api -- -m eval` | Eval with LLM-as-Judge (default: golden mode, no API calls) |
| `nx test api -- -m eval -- --live` | Eval with real LLM calls |
| `nx migrate api` | `alembic upgrade head` |
| `nx generate-types api` | `scripts/generate_ts_types.py` |
| `ruff format . && ruff check . --fix` | Python lint+format (line-length=88) |
| `uv sync --extra test` | Install Python deps + test extras (from `apps/api/`) |
| `nx typecheck web` | `tsc --noEmit` |

**Quick checks:** `nx test:unit api && nx test:agent api && nx lint api && nx typecheck web`

## Architecture

- **Two apps:** `apps/api` (FastAPI) + `apps/web` (Next.js 16 App Router). One auto-generated types lib at `libs/shared-types/`.
- **Queue:** asyncio `QueueWorker` with `FOR UPDATE SKIP LOCKED` — no Celery/Redis. Runs inside FastAPI lifespan.
- **LLM routing:** model name starting with `gemini` → Google GenAI SDK; everything else → Together AI (OpenAI-compatible). Default production models are `meta-llama/Llama-3.3-70B-Instruct-Turbo` via Together AI for all agents. Embeddings always use `models/gemini-embedding-001` (768-dim, cosine).
- **Model config (env vars):** `{research,copywriter,evaluator,optimizer,asset,formatter}_{model,temperature}` — all default to Llama-3.3-70B at varying temps. Eval models have `eval_` prefix (e.g. `eval_judge_model`, `eval_red_team_model`).
- **DB:** PostgreSQL + pgvector, `factory` schema, pool size 20.
- **Migrations:** Alembic (sync, psycopg2). `env.py` auto-replaces `asyncpg` with `psycopg2` in URL. Auto-runs on container start via `entrypoint.sh`.
- **Generated code:** `libs/shared-types/src/types/api.ts` from Pydantic via OpenAPI. Run `nx generate-types api` after schema changes.
- **Evaluator-Optimizer pattern:** Red Team extracts atomic claims → cross-references vector store → verdicts persisted to `fact_check_claims`. Failed claims → `ScriptOptimizerAgent` surgical patching (max 3 revisions → `HUMAN_REVIEW_NEEDED`).

## Model Config (env vars)

All production agents default to `meta-llama/Llama-3.3-70B-Instruct-Turbo` via Together AI. Override per-agent:

| Env var | Default | Purpose |
|---------|---------|---------|
| `research_model` | Llama-3.3-70B | Deep research agent |
| `copywriter_model` | Llama-3.3-70B | Script drafting |
| `evaluator_model` | Llama-3.3-70B | Red Team (temp 0.0) |
| `optimizer_model` | Llama-3.3-70B | Surgical script patching |
| `formatter_model` | Llama-3.3-70B | Blog/carousel formatting |

Eval suite uses separate `eval_*` vars — see `app/core/config.py:49`.

## Testing

- **pytest markers** (from `pyproject.toml`): `unit`, `agent`, `eval`, `golden`, `integration`.
- **Default skips eval:** `addopts = "-m 'not eval'"`. Run with `-m eval`.
- **Eval golden mode** (default): uses pre-recorded `reference_outputs` — deterministic, no API calls. `--live` flag for real LLM calls.
- **Agent tests** mock LLM and DB — no external services or pgvector needed.
- **Unit tests** need a running pgvector instance.
- **async def test** required for all tests (`asyncio_mode = "auto"`).

## Conventions

- **Python:** Ruff lint/format (line-length 88). No `print()` in production.
- **TypeScript:** ESLint with `@nx/enforce-module-boundaries` — apps can only import specific lib types.
- **API schema changes:** always run `nx generate-types api` to regenerate TS types.

## Gotchas

- **Postgres port:** Docker maps to `127.0.0.1:5433`, not 5432. pgAdmin connects to internal hostname `db:5432`.
- **`GEMINI_API_KEY` is mandatory** (used for embeddings even when runtime LLM is Together AI). `TOGETHER_API_KEY` is optional but needed if using default models.
- **Type generation** needs `GEMINI_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL` in env (skips gracefully if missing).
- **pnpm 11:** `shamefully-hoist=true`, `node-linker=hoisted`. Build allowlist in `pnpm-workspace.yaml`.
- **Hot reload:** `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d`.
- **Chunking:** `MarkdownTextSplitter` with 2000 char chunks, 400 overlap.
- **Design system:** See `DESIGN.md` for the UI design source of truth — editorial theme (Stone & Copper palette, Playfair Display + Inter + JetBrains Mono typography), output-first page layouts, and the complete component migration plan.

## MCP — Context7 (Library Docs)

Use Context7 MCP whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service — even well-known ones like React, Next.js, Prisma, Express, Tailwind, Django, or Spring Boot. This includes API syntax, configuration, version migration, library-specific debugging, setup instructions, and CLI tool usage. Use even when you think you know the answer — training data may not reflect recent changes. Prefer this over web search for library docs.

**Do not use for:** refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

### Steps

1. Always start with `resolve-library-id` using the library name and the user's question, unless the user provides an exact library ID in `/org/project` format
2. Pick the best match (ID format: `/org/project`) by: exact name match, description relevance, code snippet count, source reputation (High/Medium preferred), and benchmark score (higher is better). If results don't look right, try alternate names or queries (e.g., "next.js" not "nextjs", or rephrase the question). Use version-specific IDs when the user mentions a version
3. `query-docs` with the selected library ID and the user's full question (not single words)
4. Answer using the fetched docs
