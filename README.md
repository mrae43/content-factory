# Content Factory

Multi-agent system that generates multi-format content (Shorts/Reels/TikToks, blog articles, social carousels) for high-stakes domains — politics, macro-economics, historical analysis. Treats **Truth and Guardrails as first-class citizens** via a rigorous Red Team agentic loop that verifies claims against a vector database before any rendering occurs.

**Nx monorepo** with a Python FastAPI backend, Next.js frontend, and shared TypeScript types.

## Core Differentiators

- **Agentic Over Atomic** — Research, Copywriter, and Red Team agents debate and correct each other through structured revision loops.
- **Prompt Chaining with Semantic Memory** — ResearchAgent produces a condensed `refined_context` summary that the orchestrator persists and passes downstream. The CopywriterAgent works from this curated context instead of calling the vector store directly — eliminating context-window bloat, enabling auditable research summaries, and ensuring consistent behavior across revision loops.
- **Zero-Hallucination Guardrails** — Red Team breaks scripts into atomic claims, cross-references each against the vector store directly, and persists verdicts to Postgres. Claims that fail are sent back for revision (max 3 attempts before human escalation).
- **Governance-as-Code** — Full audit trail via `fact_check_claims` table with evidence references linked to source chunks. API returns the complete fact-check report alongside scripts and assets.
- **Web-Enriched RAG** — Tavily search enriches user-provided context with live web results, ingested as vector chunks for semantic retrieval by downstream agents.
- **Evaluator-Optimizer Pattern** — On revision, a dedicated `ScriptOptimizerAgent` surgically patches failed claims instead of re-drafting the entire script, preserving quality sections and reducing hallucination drift.
- **Multi-Format Output** — Pipeline branches by `format_type` after Red Team approval: `video` (asset generation), `blog` (structured articles with SEO metadata), `carousel` (platform-specific slide decks), or `all` (blog + carousel in parallel, then assets). `FormatterHarness` wraps each formatter with generate-validate-retry loops and doom loop detection.
- **Platform-Aware Validation** — `BlogValidator` and `CarouselValidator` enforce schema constraints and platform-specific character limits (Twitter 280, LinkedIn 700, Instagram 2200) before accepting output.

---

## The 9-Step Pipeline

A `RenderJob` flows through these state transitions asynchronously. After Red Team approval (Step 6), the pipeline branches by `format_type`:

- **`video`** → skips FORMATTING, goes straight to ASSET_GENERATION
- **`blog`** or **`carousel`** → FORMATTING → COMPLETED
- **`all`** (default) → FORMATTING (blog + carousel in parallel) → ASSET_GENERATION → COMPLETED

### 1. Ingestion (`PENDING`)
User submits a topic (e.g., *"BRICS De-dollarization 2025"*) along with pre-context (source URLs, raw text, audience target) via `POST /api/v1/jobs/`.

### 2. Extraction & Chunking
`MarkdownTextSplitter` chunks the raw text into `RAW-CONTEXT` scope vectors in the pgvector `research_chunks` table.

### 3. Deep Research (`RESEARCHING`)
Tavily web search enriches the topic with live results (ingested as `LOCAL`-scope vectors). The **Research Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, configurable) retrieves all chunks via semantic search, produces refined `LOCAL` chunks vetted for factual accuracy, **and synthesizes a `refined_context` summary** — a condensed, self-contained research brief persisted to the `render_jobs` table by the orchestrator.

### 4. Source Fact-Check (`FACT_CHECKING_RESEARCH`)
**MVP: Passthrough** — auto-advances to `SCRIPTING`. The Red Team at Step 6 catches issues downstream.

### 5. Script & Storyboard (`SCRIPTING`)
The **Copywriter Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, temp=0.7, configurable) receives the **`refined_context`** from the orchestrator (not the raw vector store). This curated context ensures a bounded, consistent input regardless of chunk count or embedding noise. The agent drafts a retention-optimized script + visual storyboard.

On revision (when Red Team rejects claims), the **Script Optimizer Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, temp=0.3, configurable) receives only the failed claims and patches them surgically — preserving the rest of the script.

### 6. Red Team Evaluation (`FACT_CHECKING_SCRIPT`)
The critical step. The **Red Team Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, temp=0.0, configurable) uses a three-pass evaluation with `.with_structured_output()`:

1. **Claim Extraction** — Breaks the script into atomic claims
2. **Evidence Retrieval** — Per-claim `semantic_search(query=claim.search_query, top_k=5)` against the vector store
3. **Verdict** — Evaluates each claim against enriched evidence

Results:
- **SUPPORTED** → Script passes, claims persisted to `fact_check_claims` table with evidence references. Pipeline branches based on `format_type`.
- **UNSUPPORTED/CONTESTED** → Script sent back to Step 5 with structured feedback. After 3 failures → `HUMAN_REVIEW_NEEDED`
- Human override available via `POST /api/v1/jobs/{id}/approve-script`

### 7. Format Output (`FORMATTING`)
**Skipped for `video` format.** For `blog`, `carousel`, or `all` formats, structured output agents run:

- **BlogFormatterAgent** — Two-phase LLM calls (plan outline → execute full output) producing structured blog sections with SEO metadata
- **CarouselFormatterAgent** — Two-phase LLM calls producing platform-specific slide decks with character-limit enforcement

Each formatter is wrapped in a **`FormatterHarness`** — a generate-validate-retry loop with doom loop detection (SHA-256 payload hashing). `BlogValidator` and `CarouselValidator` enforce schema constraints and platform rules. Max 2 retries (3 total attempts).

When `format_type = "all"`, both formatters run concurrently via `asyncio.gather()`.

### 8. Asset Generation (`ASSET_GENERATION`)
**MVP: Mocked** — The **Asset Studio Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, configurable) generates Veo/Lyria production prompts but returns a fake `s3://` URL. No real TTS, video rendering, or FFmpeg yet.

### 9. Completion (`COMPLETED`)
LOCAL-scope vector chunks are cleaned up. The final job state, scripts, audit trail, and asset metadata are available via the API.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `POST` | `/api/v1/jobs/` | Create a new RenderJob (returns `202 Accepted`) |
| `GET` | `/api/v1/jobs/{id}` | Poll job status with full scripts, claims audit, and assets |
| `POST` | `/api/v1/jobs/{id}/approve-script` | Approve or reject script (human-in-the-loop) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Monorepo | Nx workspace, pnpm 11 (with `allowBuilds` in `pnpm-workspace.yaml`) |
| Frontend | Next.js 16 (App Router), React Query, Zustand, shadcn, Tailwind CSS v4 |
| API | FastAPI (async, Pydantic V2), Python 3.11, uv |
| Database | PostgreSQL 16 + pgvector (HNSW index, `factory` schema) |
| ORM | SQLAlchemy 2 async (`asyncpg`) |
| Migrations | Alembic (sync via `psycopg2`) |
| AI Orchestration | LangChain + Google GenAI + Together AI (OpenAI-compatible) |
| Models | All default to `meta-llama/Llama-3.3-70B-Instruct-Turbo` via Together AI. Each agent stage configurable via `{research,copywriter,evaluator,optimizer,asset,formatter}_{model,temperature}` env vars. Eval suite uses separate `eval_*` models. Embeddings: `models/gemini-embedding-001` (Gemini). |
| Embeddings | `models/gemini-embedding-001` (768-dim, pgvector HNSW with cosine) |
| Web Search | Tavily (`langchain-tavily`) |
| Background Queue | `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` (no Celery/Redis) |
| Testing | pytest + pytest-asyncio + httpx + deepeval + LLM-as-Judge (Together AI) |
| CI/CD | GitHub Actions (lint → unit/agent tests → eval/integration/docker) |
| Containerization | Docker Compose (pgvector:pg16, pgAdmin4, API, Web) |
| Linter/Formatter | Ruff (Python), ESLint (TypeScript) |
| Design System | Stone & Copper palette, Playfair Display + Inter + JetBrains Mono — see [`DESIGN.md`](DESIGN.md) |

---

## Quick Start

```bash
# 1. Create .env with required variables (see Environment section)
cp .env.example .env

# 2. Start all services (db + pgadmin + api + web)
docker compose up -d

# 3. Migrations auto-run on container start via entrypoint.sh
# To generate new migrations or run manually:
docker compose exec api alembic revision --autogenerate -m "description"
docker compose exec api alembic upgrade head
```

### Development

```bash
# Start both dev servers locally (API on :8000, Web on :3000)
pnpm dev

# Start individually
pnpm dev:api                     # FastAPI backend only
pnpm dev:web                     # Next.js frontend only
```

### Build & Lint

```bash
pnpm build                       # Build all projects
pnpm lint                        # Lint all projects
pnpm lint:fix                    # Lint + auto-fix
```

### Docker Service Access

After `docker compose up -d`, services are available at:

| Service | URL | Notes |
|---------|-----|-------|
| Web (Next.js) | http://localhost:3000 | Frontend dashboard |
| API (FastAPI) | http://localhost:8000/docs | Swagger UI |
| pgAdmin | http://localhost:5050 | Login with `PGADMIN_EMAIL` / `PGADMIN_PASSWORD` |
| PostgreSQL | `127.0.0.1:5433` | Binary protocol only — use pgAdmin or a DB client, not a browser |

**pgAdmin DB connection:** When adding a server in pgAdmin, use Host `db` and Port `5432` (Docker internal), **not** `localhost`. pgAdmin and the database share the `factory_isolated_net` bridge network.

### Run Tests

```bash
# Backend tests (from apps/api/)
cd apps/api
uv sync --extra test             # Install deps including test extras
uv run pytest tests/ -v          # Run all tests

# Or via Docker
docker compose exec api pytest tests/ -v

# Run by marker
uv run pytest -m unit            # Unit tests only (9 files, ~100 tests)
uv run pytest -m agent           # Agent tests only (7 files, ~41 tests)
uv run pytest -m eval            # Eval benchmarks
uv run pytest -m golden          # Golden dataset validation
uv run pytest -m integration     # Integration tests (format branching, CI-only)
```

### Python Lint & Format

```bash
# From apps/api/
cd apps/api
ruff format . && ruff check . --fix
# Or on PowerShell:
./clean_code.ps1
```

---

## Environment

Required `.env` variables:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Mandatory — Google AI API key |
| `TAVILY_API_KEY` | Mandatory — Tavily web search API key |
| `TOGETHER_API_KEY` | Required — Together AI API key (all default production models route through Together AI) |
| `DATABASE_URL` | Async connection string, e.g. `postgresql+asyncpg://user:password@db:5432/content_factory` (Docker hostname `db` in container, `localhost` for local dev) |
| `POSTGRES_USER` | Docker Compose DB user |
| `POSTGRES_DB` | Docker Compose DB name |
| `POSTGRES_PORT` | Docker Compose host port (default `5433`) |
| `PGADMIN_EMAIL` | pgAdmin login email |
| `PGADMIN_PASSWORD` | pgAdmin login password |
| `API_PORT` | Docker Compose API host port (default `8000`) |

Optional `.env` overrides (all default to via Together AI unless `gemini-` prefixed):

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Research agent model |
| `RESEARCH_TEMPERATURE` | `0.2` | Research agent temperature |
| `COPYWRITER_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Copywriter agent model |
| `COPYWRITER_TEMPERATURE` | `0.7` | Copywriter agent temperature |
| `EVALUATOR_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Red Team agent model |
| `EVALUATOR_TEMPERATURE` | `0.0` | Red Team agent temperature |
| `OPTIMIZER_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Script Optimizer agent model |
| `OPTIMIZER_TEMPERATURE` | `0.3` | Script Optimizer agent temperature |
| `ASSET_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Asset Studio agent model |
| `ASSET_TEMPERATURE` | `0.5` | Asset Studio agent temperature |
| `FORMATTER_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Blog/Carousel formatter model |
| `FORMATTER_TEMPERATURE` | `0.3` | Blog/Carousel formatter temperature |
| `MAX_RED_TEAM_REVISIONS` | `3` | Max revision loops before human escalation |
| `SIMILARITY_THRESHOLD` | `0.75` | Vector search cosine similarity cutoff |
| `SYNTHID_WATERMARK_ENABLED` | `True` | SynthID flag (no implementation yet) |
| `WORKER_POLL_INTERVAL_SECONDS` | `5` | QueueWorker poll interval |
| `WORKER_LOCK_TIMEOUT_MINUTES` | `15` | Stuck job recovery timeout |
| `EVAL_RESEARCH_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Eval research agent (Together AI) |
| `EVAL_RESEARCH_TEMPERATURE` | `0.2` | Eval research temperature |
| `EVAL_COPYWRITER_MODEL` | `MiniMaxAI/MiniMax-M2.7` | Eval copywriter agent (Together AI) |
| `EVAL_COPYWRITER_TEMPERATURE` | `0.7` | Eval copywriter temperature |
| `EVAL_RED_TEAM_MODEL` | `openai/gpt-oss-120b` | Eval Red Team agent (Together AI) |
| `EVAL_RED_TEAM_TEMPERATURE` | `0.0` | Eval red team temperature |
| `EVAL_OPTIMIZER_MODEL` | `openai/gpt-oss-20b` | Eval optimizer agent (Together AI) |
| `EVAL_OPTIMIZER_TEMPERATURE` | `0.3` | Eval optimizer temperature |
| `EVAL_JUDGE_MODEL` | `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` | LLM-as-Judge model (Together AI) |
| `EVAL_JUDGE_TEMPERATURE` | `0.0` | Eval judge temperature |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend API base URL |
| `WEB_PORT` | `3000` | Docker Compose web host port |

---

## Project Structure

```
content-factory/                  # Nx workspace root
├── DESIGN.md                     # UI design source of truth — editorial theme, visual language, migration plan
├── apps/
│   ├── api/                      # Python FastAPI backend
│   │   ├── app/
│   │   │   main.py               # FastAPI app + lifespan (starts/stops QueueWorker)
│   │   │   core/config.py        # pydantic-settings, reads .env (database_url — required, eval model configs)
│   │   │   api/routes.py         # /api/v1/jobs/ endpoints + health check
│   │   │   db/
│   │   │     models.py           # SQLAlchemy models (factory schema)
│   │   │     session.py          # async engine + session factory (settings.database_url)
│   │   │     crud.py             # query helpers + queue operations
│   │   │   schemas/
│   │   │     shorts.py           # Pydantic request/response models + FormatTypeEnum, PlatformEnum
│   │   │     formats.py          # Structured format schemas (BlogSection, CarouselSlide, SeoMeta)
│   │   │   services/
│   │   │     llm.py              # Multi-provider LLM routing (Gemini + Together AI)
│   │   │     vector_store.py     # pgvector ingestion & semantic search
│   │   │     chunking.py         # Markdown text splitter
│   │   │     web_search.py       # TavilySearchService
│   │   │     format_validator.py # FormatValidator → BlogValidator, CarouselValidator
│   │   │   workers/
│   │   │     orchestrator.py     # Agentic state machine
│   │   │     queue_worker.py     # asyncio poll loop with SKIP LOCKED
│   │   │     agents.py           # BaseAgent → Research, Copywriter, RedTeam, AssetStudio
│   │   │     optimizer.py        # ScriptOptimizerAgent
│   │   │     formatters.py       # BlogFormatterAgent, CarouselFormatterAgent
│   │   │     harness.py          # FormatterHarness — generate-validate-retry with doom loop detection
│   │   │     tasks.py            # Post-completion LOCAL chunk cleanup
│   │   ├── alembic/              # Database migrations
│   │   ├── tests/                # Python test suite
│   │   ├── scripts/              # Type generation scripts
│   │   ├── pyproject.toml        # uv-managed Python deps
│   │   ├── uv.lock               # Lockfile for deterministic Python builds
│   │   ├── entrypoint.sh         # Auto-runs alembic upgrade head, then uvicorn
│   │   ├── Dockerfile
│   │   └── project.json          # Nx project config
│   └── web/                      # Next.js frontend (App Router)
│       ├── src/
│       │   ├── app/              # Pages: dashboard, jobs list, new job, job detail
│       │   ├── components/       # UI components (shadcn/ui + layout + jobs + script)
│       │   ├── hooks/            # React Query hooks (useJobs, useJob, useCreateJob)
│       │   ├── stores/           # Zustand store (sidebarOpen)
│       │   └── lib/              # API client, utilities
│       ├── Dockerfile
│       └── project.json          # Nx project config
├── libs/
│   └── shared-types/             # Auto-generated TS types from Pydantic schemas
├── pyproject.toml                # uv workspace root (member: apps/api)
├── uv.lock                       # Root uv lockfile
├── nx.json                       # Nx workspace config
├── package.json                  # Root package (Nx + dev deps)
├── pnpm-lock.yaml                # Workspace lockfile (lockfileVersion 9.0, pnpm 11)
├── pnpm-workspace.yaml           # pnpm workspace definition + allowBuilds for pnpm 11
├── tsconfig.base.json            # Shared TS config
└── docker-compose.yml            # All services (db + pgadmin + api + web)
```

---

## Test Suite

The project uses pytest with `asyncio_mode = "auto"` and five custom markers:

| Marker | Scope | Files |
|--------|-------|-------|
| `unit` | Core logic — chunking, config, CRUD, routes, queue worker, vector store, formatter harness, format validator | `tests/unit/` (9 files, ~100 tests) |
| `agent` | Agent behavior — research, copywriter, red team, asset studio, optimizer, blog formatter, carousel formatter | `tests/agents/` (7 files, ~41 tests) |
| `eval` | Outcome evals with LLM-as-Judge scoring across pipeline stages | `tests/evals/` (4 test files, 34 parametrized cases) |
| `golden` | Trajectory validation against golden dataset | `tests/golden/` (23+ cases across 6 categories) |
| `integration` | End-to-end orchestrator flows | `tests/integration/` (CI-only) |

### Outcome Eval Test Matrix

| Test File | Cases | Pipeline Stage |
|-----------|-------|----------------|
| `test_outcome_research.py` | 14 (H-001..H-004, R-001..R-004, F-001..F-002, M-001..M-004) | ResearchAgent |
| `test_outcome_script.py` | 6 (H-001..H-004, R-003, M-004) | CopywriterAgent |
| `test_outcome_factcheck.py` | 10 (H-001..H-004, R-001..R-002, R-004, E-001, F-003, F-004) | RedTeamAgent |
| `test_outcome_optimizer.py` | 4 (R-001, R-002, R-004, F-004) | ScriptOptimizerAgent |

Eval modes:
- **Golden mode** (default): Uses pre-recorded `reference_outputs` from `golden_dataset.json` — deterministic, fast, no API calls.
- **Live mode** (`--live`): Runs real agents with LLM calls, scores live output. Use `--update-baselines` to refresh golden references.

### CI Pipeline (GitHub Actions)

The `.github/workflows/ci.yml` pipeline runs on push/PR:

```
lint (python + frontend in parallel)
├── unit-tests + agent-tests (parallel, after python lint)
│    └── integration-tests (after unit + agent)
└── lint-frontend → build-frontend → docker-build-web
docker-build-api (after python lint only)
```

---

## MVP Status

### Fully Implemented (Pipeline Steps 1–9)

- **Step 1 (Ingestion)** — `POST /api/v1/jobs/` creates a PENDING RenderJob
- **Step 2 (Extraction)** — `MarkdownTextSplitter` chunks raw_text into RAW-CONTEXT scope vectors
- **Step 3 (Deep Research)** — Tavily web search + ResearchAgent produces refined LOCAL chunks **and a `refined_context` summary** (prompt chaining pattern)
- **Step 4 (Source Fact-Check)** — Passthrough; Red Team catches issues downstream
- **Step 5 (Scripting)** — CopywriterAgent receives `refined_context` from orchestrator (no direct vector store access). On revision, `ScriptOptimizerAgent` surgically patches failed claims instead of full re-draft
- **Step 6 (Red Team)** — RedTeamAgent audits script claims with three-pass evaluation, persists verdicts, configurable max revision loops
- **Step 7 (Format Output)** — BlogFormatterAgent and CarouselFormatterAgent produce structured blog/carousel outputs via Plan-then-Execute two-phase LLM calls. Wrapped in `FormatterHarness` with doom loop detection. Platform-aware validation (Twitter 280, LinkedIn 700, Instagram 2200 character limits). Branches by `format_type`: `video` skips, `blog`/`carousel` format then complete, `all` runs both in parallel then continues to asset generation
- **Step 8 (Asset Generation)** — AssetStudioAgent generates prompts (mocked `s3://` URL)
- **Step 9 (Completion)** — LOCAL-scope chunk cleanup, final state

### Infrastructure

- **Nx Monorepo** — Backend (`apps/api/`), Frontend (`apps/web/`), Shared types (`libs/shared-types/`)
- **Next.js Frontend** — App Router with shadcn/ui, React Query, Zustand; Docker-ready with standalone output
- **Postgres-backed Queue** — `QueueWorker` with `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` + crash recovery
- **Web Search Enrichment** — Tavily results ingested as LOCAL-scope vectors before research
- **Prompt Chaining (Semantic Memory)** — `refined_context` column on `render_jobs`; orchestrator mediates context between Research → Copywriter agents
- **Evaluator-Optimizer Pattern** — Configurable models/temperatures via env vars for both Red Team and Optimizer agents
- **Test Suite** — Unit (~100) + agent (~41) + integration tests with CI pipeline via GitHub Actions
- **Eval Infrastructure** — LLM-as-Judge scoring (judge.py), deterministic assertions, rubrics, golden dataset (23+ cases), 4 outcome test files with 34 parametrized cases
- **Multi-provider LLM** — Routing via model name prefix: `gemini-*` → Google GenAI SDK, all others → Together AI (OpenAI-compatible). All production agents default to Together AI (`meta-llama/Llama-3.3-70B-Instruct-Turbo`). Configurable per-stage via env vars. Embeddings always use `models/gemini-embedding-001` (Gemini). Eval suite uses separate `eval_*` model configs.
- **Multi-Format Output** — Blog and carousel formatters with Plan-then-Execute two-phase LLM calls, `FormatterHarness` generate-validate-retry with doom loop detection, platform-aware validation (per-slide character limits)
- **Docker** — 4-service Compose stack (pgvector, pgAdmin, API, Web). Migrations auto-run on API container start via `entrypoint.sh`. Single workspace lockfile at repo root (`apps/web/pnpm-lock.yaml` removed). pnpm 11 `allowBuilds` in `pnpm-workspace.yaml`.

### Intentionally Deferred (Wizard of Oz MVP)

- **SynthID Watermarking** — Config flag exists (`synthid_watermark_enabled`) but no implementation
- **GLOBAL Knowledge Base** — Skipped; platform constraints are hardcoded in agent system prompts
- **Real Asset Generation** — No TTS, video rendering, or FFmpeg; agent returns mocked URLs
