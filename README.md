# Content Factory

Multi-agent system that generates short-form video content (Shorts/Reels/TikToks) for high-stakes domains — politics, macro-economics, historical analysis. Treats **Truth and Guardrails as first-class citizens** via a rigorous Red Team agentic loop that verifies claims against a vector database before any rendering occurs.

**Nx monorepo** with a Python FastAPI backend, Next.js frontend, and shared TypeScript types.

## Core Differentiators

- **Agentic Over Atomic** — Research, Copywriter, and Red Team agents debate and correct each other through structured revision loops.
- **Prompt Chaining with Semantic Memory** — ResearchAgent produces a condensed `refined_context` summary that the orchestrator persists and passes downstream. The CopywriterAgent works from this curated context instead of calling the vector store directly — eliminating context-window bloat, enabling auditable research summaries, and ensuring consistent behavior across revision loops.
- **Zero-Hallucination Guardrails** — Red Team breaks scripts into atomic claims, cross-references each against the vector store directly, and persists verdicts to Postgres. Claims that fail are sent back for revision (max 3 attempts before human escalation).
- **Governance-as-Code** — Full audit trail via `fact_check_claims` table with evidence references linked to source chunks. API returns the complete fact-check report alongside scripts and assets.
- **Web-Enriched RAG** — Tavily search enriches user-provided context with live web results, ingested as vector chunks for semantic retrieval by downstream agents.
- **Evaluator-Optimizer Pattern** — On revision, a dedicated `ScriptOptimizerAgent` surgically patches failed claims instead of re-drafting the entire script, preserving quality sections and reducing hallucination drift.

---

## The 8-Step Pipeline

A `RenderJob` flows through these state transitions asynchronously:

### 1. Ingestion (`PENDING`)
User submits a topic (e.g., *"BRICS De-dollarization 2025"*) along with pre-context (source URLs, raw text, audience target) via `POST /api/v1/jobs/`.

### 2. Extraction & Chunking
`MarkdownTextSplitter` chunks the raw text into `RAW-CONTEXT` scope vectors in the pgvector `research_chunks` table.

### 3. Deep Research (`RESEARCHING`)
Tavily web search enriches the topic with live results (ingested as `LOCAL`-scope vectors). The **Research Agent** (`gemini-2.5-flash`) retrieves all chunks via semantic search, produces refined `LOCAL` chunks vetted for factual accuracy, **and synthesizes a `refined_context` summary** — a condensed, self-contained research brief persisted to the `render_jobs` table by the orchestrator.

### 4. Source Fact-Check (`FACT_CHECKING_RESEARCH`)
**MVP: Passthrough** — auto-advances to `SCRIPTING`. The Red Team at Step 6 catches issues downstream.

### 5. Script & Storyboard (`SCRIPTING`)
The **Copywriter Agent** (`gemini-1.5-pro`, temp=0.7) receives the **`refined_context`** from the orchestrator (not the raw vector store). This curated context ensures a bounded, consistent input regardless of chunk count or embedding noise. The agent drafts a retention-optimized script + visual storyboard.

On revision (when Red Team rejects claims), the **Script Optimizer Agent** (`gemini-2.5-flash`, temp=0.3, both configurable) receives only the failed claims and patches them surgically — preserving the rest of the script.

### 6. Red Team Evaluation (`FACT_CHECKING_SCRIPT`)
The critical step. The **Red Team Agent** (`gemini-1.5-pro`, temp=0.0, both configurable) uses a three-pass evaluation with `.with_structured_output()`:

1. **Claim Extraction** — Breaks the script into atomic claims
2. **Evidence Retrieval** — Per-claim `semantic_search(query=claim.search_query, top_k=5)` against the vector store
3. **Verdict** — Evaluates each claim against enriched evidence

Results:
- **SUPPORTED** → Script passes, claims persisted to `fact_check_claims` table with evidence references
- **UNSUPPORTED/CONTESTED** → Script sent back to Step 5 with structured feedback. After 3 failures → `HUMAN_REVIEW_NEEDED`
- Human override available via `POST /api/v1/jobs/{id}/approve-script`

### 7. Asset Generation (`ASSET_GENERATION`)
**MVP: Mocked** — The **Asset Studio Agent** (`gemini-2.5-flash`) generates Veo/Lyria production prompts but returns a fake `s3://` URL. No real TTS, video rendering, or FFmpeg yet.

### 8. Completion (`COMPLETED`)
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
| Monorepo | Nx workspace, pnpm |
| Frontend | Next.js 16 (App Router), React Query, Zustand, shadcn/ui, Tailwind CSS v4 |
| API | FastAPI (async, Pydantic V2), Python 3.11, uv |
| Database | PostgreSQL 16 + pgvector (HNSW index, `factory` schema) |
| ORM | SQLAlchemy 2 async (`asyncpg`) |
| Migrations | Alembic (sync via `psycopg2`) |
| AI Orchestration | LangChain + Google GenAI + Together AI (OpenAI-compatible) |
| Models | `gemini-2.5-flash` (research, assets, optimizer), `gemini-1.5-pro` (copywriting, red team) — both configurable via env vars. Eval suite supports Together AI models (Llama-3.3-70B, MiniMax-M2.7, Qwen3-235B, etc.) |
| Embeddings | `models/gemini-embedding-001` (768-dim, pgvector HNSW with cosine) |
| Web Search | Tavily (`langchain-tavily`) |
| Background Queue | `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` (no Celery/Redis) |
| Testing | pytest + pytest-asyncio + httpx + deepeval + LLM-as-Judge (Together AI) |
| CI/CD | GitHub Actions (lint → unit/agent tests → eval/integration/docker) |
| Containerization | Docker Compose (pgvector:pg16, pgAdmin4, API, Web) |
| Linter/Formatter | Ruff (Python), ESLint (TypeScript) |

---

## Quick Start

```bash
# 1. Create .env with required variables (see Environment section)
cp .env.example .env

# 2. Start all services (db + pgadmin + api + web)
docker compose up -d

# 3. Run migrations (DB must be running)
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

### Run Tests

```bash
# Backend tests (from apps/api/)
cd apps/api
uv sync --extra test             # Install deps including test extras
uv run pytest tests/ -v          # Run all tests

# Or via Docker
docker compose exec api pytest tests/ -v

# Run by marker
uv run pytest -m unit            # Unit tests only (6 files, ~55 tests)
uv run pytest -m agent           # Agent tests only (5 files, ~21 tests)
uv run pytest -m eval            # Eval benchmarks
uv run pytest -m golden          # Golden dataset validation
uv run pytest -m integration     # Integration tests (CI-only)
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
| `TOGETHER_API_KEY` | Optional — Together AI API key (required for live eval mode) |
| `DATABASE_URL` | Async connection string, e.g. `postgresql+asyncpg://postgres:postgres@localhost:5432/content_factory` |
| `POSTGRES_USER` | Docker Compose DB user |
| `POSTGRES_DB` | Docker Compose DB name |
| `POSTGRES_PORT` | Docker Compose host port (default `5433`) |
| `PGADMIN_EMAIL` | pgAdmin login email |
| `PGADMIN_PASSWORD` | pgAdmin login password |
| `API_PORT` | Docker Compose API host port (default `8000`) |

Optional `.env` overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `EVALUATOR_MODEL` | `gemini-1.5-pro` | Red Team agent model |
| `EVALUATOR_TEMPERATURE` | `0.0` | Red Team agent temperature |
| `OPTIMIZER_MODEL` | `gemini-2.5-flash` | Script Optimizer agent model |
| `OPTIMIZER_TEMPERATURE` | `0.3` | Script Optimizer agent temperature |
| `MAX_RED_TEAM_REVISIONS` | `3` | Max revision loops before human escalation |
| `SIMILARITY_THRESHOLD` | `0.75` | Vector search cosine similarity cutoff |
| `SYNTHID_WATERMARK_ENABLED` | `True` | SynthID flag (no implementation yet) |
| `WORKER_POLL_INTERVAL_SECONDS` | `5` | QueueWorker poll interval |
| `WORKER_LOCK_TIMEOUT_MINUTES` | `15` | Stuck job recovery timeout |
| `EVAL_RESEARCH_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Eval research agent model (Together AI) |
| `EVAL_COPYWRITER_MODEL` | `MiniMaxAI/MiniMax-M2.7` | Eval copywriter agent model (Together AI) |
| `EVAL_RED_TEAM_MODEL` | `openai/gpt-oss-120b` | Eval Red Team agent model (Together AI) |
| `EVAL_OPTIMIZER_MODEL` | `openai/gpt-oss-20b` | eval optimizer model (Together AI) |
| `EVAL_JUDGE_MODEL` | `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` | LLM-as-Judge model (Together AI) |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend API base URL |
| `WEB_PORT` | `3000` | Docker Compose web host port |

---

## Project Structure

```
content-factory/                  # Nx workspace root
├── apps/
│   ├── api/                      # Python FastAPI backend
│   │   ├── app/
│   │   │   main.py               # FastAPI app + lifespan (starts/stops QueueWorker)
│   │   │   core/config.py        # pydantic-settings, reads .env (eval model configs, Together AI)
│   │   │   api/routes.py         # /api/v1/jobs/ endpoints + health check
│   │   │   db/
│   │   │     models.py           # SQLAlchemy models (factory schema)
│   │   │     session.py          # async engine + session factory
│   │   │     crud.py             # query helpers + queue operations
│   │   │   schemas/shorts.py     # Pydantic request/response models
│   │   │   services/
│   │   │     llm.py              # Multi-provider LLM routing (Gemini + Together AI)
│   │   │     vector_store.py     # pgvector ingestion & semantic search
│   │   │     chunking.py         # Markdown text splitter
│   │   │     web_search.py       # TavilySearchService
│   │   │   workers/
│   │   │     orchestrator.py     # Agentic state machine
│   │   │     queue_worker.py     # asyncio poll loop with SKIP LOCKED
│   │   │     agents.py           # BaseAgent → Research, Copywriter, RedTeam, AssetStudio
│   │   │     optimizer.py        # ScriptOptimizerAgent
│   │   │     tasks.py            # Post-completion LOCAL chunk cleanup
│   │   ├── alembic/              # Database migrations
│   │   ├── tests/                # Python test suite
│   │   ├── scripts/              # Type generation scripts
│   │   ├── pyproject.toml        # uv-managed Python deps
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
├── nx.json                       # Nx workspace config
├── package.json                  # Root package (Nx + dev deps)
├── pnpm-workspace.yaml           # pnpm workspace definition
├── tsconfig.base.json            # Shared TS config
└── docker-compose.yml            # All services (db + pgadmin + api + web)
```

---

## Test Suite

The project uses pytest with `asyncio_mode = "auto"` and five custom markers:

| Marker | Scope | Files |
|--------|-------|-------|
| `unit` | Core logic — chunking, config, CRUD, routes, queue worker, vector store | `tests/unit/` (6 files, ~55 tests) |
| `agent` | Agent behavior — research, copywriter, red team, asset studio, optimizer | `tests/agents/` (5 files, ~21 tests) |
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
lint → unit-tests + agent-tests (parallel) → eval-tests + integration-tests (PR only) + docker-build
```

---

## MVP Status

### Fully Implemented (Pipeline Steps 1–8)

- **Step 1 (Ingestion)** — `POST /api/v1/jobs/` creates a PENDING RenderJob
- **Step 2 (Extraction)** — `MarkdownTextSplitter` chunks raw_text into RAW-CONTEXT scope vectors
- **Step 3 (Deep Research)** — Tavily web search + ResearchAgent produces refined LOCAL chunks **and a `refined_context` summary** (prompt chaining pattern)
- **Step 4 (Source Fact-Check)** — Passthrough; Red Team catches issues downstream
- **Step 5 (Scripting)** — CopywriterAgent receives `refined_context` from orchestrator (no direct vector store access). On revision, `ScriptOptimizerAgent` surgically patches failed claims instead of full re-draft
- **Step 6 (Red Team)** — RedTeamAgent audits script claims with three-pass evaluation, persists verdicts, configurable max revision loops
- **Step 7 (Asset Generation)** — AssetStudioAgent generates prompts (mocked `s3://` URL)
- **Step 8 (Completion)** — LOCAL-scope chunk cleanup, final state

### Infrastructure

- **Nx Monorepo** — Backend (`apps/api/`), Frontend (`apps/web/`), Shared types (`libs/shared-types/`)
- **Next.js Frontend** — App Router with shadcn/ui, React Query, Zustand; Docker-ready with standalone output
- **Postgres-backed Queue** — `QueueWorker` with `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` + crash recovery
- **Web Search Enrichment** — Tavily results ingested as LOCAL-scope vectors before research
- **Prompt Chaining (Semantic Memory)** — `refined_context` column on `render_jobs`; orchestrator mediates context between Research → Copywriter agents
- **Evaluator-Optimizer Pattern** — Configurable models/temperatures via env vars for both Red Team and Optimizer agents
- **Test Suite** — Unit (~55) + agent (~21) + integration tests with CI pipeline via GitHub Actions
- **Eval Infrastructure** — LLM-as-Judge scoring (judge.py), deterministic assertions, rubrics, golden dataset (23+ cases), 4 outcome test files with 34 parametrized cases
- **Multi-provider LLM** — Gemini (production) + Together AI (evals) routing via model name prefix. Configurable eval models for each agent stage
- **Docker** — 4-service Compose stack (pgvector, pgAdmin, API, Web)

### Intentionally Deferred (Wizard of Oz MVP)

- **SynthID Watermarking** — Config flag exists (`synthid_watermark_enabled`) but no implementation
- **GLOBAL Knowledge Base** — Skipped; platform constraints are hardcoded in agent system prompts
- **Real Asset Generation** — No TTS, video rendering, or FFmpeg; agent returns mocked URLs
