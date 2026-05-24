# Content Factory

Multi-agent system that generates multi-format content (Shorts/Reels/TikToks, blog articles, social carousels) for high-stakes domains — politics, macro-economics, historical analysis. Treats **Truth and Guardrails as first-class citizens** via a rigorous Red Team agentic loop that verifies claims against a vector database before any rendering occurs.

**Nx monorepo** with a Python FastAPI backend, Next.js frontend, and shared TypeScript types.

## Core Differentiators

- **Agentic Over Atomic** — Research, Copywriter, and Red Team agents debate and correct each other through structured revision loops.
- **Prompt Chaining with Semantic Memory + Evidence Injection** — ResearchAgent produces a condensed `refined_context` summary that the orchestrator persists and passes downstream. A `ContextBuilder` service then performs structured RAG retrieval using `topic` + `story_directives`, producing `evidence_sections` — a formatted text block of similarity-scored, relevance-tagged chunks. The CopywriterAgent works from both `refined_context` and `evidence_sections` instead of calling the vector store directly — eliminating context-window bloat, enabling auditable research summaries with grounded evidence, and ensuring consistent behavior across revision loops.
- **Structured Evidence Retrieval** — A dedicated `ContextBuilder` service queries the vector store before script writing, enriching the copywriter prompt with similarity-scored, relevance-tagged evidence chunks (`story_directives` guide the query composition). Evidence sections are injected directly into agent prompts for grounded generation.
- **Zero-Hallucination Guardrails** — Red Team breaks scripts into atomic claims, cross-references each against the vector store directly, and persists verdicts to Postgres. Claims that fail are sent back for revision (max 3 attempts before human escalation). Configurable `GuardrailStrictness` profiles (`Low`, `Medium`, `High`) control similarity thresholds, `claim_categories` (claim types checked), and whether `UNCERTAIN` verdicts trigger revision (`uncertain_is_soft_fail`) or require human review (`requires_human_review`). An `uncertain_pass_through` escape hatch on `StoryDirectives` waives only the UNCERTAIN revision trigger under High profile.
- **Governance-as-Code** — Full audit trail via `fact_check_claims` table with evidence references linked to source chunks. API returns the complete fact-check report alongside scripts and assets.
- **Web-Enriched RAG** — Tavily search enriches user-provided context with live web results, ingested as vector chunks for semantic retrieval by downstream agents.
- **Evaluator-Optimizer Pattern** — On revision, a dedicated `ScriptOptimizerAgent` surgically patches failed claims instead of re-drafting the entire script, preserving quality sections and reducing hallucination drift.
- **Multi-Format Output** — Pipeline branches by `format_type` after Red Team approval: `video` (asset generation), `blog` (structured articles with SEO metadata), `carousel` (platform-specific slide decks), or `all` (blog + carousel in parallel, then assets). `FormatterHarness` wraps each formatter with generate-validate-retry loops and doom loop detection.
- **S3/SeaweedFS Object Storage** — `StorageAdapter` abstracts between `S3Storage` (boto3, auto-creates buckets, `device_id/job_id` key prefixing) and `LocalStorage` (static files). SeaweedFS runs as a first-class Docker service. Default backend: `s3`.
- **Platform-Aware Validation** — `BlogValidator` and `CarouselValidator` enforce schema constraints and platform-specific character limits (Twitter 280, LinkedIn 700, Instagram 2200) before accepting output.

---

## The 9-Step Pipeline

A `RenderJob` flows through these state transitions asynchronously. A **Context Retrieval** phase (Step 4) now replaces the legacy `FACT_CHECKING_RESEARCH` passthrough — the `ResearchAgent` runs here (after web ingest in Step 3) and a `ContextBuilder` assembles retrieved evidence chunks into an `AssembledContext` for the copywriter.

After Red Team approval (Step 6), the pipeline branches by `format_type`:

- **`video`** → skips FORMATTING, goes straight to ASSET_GENERATION
- **`blog`** or **`carousel`** → FORMATTING → COMPLETED
- **`all`** (default) → FORMATTING (blog + carousel in parallel) → ASSET_GENERATION → COMPLETED

### 1. Ingestion (`PENDING`)
User submits a topic (e.g., *"BRICS De-dollarization 2025"*) along with pre-context (source URLs, raw text, audience target), `format_type`, `platform`, and optional `device_id` (for S3 key prefixing) via `POST /api/v1/jobs/`.

### 2. Extraction & Chunking
`MarkdownTextSplitter` chunks the raw text into `RAW-CONTEXT` scope vectors in the pgvector `research_chunks` table.

### 3. Deep Research (`RESEARCHING`)
Tavily web search enriches the topic with live results (ingested as `LOCAL`-scope vectors with `source_type: "WEB_SEARCH"` metadata). The orchestrator advances to `RETRIEVAL`.

### 4. Context Retrieval & Synthesis (`RETRIEVAL`)
The **Research Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, configurable) retrieves all chunks via semantic search, produces refined `LOCAL` chunks vetted for factual accuracy, **and synthesizes a `refined_context` summary** — a condensed, self-contained research brief persisted to the `render_jobs` table by the orchestrator.

The **ContextBuilder** (`app/services/context_builder.py`) then performs a structured RAG query combining the topic with `story_directives` (target_audience, tone, angle) against both `RAW-CONTEXT` and `LOCAL` scopes. Results are enriched with `topic_relevance` labels (HIGH ≥ 0.75, MEDIUM ≥ 0.5, LOW) and `source_type` metadata, then formatted into `evidence_sections` — a text block injected into the Copywriter/Optimizer agent prompts. The full `AssembledContext` (narrative_summary, evidence_sections, raw_chunks) is persisted as a JSONB column on `render_jobs`.

A retry mechanism (`retrieval_retry_count`, max `retrieval_retry_max` = 3) ensures that if `assembled_context` is missing when SCRIPTING begins, the job loops back to RETRIEVAL instead of failing.

### 5. Script & Storyboard (`SCRIPTING`)
The **Copywriter Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, temp=0.7, configurable) receives the **`refined_context`** + **`evidence_sections`** (from the AssembledContext) + **`story_directives`** (target_audience, tone, angle) from the orchestrator (not the raw vector store). This curated, evidence-rich context ensures a bounded, consistent input regardless of chunk count or embedding noise, and grounds the script in verifiable source material. The agent drafts a retention-optimized script + visual storyboard.

When evidence_sections is empty (e.g., ContextBuilder retrieved zero chunks), the agent receives a fallback message `"No additional evidence was retrieved"` and proceeds with `refined_context` alone.

On revision (when Red Team rejects claims), the **Script Optimizer Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, temp=0.3, configurable) receives the same `evidence_sections` and `story_directives` alongside the failed claims and patches them surgically — preserving the rest of the script.

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
The orchestrator branches by format: **video** jobs use the **Asset Studio Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, configurable — generates production prompts, returns mocked `s3://` URL); **carousel** jobs use the **CarouselImageAgent** which generates real images via Together AI `FLUX.1-schnell` with platform-specific dimensions (Instagram/LinkedIn 1088×1344, Twitter 1088×1616, TikTok 1088×1920, YouTube 1920×1088), editorial brand styling (copper and stone tones, flat vector illustration, no text/typography). Images are uploaded via `StorageAdapter` (default: S3/SeaweedFS, fallback: `local` → `static/carousel_images/`) with a `device_id/job_id` folder prefix for multi-device isolation. Image generation includes retry logic (3 attempts with exponential backoff) **and a global rate-limit coordinator** (asyncio `Lock`, 3s minimum gap between calls, exponential backoff on HTTP 429). A `POST /api/v1/jobs/{id}/regenerate-assets` endpoint allows re-running carousel image generation post-completion. **SeaweedFS** (S3-compatible object store) runs as a Docker service alongside the stack.

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
| `POST` | `/api/v1/jobs/{id}/regenerate-assets` | Re-run carousel image generation after completion |

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
| Image Generation | Together AI `v1/images/generations` with `FLUX.1-schnell` (configurable via `image_model` env var). Global rate-limit coordinator with exponential backoff. |
| Storage | `StorageAdapter` dispatcher — default **S3** (`app/storage/s3.py`, via `boto3`, targets SeaweedFS), fallback **local** (`app/storage/local.py` → `static/carousel_images/`). Configured via `STORAGE_BACKEND` env var. |
| Models | All default to `meta-llama/Llama-3.3-70B-Instruct-Turbo` via Together AI. Each agent stage configurable via `{research,copywriter,evaluator,optimizer,asset,formatter}_{model,temperature}` env vars. Image model: `black-forest-labs/FLUX.1-schnell`. Eval suite uses separate `eval_*` models. Embeddings: `models/gemini-embedding-001` (Gemini). |
| Embeddings | `models/gemini-embedding-001` (768-dim, pgvector HNSW with cosine) |
| Web Search | Tavily (`langchain-tavily`) |
| Background Queue | `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` (no Celery/Redis) |
| Testing | pytest + pytest-asyncio + httpx + deepeval + LLM-as-Judge (Together AI) |
| CI/CD | GitHub Actions (lint → unit/agent tests → eval/integration/docker) |
| Containerization | Docker Compose (pgvector:pg16, pgAdmin4, SeaweedFS, API, Web) |
| Linter/Formatter | Ruff (Python), ESLint (TypeScript) |
| Design System | Stone & Copper palette, Playfair Display + Inter + JetBrains Mono — see [`DESIGN.md`](DESIGN.md) |

---

## Quick Start

```bash
# 1. Create .env with required variables (see Environment section)
cp .env.example .env

# 2. Start all services (db + pgadmin + seaweedfs + api + web)
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
| SeaweedFS (S3) | http://localhost:8333 | S3-compatible object store — bucket `media-images` |

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
uv run pytest -m unit            # Unit tests only (10 files)
uv run pytest -m agent           # Agent tests only (9 files)
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

### Job Creation Fields

| Field | Type | Description |
|-------|------|-------------|
| `topic` | `string` | Content topic for generation |
| `pre_context` | `string` | Optional raw text, URLs, audience constraints |
| `format_type` | `enum` | `video`, `blog`, `carousel`, or `all` |
| `platform` | `enum` | `twitter`, `linkedin`, `instagram`, `youtube`, `tiktok` |
| `device_id` | `string?` | Client device identifier for S3 key prefixing (sent from `localStorage`) |

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
| `CONTEXT_BUILDER_TOP_K` | `10` | Number of chunks retrieved by ContextBuilder |
| `RETRIEVAL_RETRY_MAX` | `3` | Max retries for assembled_context before failing |
| `IMAGE_MODEL` | `black-forest-labs/FLUX.1-schnell` | Carousel image generation model (Together AI) |
| `IMAGE_GEN_MAX_RETRIES` | `3` | Max retry attempts per image generation |
| `IMAGE_GEN_TIMEOUT_SECONDS` | `30` | HTTP timeout per image generation request |
| `IMAGE_STORAGE_PATH` | `static/carousel_images` | Local directory for generated carousel images (fallback) |
| `IMAGE_GEN_SLIDE_DELAY` | `1.5` | Seconds to wait between successive slide image generations |
| `STORAGE_BACKEND` | `s3` | Storage adapter (`s3` via SeaweedFS, `local` fallback) |
| `S3_ENDPOINT_URL` | `http://seaweedfs:8333` | S3-compatible endpoint |
| `S3_ACCESS_KEY_ID` | `factory` | S3 access key |
| `S3_SECRET_ACCESS_KEY` | `factory-secret` | S3 secret key |
| `S3_BUCKET_IMAGES` | `media-images` | S3 bucket for generated images |
| `S3_BUCKET_VIDEOS` | `media-videos` | S3 bucket for video assets |
| `S3_PUBLIC_URL` | `http://localhost:8333` | Public URL prefix for browser-accessible image URLs |
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
│   │   │   main.py               # FastAPI app + lifespan (starts/stops QueueWorker), /images and /static mounts, redirect_slashes=False
│   │   │   core/
│   │   │     config.py            # pydantic-settings, reads .env (database_url — required, eval model configs)
│   │   │     guardrails.py        # GuardrailStrictness enum, GUARDRAIL_PROFILES, get_guardrail_config()
│   │   │   api/routes.py         # /api/v1/jobs/ endpoints + health check + /regenerate-assets
│   │   │   db/
│   │   │     models.py           # SQLAlchemy models (factory schema) — RenderJob with device_id (S3 key prefix), refined_context, research_confidence, citation_index, assembled_context (JSONB), retrieval_retry_count. Script.format_payload uses TrackedJSONB (MutableDict) for in-place mutation support.
│   │   │     session.py          # async engine + session factory (settings.database_url)
│   │   │     crud.py             # query helpers + queue operations
│   │   │   schemas/
│   │   │     shorts.py           # Pydantic request/response models + FormatTypeEnum, PlatformEnum, device_id for S3 key prefixing
│   │   │     formats.py          # Structured format schemas (BlogSection, CarouselSlide, SeoMeta)
│   │   │   services/
│   │   │     llm.py              # Multi-provider LLM routing (Gemini + Together AI)
│   │   │     vector_store.py     # pgvector ingestion & semantic search
│   │   │     chunking.py         # Markdown text splitter
│   │   │     web_search.py       # TavilySearchService
│   │   │     context_builder.py  # RAG query composition, evidence formatting, AssembledContext
│   │   │     format_validator.py # FormatValidator → BlogValidator, CarouselValidator
│   │   │     image_gen.py        # ImageGenerationService — Together AI FLUX.1-schnell, retry logic, platform dimensions, global rate-limit coordinator (asyncio Lock, 3s min gap, exponential backoff on 429)
│   │   │   storage/
│   │   │     adapter.py          # get_storage() dispatcher (s3 or local)
│   │   │     s3.py               # S3Storage — boto3 client, SeaweedFS, auto-create bucket
│   │   │     local.py            # LocalStorage — saves to static/carousel_images/
│   │   │   workers/
│   │   │     orchestrator.py     # Agentic state machine
│   │   │     queue_worker.py     # asyncio poll loop with SKIP LOCKED
│   │   │     agents.py           # BaseAgent → Research, Copywriter, RedTeam, AssetStudio
│   │   │     optimizer.py        # ScriptOptimizerAgent
│   │   │     formatters.py       # BlogFormatterAgent, CarouselFormatterAgent
│   │   │     carousel_image_agent.py  # CarouselImageAgent — real image gen via Together AI FLUX
│   │   │     harness.py          # FormatterHarness — generate-validate-retry with doom loop detection
│   │   │     tasks.py            # Post-completion LOCAL chunk cleanup
│   │   ├── alembic/              # Database migrations
│   │   ├── tests/                # Python test suite
│   │   │   ├── evals/
│   │   │   │   ├── contracts/    # 27 eval contracts across 8 pipeline stages (Research → E2E)
│   │   │   │   ├── fixtures/     # eval1_research.json — frozen Tavily corpus + cached scores
│   │   │   │   ├── plans/        # Implementation plans for eval 1.x
│   │   │   │   ├── audit/        # Eval audit trail
│   │   │   │   ├── conftest.py   # EvalRunner, researching_runner, quality_corpus fixtures
│   │   │   │   ├── schemas.py    # ResearchingCase, QualityCorpus, CachedChunkScore, etc.
│   │   │   │   ├── chunk_quality_scorer.py
│   │   │   │   ├── assertions.py # Deterministic check helpers (chunk count, domain diversity, etc.)
│   │   │   │   ├── judge.py      # LLM-as-Judge scoring
│   │   │   │   ├── rubrics.py    # Weighted scoring rubrics
│   │   │   │   └── baselines.json
│   │   ├── scripts/              # Type generation scripts + capture_corpus.py (Eval 1 corpus builder)
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
├── s3.json / s3.example.json     # SeaweedFS S3 identity config (anonymous read-only + factory admin)
└── docker-compose.yml            # All services (db + pgadmin + seaweedfs + api + web)
```

---

## Test Suite

The project uses pytest with `asyncio_mode = "auto"` and five custom markers:

| Marker | Scope | Files |
|--------|-------|-------|
| `unit` | Core logic — chunking, config, CRUD, routes, queue worker, vector store, context builder, formatter harness, format validator, image gen service, guardrail config, story directives | `tests/unit/` (10 files) |
| `agent` | Agent behavior — research, copywriter, red team, asset studio, optimizer, blog formatter, carousel formatter, carousel image, video formatter | `tests/agents/` (9 files, 65+ tests) |
| `eval` | Outcome evals with LLM-as-Judge scoring across pipeline stages + **Eval 1** (research coverage + chunk quality) | `tests/evals/` (6 files, 40+ parametrized cases) |
| `golden` | Trajectory validation against golden dataset | `tests/golden/` (23+ cases across 6 categories) |
| `integration` | End-to-end orchestrator flows with RETRIEVAL phase, retry logic, evidence context passing, formatting transitions | `tests/integration/` (60+ tests, CI-only) |

### Eval Contracts & Criteria

The eval suite is governed by formal contracts at `apps/api/tests/evals/contracts/` — 27 files across 8 pipeline stages (Research Desk → Retrieval Desk → Writer's Desk → Fact-Check Desk → Fact-Check Loop → Layout Desk → Pipeline Status → End-to-End). Each contract specifies the eval method, pass conditions, and thresholds. The master criteria document at `apps/api/tests/evals/evals-criteria.md` (409 lines) defines 5 guiding principles, 14 failure codes (F-R1 to F-P2), and 7 recommended eval datasets.

**Eval 1 (Research Phase)** adds two dedicated test files:
- `test_eval1_research_coverage.py` — Validates chunk count, domain diversity, duplicate detection, scope/source_type metadata from Tavily-researched topics
- `test_eval1_chunk_quality.py` — LLM-judged scoring (relevance, density, coherence) on a frozen corpus of 7 canonical topics (BRICS, Quantum Computing, Fed Rates, etc.) captured via `scripts/capture_corpus.py`

### Outcome Eval Test Matrix

| Test File | Cases | Pipeline Stage |
|-----------|-------|----------------|
| `test_outcome_research.py` | 14 (H-001..H-004, R-001..R-004, F-001..F-002, M-001..M-004) | ResearchAgent |
| `test_outcome_script.py` | 6 (H-001..H-004, R-003, M-004) | CopywriterAgent |
| `test_outcome_factcheck.py` | 10 (H-001..H-004, R-001..R-002, R-004, E-001, F-003, F-004) | RedTeamAgent |
| `test_outcome_optimizer.py` | 4 (R-001, R-002, R-004, F-004) | ScriptOptimizerAgent |
| `test_eval1_research_coverage.py` | 7 (coverage-happy, coverage-minimal-sources, coverage-sparse, metadata-error, high-volume, duplicate-content, scope-validation) | Researching (Tavily ingest) |
| `test_eval1_chunk_quality.py` | 7 (quality-brics, quality-sparse-FR1, quality-boilerplate, quality-fusion, quality-ev-battery, quality-space, quality-ai-regulation) | Chunk Quality (frozen corpus) |

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

- **Step 1 (Ingestion)** — `POST /api/v1/jobs/` creates a PENDING RenderJob with `ResearchInputs`, `StoryDirectives`, `format_type`, and `platform`
- **Step 2 (Extraction)** — `MarkdownTextSplitter` chunks raw_text into RAW-CONTEXT scope vectors (with `source_type: "USER_PROVIDED"` metadata)
- **Step 3 (Deep Research / Web Enrichment)** — Tavily web search ingests live results as LOCAL-scope vectors (with `source_type: "WEB_SEARCH"` metadata). Advances to RETRIEVAL.
- **Step 4 (Context Retrieval & Synthesis)** — ResearchAgent produces `refined_context`, `research_confidence`, and `citation_index`. **ContextBuilder** then performs a structured RAG query (topic + story_directives) against RAW-CONTEXT + LOCAL scopes, producing an `AssembledContext` (narrative_summary + evidence_sections + raw_chunks) persisted as JSONB. Retry mechanism (max 3) for missing context.
- **Step 5 (Scripting)** — CopywriterAgent receives `refined_context`, `evidence_sections` (from AssembledContext), and `story_directives` (target_audience, tone, angle) from orchestrator. On revision, `ScriptOptimizerAgent` surgically patches failed claims with the same evidence context instead of full re-draft.
- **Step 6 (Red Team)** — RedTeamAgent audits script claims with three-pass evaluation, persists verdicts, configurable max revision loops
- **Step 7 (Format Output)** — BlogFormatterAgent and CarouselFormatterAgent produce structured blog/carousel outputs via Plan-then-Execute two-phase LLM calls. Wrapped in `FormatterHarness` with doom loop detection. Platform-aware validation (Twitter 280, LinkedIn 700, Instagram 2200 character limits). Branches by `format_type`: `video` skips, `blog`/`carousel` format then complete, `all` runs both in parallel then continues to asset generation
- **Step 8 (Asset Generation)** — CarouselImageAgent generates real images via Together AI FLUX.1-schnell with platform-specific dimensions (1088×1344/1616/1920), editorial styling (no text/typography), S3/SeaweedFS storage with `device_id/job_id` folder prefixing, global rate-limit coordinator (3s min gap, exponential backoff on 429), and retry logic (3 attempts). AssetStudioAgent generates video production prompts (mocked `s3://` URL). Regenerate endpoint available for carousel images post-completion.
- **Step 9 (Completion)** — LOCAL-scope chunk cleanup, final state

### Infrastructure

- **Nx Monorepo** — Backend (`apps/api/`), Frontend (`apps/web/`), Shared types (`libs/shared-types/`)
- **Next.js Frontend** — App Router with shadcn/ui, React Query, Zustand; Docker-ready with standalone output
- **Postgres-backed Queue** — `QueueWorker` with `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` + crash recovery
- **Web Search Enrichment** — Tavily results ingested as LOCAL-scope vectors with `source_type: "WEB_SEARCH"` metadata
- **Context Builder (Structured RAG)** — `ContextBuilder` service (`app/services/context_builder.py`) composes a multi-field query from `topic` + `story_directives`, retrieves from both RAW-CONTEXT and LOCAL scopes, enriches chunks with `topic_relevance` labels, and formats evidence sections for prompt injection. `AssembledContext` persisted as JSONB on `render_jobs`.
- **Prompt Chaining + Evidence Injection** — Orchestrator mediates Research → ContextBuilder → Copywriter pipeline: `refined_context` (research summary) + `evidence_sections` (retrieved chunks) + `story_directives` (audience/tone/angle) are injected into agent prompts. Copywriter and Optimizer both receive the same evidence context.
- **Evaluator-Optimizer Pattern** — Configurable models/temperatures via env vars for both Red Team and Optimizer agents
- **Test Suite** — Unit + agent + integration (200+ tests) with CI pipeline via GitHub Actions
- **Eval Infrastructure** — LLM-as-Judge scoring (judge.py), deterministic assertions, rubrics, golden dataset (23+ cases), 6 outcome + eval1 test files with 40+ parametrized cases, 27 eval contracts across 8 pipeline stages, master criteria document (409 lines), frozen Tavily corpus (7 canonical topics via `scripts/capture_corpus.py`)
- **Multi-provider LLM** — Routing via model name prefix: `gemini-*` → Google GenAI SDK, all others → Together AI (OpenAI-compatible). All production agents default to Together AI (`meta-llama/Llama-3.3-70B-Instruct-Turbo`). Configurable per-stage via env vars. Embeddings always use `models/gemini-embedding-001` (Gemini). Eval suite uses separate `eval_*` model configs.
- **Multi-Format Output** — Blog and carousel formatters with Plan-then-Execute two-phase LLM calls, `FormatterHarness` generate-validate-retry with doom loop detection, platform-aware validation (per-slide character limits)
- **Carousel Image Generation** — Real image gen via Together AI `FLUX.1-schnell` with platform-specific dimensions (1088×1344/1616/1920), editorial brand styling (no text/typography), global rate-limit coordinator (asyncio Lock, 3s min gap, exponential backoff on 429), S3/SeaweedFS storage with `device_id/job_id` folder prefixing, and retry logic (3 attempts). Regenerate endpoint available post-completion.
- **S3/SeaweedFS Storage** — `StorageAdapter` dispatcher with `S3Storage` (boto3, auto-create bucket) and `LocalStorage` (static files) backends. SeaweedFS runs as a Docker service. Configurable via `S3_*` env vars. Default backend: `s3`.
- **Docker** — 5-service Compose stack (pgvector, pgAdmin, SeaweedFS, API, Web). Migrations auto-run on API container start via `entrypoint.sh`. Single workspace lockfile at repo root (`apps/web/pnpm-lock.yaml` removed). pnpm 11 `allowBuilds` in `pnpm-workspace.yaml`.

### Intentionally Deferred (Wizard of Oz MVP)

- **SynthID Watermarking** — Config flag exists (`synthid_watermark_enabled`) but no implementation
- **GLOBAL Knowledge Base** — Skipped; platform constraints are hardcoded in agent system prompts
- **Video Asset Generation** — No TTS, video rendering, or FFmpeg; AssetStudioAgent returns mocked URLs (carousel images are real via Together AI FLUX)
