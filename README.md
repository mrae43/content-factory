# Content Factory

Multi-agent system that generates multi-format content (Shorts/Reels/TikToks, blog articles, social carousels, **and now full video assets**) for high-stakes domains — politics, macro-economics, historical analysis. Treats **Truth and Guardrails as first-class citizens** via a rigorous Red Team agentic loop that verifies claims against a vector database before any rendering occurs.

**Nx monorepo** with a Python FastAPI backend, Next.js 16 frontend (App Router, React 19), and shared TypeScript types. Ships with an editorial design system (Stone & Copper palette, Playfair Display + Inter + JetBrains Mono typography, dark mode).

## Core Differentiators

- **Agentic Over Atomic** — Copywriter and Red Team agents debate and correct each other through structured revision loops (Evaluator-Optimizer pattern).
- **Prompt Chaining with Narrative + Evidence Injection** — The orchestrator builds a condensed `refined_context` narrative directly from `user_reference` + `story_directives`, persisted to the `render_jobs` table. A `ContextBuilder` service then performs structured RAG retrieval using `title` + `story_directives` + `user_reference`, producing `evidence_sections` — a formatted text block of similarity-scored, relevance-tagged chunks that spans **both run-local and long-term (GLOBAL) memory**. The CopywriterAgent works from both `refined_context` and `evidence_sections` instead of calling the vector store directly — eliminating context-window bloat, enabling auditable research summaries with grounded evidence, and ensuring consistent behavior across revision loops.
- **Structured Evidence Retrieval** — A dedicated `ContextBuilder` service queries the vector store before script writing, enriching the copywriter prompt with similarity-scored, relevance-tagged evidence chunks (`story_directives` guide the query composition). Dual diversified queries (title-only + title+angle+reference) retrieve from three scopes: `RAW-CONTEXT` (user-provided), `LOCAL` (web search), and `GLOBAL` (long-term compressed facts from prior completed jobs). Evidence sections are injected directly into agent prompts for grounded generation.
- **Long-Term Memory (GLOBAL Scope)** — At pipeline completion, SUPPORTED claims are compressed into factual statements via a dedicated LLM (`promotion_model`) and ingested as `GLOBAL`-scope chunks (`job_id=None`). These survive job deletion and are queried by the ContextBuilder on all subsequent runs — creating a persistent organizational knowledge base that grows with every completed job.
- **Working Memory Across Pipeline** — A tripartite `working_memory` JSONB column on `RenderJob` persists context across pipeline transitions: `copywriter_rationale` (narrative intent and claim disambiguations), `optimizer_phase` (per-iteration patch summaries, resolved claims, fallback rate), and `epistemic_ledger` (weak passes — claims with UNCERTAIN/CONTESTED/SUPPORTED-low-confidence). Agents remain stateless; the orchestrator reads and writes working memory, injecting relevant sections into agent prompts. Formatters consume the epistemic ledger for automatic hedging of uncertain content.
- **Zero-Hallucination Guardrails** — Red Team breaks scripts into atomic claims, cross-references each against the vector store directly, and persists verdicts to Postgres. Claims that fail are sent back for revision (max 3 attempts before human escalation). Configurable `GuardrailStrictness` profiles (`Low`, `Medium`, `High`) control similarity thresholds, `claim_categories` (claim types checked), and whether `UNCERTAIN` verdicts trigger revision (`uncertain_is_soft_fail`) or require human review (`requires_human_review`). An `uncertain_pass_through` escape hatch on `StoryDirectives` waives only the UNCERTAIN revision trigger under High profile.
- **Governance-as-Code** — Full audit trail via `fact_check_claims` table with evidence references linked to source chunks. API returns the complete fact-check report alongside scripts and assets.
- **Web-Enriched RAG** — Tavily search enriches user-provided context with live web results, ingested as vector chunks for semantic retrieval by downstream agents.
- **Evaluator-Optimizer with Short-Term Memory** — On revision, a dedicated `ScriptOptimizerAgent` surgically patches failed claims instead of re-drafting the entire script, preserving quality sections and reducing hallucination drift. An **Optimization History Ledger** (ADR 0006) tracks claim identity across evaluator-optimizer iterations via Gemini embedding anchoring — a `ClaimMapper` service (`app/services/claim_mapper.py`) computes cosine similarity matrices with greedy 1-to-1 assignment to resolve which claims were fixed, regressed, or unchanged, preventing the optimizer from reverting previously-successful patches.
- **Live Video Generation** — The `VideoGeneratorAgent` (`ServiceAgent`) generates real videos via Together AI's video API — submits a generation job, polls for completion (up to 300s), downloads the result, and uploads to S3/SeaweedFS. Replaces the legacy mocked-URL `AssetStudioAgent`. The `VideoFormatterAgent` produces a single-shot `unified_visual_prompt` with scene structure, duration, and visual style. A provider-agnostic `VideoGenProvider` ABC mirrors the LLM provider registry pattern for future backend swaps.
- **Multi-Format Output** — Pipeline branches by `format_type` after Red Team approval: `video` → formatting (video storyboard) → asset generation (render), `blog` → formatting → complete, `carousel` → formatting → asset generation (images) → complete, or `all` → blog + carousel formatting in parallel → asset generation → complete. `AgentHarness` wraps each formatter with generate-validate-retry loops and doom loop detection.
- **S3/SeaweedFS Object Storage** — `StorageAdapter` abstracts between `S3Storage` (boto3, auto-creates buckets, `device_id/job_id` key prefixing) and `LocalStorage` (static files). SeaweedFS runs as a first-class Docker service. Default backend: `s3`.
- **Platform-Aware Validation** — `BlogValidator` and `CarouselValidator` enforce schema constraints and platform-specific character limits (Twitter 280, LinkedIn 700, Instagram 2200) before accepting output.
- **Declarative Tool Registry** — All pipeline capabilities (image generation/upload, video gen/poll/upload, web search, semantic search, chunk ingestion, format validation) are registered as first-class `Tool` objects in a singleton `ToolRegistry` (`app/services/tools.py`). **9 standard tools** registered at startup. Symmetric permissions enforce that both the tool and the agent must consent to binding, caught at composition time. Agents declare dependencies via `_required_di_tools` and `_required_llm_tools` class variables; `AgentHarness` injects permitted tools automatically.
- **Context Checkpointing for Retries** — The RedTeamAgent checkpoints intermediate state (claim extraction + evidence retrieval) across both tenacity and harness retry layers, preventing token waste. Two-tier retry policies (`agent_api_retry` inner 3 attempts, `agent_parent_retry` outer 3 attempts) with self-healing on checkpoint corruption.
- **Two-Class Agent Hierarchy** — `LLMAgent` (provider-agnostic `self.llm`, tool-calling via `_run_tool_loop`) and `ServiceAgent` (deterministic, no LLM) both extend `BaseAgent`. All agents share declarative tool declarations, permission enforcement, and the `run(context)` → `_execute(context)` contract.

---

## The 9-Step Pipeline

A `RenderJob` flows through these state transitions asynchronously. A **Context Retrieval** phase (Step 4) replaces the legacy `FACT_CHECKING_RESEARCH` passthrough — the `refined_context` is built directly from `user_reference` + `story_directives`, and a `ContextBuilder` assembles retrieved evidence chunks (from RAW-CONTEXT, LOCAL, and GLOBAL scopes) into an `AssembledContext` for the copywriter.

After Red Team approval (Step 6), the pipeline branches by `format_type`:

- **`video`** → FORMATTING (VideoFormatterAgent produces storyboard) → ASSET_GENERATION (VideoGeneratorAgent renders) → COMPLETED
- **`blog`** or **`carousel`** → FORMATTING → ASSET_GENERATION (carousel images only) → COMPLETED
- **`all`** (default) → FORMATTING (blog + carousel in parallel) → ASSET_GENERATION (video + carousel images) → COMPLETED

### 1. Ingestion (`PENDING`)
User submits a `title` (e.g., *"BRICS De-dollarization 2025"*) along with `user_reference` (narrative foundation text), `research_inputs.source_urls` (URLs for Tavily extraction), `story_directives` (target_audience, tone, angle, guardrail_strictness), `format_type`, `platform`, and optional `device_id` (for S3 key prefixing) via `POST /api/v1/jobs/`.

### 2. Extraction & Chunking
`MarkdownTextSplitter` chunks the raw text into `RAW-CONTEXT` scope vectors in the pgvector `research_chunks` table.

### 3. Deep Research (`RESEARCHING`)
Tavily web search enriches the title with live results (ingested as `LOCAL`-scope vectors with `source_type: "WEB_SEARCH"` metadata). User-provided `source_urls` are then extracted via Tavily's extract API (ingested as `LOCAL`-scope vectors with `source_type: "URL_EXTRACT"` metadata). The orchestrator advances to `RETRIEVAL`.

### 4. Context Retrieval & Synthesis (`RETRIEVAL`)
The orchestrator builds the `refined_context` narrative directly from `user_reference` + `story_directives` (no LLM Research Agent call) — a condensed, self-contained research brief persisted to the `render_jobs` table.

The **ContextBuilder** (`app/services/context_builder.py`) then performs a **dual diversified RAG query** (q1: title, q2: title + angle + user_reference) against three vector scopes:

| Scope | Source | Lifespan |
|-------|--------|----------|
| `RAW-CONTEXT` | User-provided text chunks | Survives until job deletion |
| `LOCAL` | Tavily web search + URL extractions | **Cleaned up after COMPLETED** |
| `GLOBAL` | Compressed facts from prior completed jobs | **Persistent** — `job_id=None`, never cleaned up |

Results are enriched with `topic_relevance` labels (HIGH ≥ 0.75, MEDIUM ≥ 0.5, LOW), `source_type` metadata, and `similarity_score`. They are formatted into two evidence sections — `=== CURRENT RUN RESEARCH ===` (local) and `=== SYSTEM INTEL ===` (global) — then injected into agent prompts as `evidence_sections`. The full `AssembledContext` (narrative_summary, evidence_sections, raw_chunks) is persisted as a JSONB column on `render_jobs`.

A retry mechanism (`retrieval_retry_count`, max `retrieval_retry_max` = 3) ensures that if `assembled_context` is missing when SCRIPTING begins, the job loops back to RETRIEVAL instead of failing.

### 5. Script & Storyboard (`SCRIPTING`)
The **Copywriter Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, temp=0.7, configurable) receives the **`refined_context`** + **`evidence_sections`** (from the AssembledContext) + **`story_directives`** (target_audience, tone, angle) + **`working_memory.copywriter_rationale`** from the orchestrator. This curated, evidence-rich context ensures a bounded, consistent input regardless of chunk count or embedding noise, and grounds the script in verifiable source material. The agent drafts a retention-optimized script and populates `working_memory.copywriter_rationale` with narrative intent and claim disambiguations.

When `evidence_sections` is empty (e.g., ContextBuilder retrieved zero chunks), the agent receives a fallback message `"No additional evidence was retrieved"` and proceeds with `refined_context` alone.

On revision (when Red Team rejects claims), the **Script Optimizer Agent** (`openai/gpt-oss-20b`, temp=0.3, configurable) receives the same `evidence_sections` and `story_directives` alongside the **active failures**, **optimization history** from the ledger, and **`working_memory.optimizer_phase`** — then patches only the failed claims surgically, preserving the rest of the script and never reverting previously-successful patches. A `retrieve_evidence_for_claim` tool (two-tier: pre-existing Red Team evidence first, semantic search fallback) enables the optimizer to verify its own patches.

### 6. Red Team Evaluation (`FACT_CHECKING_SCRIPT`)
The critical step. The **Red Team Agent** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`, temp=0.0, configurable) uses a three-pass evaluation with `.with_structured_output()`:

1. **Claim Extraction** — Breaks the script into atomic claims
2. **Evidence Retrieval** — Per-claim `semantic_search(query=claim.search_query, top_k=5)` against the vector store
3. **Verdict** — Evaluates each claim against enriched evidence

After each Red Team pass, the **ClaimMapper** (`app/services/claim_mapper.py`) updates the **Optimization History Ledger** — a structured JSONB column on the `Script` model that tracks claim identity across iterations via Gemini embedding anchoring. It computes a cosine similarity matrix between previous and new claims, runs greedy 1-to-1 assignment (threshold 0.75), and resolves which claims were fixed, regressed, or unchanged. The ledger feeds into the `ScriptOptimizerAgent` prompt on the next revision so the optimizer never reverts a previously-successful patch.

The orchestrator then updates the **epistemic ledger** (`working_memory.epistemic_ledger`) — tracking `weak_passes`: claims with UNCERTAIN, CONTESTED, or SUPPORTED-low-confidence verdicts. This ledger is consumed by formatters to automatically hedge uncertain content.

Results:
- **SUPPORTED** → Script passes, claims persisted to `fact_check_claims` table with evidence references (`evidence_text_inline` snapshot of raw chunk content, `hedge_required` flag for uncertain claims). A `hedge_index` JSONB is derived on the `RenderJob` for formatters to apply hedged language. Pipeline branches based on `format_type`.
- **UNSUPPORTED/CONTESTED** → Script sent back to Step 5 with structured feedback, active failures from the ledger, and the optimizer phase from working memory. After `max_red_team_revisions` failures → `HUMAN_REVIEW_NEEDED`
- Human override available via `POST /api/v1/jobs/{id}/approve-script`

### 7. Format Output (`FORMATTING`)
Format-specific agents produce structured output depending on `format_type`:

- **BlogFormatterAgent** (LLMAgent) — Two-phase LLM calls (plan outline → execute full output) producing structured blog sections with SEO metadata
- **CarouselFormatterAgent** (LLMAgent) — Two-phase LLM calls producing platform-specific slide decks with character-limit enforcement
- **VideoFormatterAgent** (LLMAgent) — Two-phase LLM calls producing a `VideoFormatPayload` with `unified_visual_prompt`, scene breakdown, `total_duration_seconds`, `visual_style`, and `audio_direction`

Each formatter is wrapped in an **`AgentHarness`** — a generate-validate-retry loop with tool injection, doom loop detection (SHA-256 payload hashing), LLM-callable tool binding (`_inject_llm_tools`), and validator integration. `BlogValidator` and `CarouselValidator` enforce schema constraints and platform rules. Max 2 retries (3 total attempts). Formatters receive the **epistemic ledger** from working memory to apply hedging on weak claims.

When `format_type = "all"`, blog and carousel formatters run concurrently via `asyncio.gather()`. Video format always routes through FORMATTING first (to produce the storyboard), then to ASSET_GENERATION for rendering.

After formatting, `_next_status_after_formatting()` routes to:
- **COMPLETED** — blog-only or carousel-only jobs (no visual assets needed)
- **ASSET_GENERATION** — video jobs, carousel jobs, or `all` jobs (assets needed)

### 8. Asset Generation (`ASSET_GENERATION`)
The orchestrator branches by format:

- **Video** — Runs the **VideoGeneratorAgent** (`ServiceAgent`), a deterministic 4-step pipeline:
  1. Submit via `generate_video` tool (Together AI `videos.create`)
  2. Poll for completion via `poll_video` tool (up to `video_gen_max_poll_retries`=60 attempts at 5s intervals = 300s timeout)
  3. Download video bytes via `aiohttp`
  4. Upload to S3/SeaweedFS via `upload_video` tool (fallback: return download URL with `confidence_score=0.7` if upload fails)
  - Uses `VideoGenProvider` ABC (`app/services/video_gen.py`) with `TogetherVideoGen` implementation. Provider registry pattern mirrors the LLM provider system.

- **Carousel** — Runs the **CarouselImageAgent** which generates real images via Together AI `FLUX.1-schnell` with platform-specific dimensions (Instagram/LinkedIn 1088×1344, Twitter 1088×1616, TikTok 1088×1920, YouTube 1920×1088), editorial brand styling (copper and stone tones, flat vector illustration, no text/typography). Images are uploaded via `StorageAdapter` (default: **S3** via SeaweedFS using boto3 with auto-created buckets, fallback: **local** → `static/carousel_images/`) with a `device_id/job_id` folder prefix for multi-device isolation. Includes retry logic (3 attempts with exponential backoff) and a global rate-limit coordinator (asyncio `Lock`, 3s minimum gap between calls, exponential backoff on HTTP 429).

- **`all`** — Runs both video and carousel generation for their respective formats.

A `POST /api/v1/jobs/{id}/regenerate-assets` endpoint allows re-running carousel image generation post-completion. **SeaweedFS** (S3-compatible object store) runs as a Docker service alongside the stack.

The legacy `AssetStudioAgent` (mocked `s3://` URLs) is deprecated and no longer used in the orchestrator path.

### 9. Completion (`COMPLETED`)
Two-phase teardown:

1. **GLOBAL Promotion** — The orchestrator retrieves the latest SUPPORTED claims from the current job, uses a dedicated LLM (`promotion_model`, default `openai/gpt-oss-20b`) to compress each into a factual statement, and ingests them as `GLOBAL`-scope chunks (`job_id=None`, `source_type="COMPRESSED_FACT"`) — creating persistent long-term memory for future jobs.
2. **LOCAL Cleanup** — `LOCAL`-scope vector chunks (Tavily web search results and URL extractions) are garbage-collected. `RAW-CONTEXT` and `GLOBAL` chunks survive.

The final job state, scripts, audit trail, asset metadata, and S3 URLs are available via the API.

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
| Database | PostgreSQL 16 + pgvector (HNSW index, `factory` schema, indexes on GLOBAL scope) |
| ORM | SQLAlchemy 2 async (`asyncpg`) |
| Migrations | Alembic (sync via `psycopg2`) |
| AI Orchestration | LangChain + Google GenAI + Together AI (OpenAI-compatible). Claim embedding mapping via numpy. |
| Video Generation | Together AI `videos.create` / `videos.retrieve` via `AsyncTogether`. Provider-agnostic `VideoGenProvider` ABC with registry pattern (`TogetherVideoGen`). Poll-and-download pattern with S3 upload. |
| Image Generation | Together AI `v1/images/generations` with `FLUX.1-schnell` (configurable via `image_model` env var). Global rate-limit coordinator with exponential backoff. |
| Storage | `StorageAdapter` dispatcher — default **S3** (`app/storage/s3.py`, via `boto3`, auto-creates buckets, targets SeaweedFS, `device_id/job_id` key prefixing), fallback **local** (`app/storage/local.py` → `static/carousel_images/`). Configured via `STORAGE_BACKEND` env var. |
| Models | Two tiers via Together AI: **Premium** (`meta-llama/Llama-3.3-70B-Instruct-Turbo` for CopywriterAgent, RedTeamAgent), **Standard** (`openai/gpt-oss-20b` for ScriptOptimizerAgent, formatters, promotion). Video gen: Together AI (provider default model). Image model: `black-forest-labs/FLUX.1-schnell`. Eval suite uses separate `eval_*` models. Embeddings: `models/gemini-embedding-001` (Gemini). |
| Embeddings | `models/gemini-embedding-001` (768-dim, pgvector HNSW with cosine) |
| Web Search | Tavily (`langchain-tavily`) |
| Background Queue | `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` (no Celery/Redis). Two-tier tenacity retry policies (`agent_api_retry` + `agent_parent_retry`) |
| Working Memory | Tripartite JSONB on `RenderJob` (`copywriter_rationale`, `optimizer_phase`, `epistemic_ledger`) — orchestrator-owned, agents stay stateless |
| Testing | pytest + pytest-asyncio + httpx + deepeval + LLM-as-Judge (Together AI) |
| CI/CD | GitHub Actions (lint → unit/agent tests → eval/integration/docker) |
| Containerization | Docker Compose (pgvector:pg16, pgAdmin4, SeaweedFS, API, Web) |
| Linter/Formatter | Ruff (Python, line-length 88), ESLint (TypeScript) |
| Design System | Stone & Copper palette, Playfair Display + Inter + JetBrains Mono — see [`DESIGN.md`](DESIGN.md) |

---

## Quick Start

```bash
# 1. Create .env with required variables (see Environment section)
cp .env.example .env

# 2. (Optional) Configure SeaweedFS S3 identity
cp s3.example.json s3.json

# 3. Start all services (db + pgadmin + seaweedfs + api + web)
docker compose up -d

# 4. Migrations auto-run on container start via entrypoint.sh
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
| SeaweedFS (S3) | http://localhost:8333 | S3-compatible object store — buckets `media-images`, `media-videos` |

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
uv run pytest -m unit            # Unit tests only (14 files)
uv run pytest -m agent           # Agent tests only (11 files)
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
| `title` | `string` | Content title for generation |
| `user_reference` | `string` | User-provided background text as narrative foundation |
| `research_inputs.source_urls` | `string[]` | URLs to scrape via Tavily extract API |
| `story_directives` | `object` | Editorial guardrails — target_audience, tone, angle, guardrail_strictness, uncertain_pass_through |
| `format_type` | `enum` | `video`, `blog`, `carousel`, or `all` |
| `platform` | `enum` | `twitter`, `linkedin`, `instagram`, `youtube`, `tiktok` |
| `device_id` | `string?` | Client device identifier for S3 key prefixing (sent from `localStorage`) |

Optional `.env` overrides — two model tiers via Together AI (unless `gemini-` prefixed):

**Premium tier** (`meta-llama/Llama-3.3-70B-Instruct-Turbo`):

| Variable | Default | Description |
|----------|---------|-------------|
| `COPYWRITER_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Copywriter agent model (script drafting) |
| `COPYWRITER_TEMPERATURE` | `0.7` | Copywriter agent temperature |
| `EVALUATOR_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Red Team agent model (fact-checking) |
| `EVALUATOR_TEMPERATURE` | `0.0` | Red Team agent temperature |

**Standard tier** (`openai/gpt-oss-20b`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPTIMIZER_MODEL` | `openai/gpt-oss-20b` | Script Optimizer agent model (surgical patching) |
| `OPTIMIZER_TEMPERATURE` | `0.3` | Script Optimizer agent temperature |
| `ASSET_MODEL` | `openai/gpt-oss-20b` | (Legacy) Asset Studio agent model (deprecated — video now uses VideoGeneratorAgent) |
| `ASSET_TEMPERATURE` | `0.5` | (Legacy) Asset Studio agent temperature |
| `FORMATTER_MODEL` | `openai/gpt-oss-20b` | Blog/Carousel/Video formatter model |
| `FORMATTER_TEMPERATURE` | `0.3` | Blog/Carousel/Video formatter temperature |
| `MAX_RED_TEAM_REVISIONS` | `3` | Max revision loops before human escalation |
| `SIMILARITY_THRESHOLD` | `0.75` | Vector search cosine similarity cutoff |
| `CONTEXT_BUILDER_TOP_K` | `10` | Number of chunks retrieved by ContextBuilder |
| `RETRIEVAL_RETRY_MAX` | `3` | Max retries for assembled_context before failing |
| `IMAGE_MODEL` | `black-forest-labs/FLUX.1-schnell` | Carousel image generation model (Together AI) |
| `IMAGE_GEN_MAX_RETRIES` | `3` | Max retry attempts per image generation |
| `IMAGE_GEN_TIMEOUT_SECONDS` | `30` | HTTP timeout per image generation request |
| `IMAGE_STORAGE_PATH` | `static/carousel_images` | Local directory for generated carousel images (fallback `local` backend) |
| `IMAGE_GEN_SLIDE_DELAY` | `1.5` | Seconds to wait between successive slide image generations |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model for vector search and claim mapper |
| `EMBEDDING_DIMENSION` | `768` | Embedding vector dimension (must match model output) |
| `STORAGE_BACKEND` | `s3` | Storage adapter — `s3` (SeaweedFS, boto3, auto-create buckets, `device_id/job_id` key prefixing) or `local` (static files) |
| `S3_ENDPOINT_URL` | `http://seaweedfs:8333` | S3-compatible endpoint (SeaweedFS default) |
| `S3_ACCESS_KEY_ID` | `factory` | S3 access key |
| `S3_SECRET_ACCESS_KEY` | `factory-secret` | S3 secret key |
| `S3_BUCKET_IMAGES` | `media-images` | S3 bucket for generated carousel images |
| `S3_BUCKET_VIDEOS` | `media-videos` | S3 bucket for video assets |
| `S3_PUBLIC_URL` | `http://localhost:8333` | Public URL prefix for browser-accessible image URLs |
| `SYNTHID_WATERMARK_ENABLED` | `True` | SynthID flag (no implementation yet) |
| `WORKER_POLL_INTERVAL_SECONDS` | `5` | QueueWorker poll interval |
| `WORKER_LOCK_TIMEOUT_MINUTES` | `15` | Stuck job recovery timeout |
| `PROMOTION_MODEL` | `openai/gpt-oss-20b` | LLM for GLOBAL scope fact compression |
| `PROMOTION_TEMPERATURE` | `0.3` | Temperature for GLOBAL promotion LLM |
| `VIDEO_GEN_PROVIDER` | `together` | Video generation provider |
| `VIDEO_GEN_MODEL` | `""` | Video generation model (empty = provider default) |
| `VIDEO_GEN_POLL_INTERVAL_SECONDS` | `5` | Polling interval between video gen status checks |
| `VIDEO_GEN_MAX_POLL_RETRIES` | `60` | Max poll attempts before timeout (5s × 60 = 300s) |
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
├── CONTEXT.md                    # Domain glossary — pipeline concepts, working memory, tools, model tiers
├── apps/
│   ├── api/                      # Python FastAPI backend
│   │   ├── app/
│   │   │   main.py               # FastAPI app + lifespan (starts/stops QueueWorker), /images and /static mounts, redirect_slashes=False
│   │   │   core/
│   │   │     config.py            # pydantic-settings, reads .env (database_url — required, promotion_model, video_gen_*, eval model configs)
│   │   │     guardrails.py        # GuardrailStrictness enum, GUARDRAIL_PROFILES, get_guardrail_config()
│   │   │   api/routes.py         # /api/v1/jobs/ endpoints + health check + /regenerate-assets
│   │   │   db/
│   │   │     models.py           # SQLAlchemy models (factory schema) — RenderJob with title, device_id (S3 key prefix), user_reference, source_urls (JSONB), story_directives (JSONB), refined_context, assembled_context (JSONB), working_memory (JSONB), retrieval_retry_count. Script.format_payload uses TrackedJSONB (MutableDict) for in-place mutation support.
│   │   │     session.py          # async engine + session factory (settings.database_url)
│   │   │     crud.py             # query helpers + queue operations
│   │   │   schemas/
│   │   │     shorts.py           # Pydantic request/response models + FormatTypeEnum, PlatformEnum, device_id for S3 key prefixing, AssembledContext
│   │   │     formats.py          # Structured format schemas (BlogSection, CarouselSlide, VideoFormatPayload, SeoMeta)
│   │   │   services/
│   │   │     llm.py              # Multi-provider LLM routing (Gemini + Together AI)
│   │   │     vector_store.py     # pgvector ingestion & semantic search (RAW-CONTEXT, LOCAL, GLOBAL scopes)
│   │   │     chunking.py         # Markdown text splitter
│   │   │     web_search.py       # TavilySearchService
│   │   │     context_builder.py  # RAG query composition (dual diversified queries), evidence formatting, AssembledContext — queries RAW-CONTEXT + LOCAL + GLOBAL
│   │   │     claim_mapper.py     # Embedding-based claim identity tracking for the Optimization History Ledger (cosine similarity matrix + greedy 1-to-1 assignment)
│   │   │     tools.py            # Tool + ToolRegistry — declarative, permission-gated tool registration (singleton, 9 standard tools)
│   │   │     optimizer_tools.py  # make_gated_search_tool — two-tier evidence retrieval for ScriptOptimizerAgent (pre-existing evidence + semantic search fallback)
│   │   │     format_validator.py # FormatValidator → BlogValidator, CarouselValidator
│   │   │     image_gen.py        # ImageGenerationService — Together AI FLUX.1-schnell, retry logic, platform dimensions, global rate-limit coordinator (asyncio Lock, 3s min gap, exponential backoff on 429)
│   │   │     video_gen.py        # VideoGenProvider ABC + TogetherVideoGen (Together AI videos.create/retrieve). Provider registry pattern. Tool factories: make_generate_video_tool, make_poll_video_tool
│   │   │   storage/
│   │   │     adapter.py          # get_storage() dispatcher (s3 or local)
│   │   │     s3.py               # S3Storage — boto3 client, SeaweedFS, auto-create bucket
│   │   │     local.py            # LocalStorage — saves to static/carousel_images/
│   │   │   workers/
│   │   │     orchestrator.py     # Agentic state machine — 11 states, working memory management, GLOBAL promotion, epistemic ledger
│   │   │     queue_worker.py     # asyncio poll loop with SKIP LOCKED
│   │   │     agents.py           # BaseAgent → LLMAgent (Copywriter, RedTeam, AssetStudio-deprecated) + ServiceAgent
│   │   │     optimizer.py        # ScriptOptimizerAgent (receives optimization history ledger + working memory optimizer phase)
│   │   │     formatters.py       # BlogFormatterAgent, CarouselFormatterAgent, VideoFormatterAgent
│   │   │     video_generator_agent.py  # VideoGeneratorAgent (ServiceAgent) — 4-step: submit → poll → download → upload S3
│   │   │     carousel_image_agent.py  # CarouselImageAgent (ServiceAgent) — real image gen via Together AI FLUX
│   │   │     harness.py          # AgentHarness — generate-validate-retry with doom loop detection, tool injection, dual ServiceAgent/LLMAgent paths
│   │   │     retry_policies.py   # Centralized two-tier tenacity configs (agent_api_retry + agent_parent_retry)
│   │   │     tasks.py            # Post-completion LOCAL chunk cleanup + GLOBAL scope fact promotion
│   │   ├── alembic/              # Database migrations
│   │   │   versions/             # incl. migration for GLOBAL scope, working memory columns
│   │   ├── tests/                # Python test suite
│   │   │   ├── evals/
│   │   │   │   ├── contracts/    # 27 eval contracts across 8 pipeline stages (Research → E2E)
│   │   │   │   ├── fixtures/     # eval1_research.json — frozen Tavily corpus + cached scores
│   │   │   │   ├── plans/        # Implementation plans for eval 1.x + eval 9 (memory)
│   │   │   │   ├── audit/        # Eval audit trail
│   │   │   │   ├── conftest.py   # EvalRunner, researching_runner, quality_corpus fixtures
│   │   │   │   ├── schemas.py    # ResearchingCase, QualityCorpus, CachedChunkScore, etc.
│   │   │   │   ├── chunk_quality_scorer.py
│   │   │   │   ├── assertions.py # Deterministic check helpers (chunk count, domain diversity, etc.)
│   │   │   │   ├── judge.py      # LLM-as-Judge scoring
│   │   │   │   ├── rubrics.py    # Weighted scoring rubrics
│   │   │   │   └── baselines.json
│   │   │   ├── agents/           # 11 test files
│   │   │   │   ├── test_asset_studio_agent.py
│   │   │   │   ├── test_blog_formatter.py
│   │   │   │   ├── test_carousel_formatter.py
│   │   │   │   ├── test_carousel_image_agent.py
│   │   │   │   ├── test_copywriter_agent.py
│   │   │   │   ├── test_optimizer_agent.py
│   │   │   │   ├── test_red_team_agent.py
│   │   │   │   ├── test_tool_loop.py
│   │   │   │   ├── test_video_formatter_agent.py
│   │   │   │   └── test_video_generator_agent.py
│   │   │   ├── unit/             # 14 test files
│   │   │   │   ├── test_chunking.py
│   │   │   │   ├── test_claim_mapper.py
│   │   │   │   ├── test_config.py
│   │   │   │   ├── test_context_builder.py
│   │   │   │   ├── test_crud.py
│   │   │   │   ├── test_format_validator.py
│   │   │   │   ├── test_formatter_harness.py
│   │   │   │   ├── test_image_gen_service.py
│   │   │   │   ├── test_promote_to_global.py
│   │   │   │   ├── test_queue_worker.py
│   │   │   │   ├── test_routes.py
│   │   │   │   ├── test_tool_wiring.py
│   │   │   │   ├── test_vector_store.py
│   │   │   │   ├── test_web_search.py
│   │   │   │   └── test_working_memory.py
│   │   │   ├── integration/      # 3 test files
│   │   │   │   ├── test_formatting_transition.py
│   │   │   │   ├── test_orchestrator_transitions.py
│   │   │   │   └── test_researching_source_types.py
│   │   │   └── golden/           # 23+ cases across 6 categories
│   │   ├── scripts/              # Type generation scripts + capture_corpus.py (Eval 1 corpus builder), generate_openapi.py
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
| `unit` | Core logic — chunking, config, CRUD, routes, queue worker, vector store, context builder, formatter harness, format validator, image gen service, guardrail config, story directives, **working memory, GLOBAL promotion, tool wiring** | `tests/unit/` (14 files) |
| `agent` | Agent behavior — copywriter, red team, asset studio (deprecated), optimizer, blog/carousel/video formatters, carousel image, **video generator** | `tests/agents/` (11 files, 80+ tests) |
| `eval` | Outcome evals with LLM-as-Judge scoring across pipeline stages + **Eval 1** (research coverage + chunk quality) + **Eval 9** (memory implementation) | `tests/evals/` (6 files, 40+ parametrized cases) |
| `golden` | Trajectory validation against golden dataset | `tests/golden/` (23+ cases across 6 categories) |
| `integration` | End-to-end orchestrator flows with RETRIEVAL phase, retry logic, evidence context passing, formatting transitions, **video/carousel/branch transitions** | `tests/integration/` (3 files, 80+ tests, CI-only) |

### Eval Contracts & Criteria

The eval suite is governed by formal contracts at `apps/api/tests/evals/contracts/` — 27 files across 8 pipeline stages (Research Desk → Retrieval Desk → Writer's Desk → Fact-Check Desk → Fact-Check Loop → Layout Desk → Pipeline Status → End-to-End). Each contract specifies the eval method, pass conditions, and thresholds. The master criteria document at `apps/api/tests/evals/evals-criteria.md` (409 lines) defines 5 guiding principles, 14 failure codes (F-R1 to F-P2), and 7 recommended eval datasets.

**Eval 1 (Research Phase)** adds two dedicated test files:
- `test_eval1_research_coverage.py` — Validates chunk count, domain diversity, duplicate detection, scope/source_type metadata from Tavily-researched topics
- `test_eval1_chunk_quality.py` — LLM-judged scoring (relevance, density, coherence) on a frozen corpus of 7 canonical topics (BRICS, Quantum Computing, Fed Rates, etc.) captured via `scripts/capture_corpus.py`

**Eval 9 (Memory Implementation)** adds a dedicated plan at `tests/evals/plans/eval9-memory-implementation-plan.md`.

### Outcome Eval Test Matrix

| Test File | Cases | Pipeline Stage |
|-----------|-------|----------------|
| `test_outcome_research.py` | 14 (H-001..H-004, R-001..R-004, F-001..F-002, M-001..M-004) | Researching (Tavily ingest) |
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

- **Step 1 (Ingestion)** — `POST /api/v1/jobs/` creates a PENDING RenderJob with `title`, `user_reference`, `research_inputs.source_urls`, `story_directives`, `format_type`, `platform`, and optional `device_id` (for S3 key prefixing)
- **Step 2 (Extraction)** — `MarkdownTextSplitter` chunks raw_text into RAW-CONTEXT scope vectors (with `source_type: "USER_PROVIDED"` metadata)
- **Step 3 (Deep Research / Web Enrichment)** — Tavily web search ingests live results as LOCAL-scope vectors (with `source_type: "WEB_SEARCH"` metadata). User-provided `source_urls` are extracted via Tavily extract API (with `source_type: "URL_EXTRACT"` metadata). Advances to RETRIEVAL.
- **Step 4 (Context Retrieval & Synthesis)** — The orchestrator builds `refined_context` directly from `user_reference` + `story_directives` (no LLM Research Agent). **ContextBuilder** then performs a **dual diversified RAG query** (title + angle + user_reference) against RAW-CONTEXT, LOCAL, and **GLOBAL** scopes, producing an `AssembledContext` (narrative_summary + evidence_sections + raw_chunks) persisted as JSONB. Retry mechanism (max 3) for missing context.
- **Step 5 (Scripting)** — CopywriterAgent receives `refined_context`, `evidence_sections`, `story_directives`, and `working_memory.copywriter_rationale` from orchestrator. On revision, `ScriptOptimizerAgent` surgically patches failed claims with the same evidence context plus active failures, optimization history from the ledger, and `working_memory.optimizer_phase` — preventing reversion of previously-successful patches. The optimizer uses a `retrieve_evidence_for_claim` tool for self-verification.
- **Step 6 (Red Team)** — RedTeamAgent audits script claims with three-pass evaluation, persists verdicts with `evidence_text_inline` snapshots and `hedge_required` flags, configurable max revision loops. **ClaimMapper** updates the Optimization History Ledger after each pass. **Epistemic ledger** (`working_memory.epistemic_ledger`) tracks weak passes for formatter hedging.
- **Step 7 (Format Output)** — BlogFormatterAgent, CarouselFormatterAgent, and **VideoFormatterAgent** produce structured blog/carousel/video outputs via Plan-then-Execute two-phase LLM calls. Wrapped in `AgentHarness` with tool injection, doom loop detection, and validator integration. Platform-aware validation (Twitter 280, LinkedIn 700, Instagram 2200 character limits). Formatters consume the epistemic ledger for automatic hedging. Branches by `format_type`: video → FORMATTING → ASSET_GENERATION, blog/carousel → FORMATTING → COMPLETED (no assets) or ASSET_GENERATION, `all` → formats in parallel → asset generation.
- **Step 8 (Asset Generation)** — **VideoGeneratorAgent** (ServiceAgent) generates real videos via Together AI: submit → poll (up to 300s) → download → upload to S3. Provider-agnostic `VideoGenProvider` ABC with `TogetherVideoGen` implementation. **CarouselImageAgent** generates real images via Together AI FLUX.1-schnell with platform-specific dimensions (1088×1344/1616/1920), editorial styling (no text/typography), S3/SeaweedFS storage with `device_id/job_id` folder prefixing, global rate-limit coordinator (3s min gap, exponential backoff on 429), and retry logic (3 attempts). Regenerate endpoint available for carousel images post-completion. Legacy `AssetStudioAgent` (mocked URLs) is deprecated.
- **Step 9 (Completion)** — **Two-phase teardown**: (1) GLOBAL promotion — SUPPORTED claims compressed into factual statements via `promotion_model` LLM and ingested as persistent `GLOBAL`-scope chunks (`job_id=None`); (2) LOCAL-scope chunk cleanup. Final job state, scripts, claims audit, and S3 asset URLs available via API. Working memory JSONB retained as audit trail.

### Infrastructure

- **Nx Monorepo** — Backend (`apps/api/`), Frontend (`apps/web/`), Shared types (`libs/shared-types/`)
- **Next.js Frontend** — App Router with shadcn/ui, React Query, Zustand; Docker-ready with standalone output
- **Postgres-backed Queue** — `QueueWorker` with `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` + crash recovery. Two-tier tenacity retry policies (`agent_api_retry` + `agent_parent_retry`) for transient API errors.
- **Declarative Tool Registry** — Singleton `ToolRegistry` (`app/services/tools.py`) registers all pipeline capabilities as first-class `Tool` objects. **9 standard tools**: `generate_image`, `upload_image`, `execute_web_search`, `semantic_search`, `ingest_chunks`, `validate_format`, **`generate_video`**, **`poll_video`**, **`upload_video`**. Symmetric permissions enforce that both the tool and the agent must consent to binding. `register_standard_tools()` registers 9 capabilities at startup. Agents declare `_required_di_tools` / `_required_llm_tools` class variables; `AgentHarness` injects permitted tools automatically at composition time.
- **Two-Class Agent Architecture** — `LLMAgent` (provider-agnostic `self.llm`, `_run_tool_loop` for LLM-decided tool calls, context checkpointing with self-healing) and `ServiceAgent` (no LLM, deterministic DI tools only) both extend `BaseAgent`. All agents share the `run(context)` → `_execute(context)` contract.
- **Context Checkpointing for Retries** — RedTeamAgent checkpoints claim extraction + evidence retrieval across tenacity and harness retries. Self-healing wipes corrupted `_*_checkpoint` keys on `ValidationError`/`TypeError`/`KeyError`/`ValueError` and re-raises for a fresh retry.
- **Web Search Enrichment** — Tavily web search by `title` + Tavily extract from user-provided `source_urls`, both ingested as LOCAL-scope vectors with `source_type: "WEB_SEARCH"` / `"URL_EXTRACT"` metadata
- **Context Builder (Structured RAG)** — `ContextBuilder` service (`app/services/context_builder.py`) composes **dual diversified queries** from `title` + `story_directives` + `user_reference`, retrieves from RAW-CONTEXT, LOCAL, and **GLOBAL** scopes, enriches chunks with `topic_relevance` labels, and formats evidence sections for prompt injection. `AssembledContext` persisted as JSONB on `render_jobs`.
- **Prompt Chaining + Evidence Injection** — Orchestrator mediates ContextBuilder → Copywriter pipeline: `refined_context` (narrative from user_reference + story_directives) + `evidence_sections` (retrieved chunks from all three scopes) + `story_directives` (audience/tone/angle) + `working_memory` are injected into agent prompts. Copywriter and Optimizer both receive the same evidence context.
- **Evaluator-Optimizer with Short-Term Memory** — Configurable models/temperatures via env vars for both Red Team and Optimizer agents. **Optimization History Ledger** (ADR 0006) tracks claim identity via `ClaimMapper` service (numpy cosine similarity, greedy 1-to-1 assignment at 0.75 threshold). `optimization_history` JSONB column on `Script` model prevents optimizer oscillation.
- **Working Memory Across Pipeline** — Tripartite `working_memory` JSONB on `RenderJob`: `copywriter_rationale` (narrative intent + claim disambiguations), `optimizer_phase` (per-iteration patch summaries, fallback rate), `epistemic_ledger` (weak passes for hedging). Orchestrator-owned; agents remain stateless. Formatters consume epistemic ledger for automatic hedging of uncertain content.
- **GLOBAL Scope / Long-Term Memory** — At COMPLETED, SUPPORTED claims are compressed into factual statements via a dedicated LLM (`promotion_model`) and ingested as `GLOBAL`-scope chunks (`job_id=None`, `source_type="COMPRESSED_FACT"`) — creating a persistent organizational knowledge base across job runs. ContextBuilder queries GLOBAL scope alongside run-local scopes on every job.
- **Live Video Generation** — `VideoGeneratorAgent` (ServiceAgent, `app/workers/video_generator_agent.py`) submits to Together AI via `videos.create`, polls via `videos.retrieve` (up to 60 attempts at 5s intervals), downloads, and uploads to S3. `VideoGenProvider` ABC (`app/services/video_gen.py`) with `TogetherVideoGen` implementation and provider registry pattern. `VideoFormatterAgent` produces `VideoFormatPayload` with unified visual prompt. Configurable via `video_gen_*` env vars. Legacy `AssetStudioAgent` (mocked URLs) is deprecated.
- **Optimizer Tool for Self-Verification** — `make_gated_search_tool()` in `app/services/optimizer_tools.py` creates a `retrieve_evidence_for_claim` tool for `ScriptOptimizerAgent`. Two-tier strategy: primary hit uses pre-existing Red Team evidence; fallback performs semantic search across RAW-CONTEXT, LOCAL, and GLOBAL scopes. Tracks fallback rate for monitoring.
- **Test Suite** — Unit + agent + integration (250+ tests) with CI pipeline via GitHub Actions. 14 unit test files, 11 agent test files, 3 integration test files, 23+ golden cases.
- **Eval Infrastructure** — LLM-as-Judge scoring (judge.py), deterministic assertions, rubrics, golden dataset, 6 outcome + eval1 test files with 40+ parametrized cases, 27 eval contracts across 8 pipeline stages, master criteria document (409 lines), frozen Tavily corpus (7 canonical topics via `scripts/capture_corpus.py`), Eval 9 memory implementation plan.
- **Multi-provider LLM** — Routing via model name prefix: `gemini-*` → Google GenAI SDK, all others → Together AI (OpenAI-compatible). Two production tiers via Together AI: **Premium** (`meta-llama/Llama-3.3-70B-Instruct-Turbo` for CopywriterAgent, RedTeamAgent) and **Standard** (`openai/gpt-oss-20b` for ScriptOptimizerAgent, formatters, promotion). Configurable per-stage via env vars. Embeddings always use `models/gemini-embedding-001` (Gemini). Eval suite uses separate `eval_*` model configs.
- **Multi-Format Output** — Blog, carousel, and video formatters with Plan-then-Execute two-phase LLM calls, `AgentHarness` generate-validate-retry with doom loop detection, tool injection, and validator integration. Platform-aware validation (per-slide character limits). Video format routes through FORMATTING to produce storyboard, then to ASSET_GENERATION for rendering.
- **Carousel Image Generation** — Real image gen via Together AI `FLUX.1-schnell` with platform-specific dimensions (1088×1344/1616/1920), editorial brand styling (no text/typography), global rate-limit coordinator (asyncio Lock, 3s min gap, exponential backoff on 429), S3/SeaweedFS storage with `device_id/job_id` folder prefixing, and retry logic (3 attempts). Regenerate endpoint available post-completion.
- **S3/SeaweedFS Cloud Storage** — `StorageAdapter` dispatcher with `S3Storage` (boto3, auto-create buckets, `device_id/job_id` key prefixing) and `LocalStorage` (static files) backends. Images uploaded to SeaweedFS S3 buckets (`media-images`, `media-videos`). Video files also uploaded via S3. Configurable via `S3_*` env vars. Default backend: `s3`.
- **Editorial Frontend Design** — App Router dark mode with Stone & Copper color tokens (oklch), Playfair Display + Inter + JetBrains Mono typography, StatusBar (Live/Stalled/Disconnected), Tabbed detail layout (TabBar), MiniPipeline tooltips, Editorial Timeline, reusable format viewers with CopyButton.
- **Docker** — 5-service Compose stack (pgvector, pgAdmin, SeaweedFS, API, Web). Migrations auto-run on API container start via `entrypoint.sh`. Single workspace lockfile at repo root (`apps/web/pnpm-lock.yaml` removed). pnpm 11 `allowBuilds` in `pnpm-workspace.yaml`. SeaweedFS S3 identity configured via `s3.json`.

### Intentionally Deferred

- **SynthID Watermarking** — Config flag exists (`synthid_watermark_enabled`) but no implementation
