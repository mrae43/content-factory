# Content Factory

Multi-agent system that generates short-form video content (Shorts/Reels/TikToks) for high-stakes domains — politics, macro-economics, historical analysis. Treats **Truth and Guardrails as first-class citizens** via a rigorous Red Team agentic loop that verifies claims against a vector database before any rendering occurs.

## Core Differentiators

- **Agentic Over Atomic** — Research, Copywriter, and Red Team agents debate and correct each other through structured revision loops.
- **Zero-Hallucination Guardrails** — Red Team breaks scripts into atomic claims, cross-references each against research sources, and persists verdicts to Postgres. Claims that fail are sent back for revision (max 3 attempts before human escalation).
- **Governance-as-Code** — Full audit trail via `fact_check_claims` table with evidence references linked to source chunks. API returns the complete fact-check report alongside scripts and assets.
- **Web-Enriched RAG** — Tavily search enriches user-provided context with live web results, ingested as vector chunks for semantic retrieval by downstream agents.

---

## The 8-Step Pipeline

A `RenderJob` flows through these state transitions asynchronously:

### 1. Ingestion (`PENDING`)
User submits a topic (e.g., *"BRICS De-dollarization 2025"*) along with pre-context (source URLs, raw text, audience target) via `POST /api/v1/jobs/`.

### 2. Extraction & Chunking
`MarkdownTextSplitter` chunks the raw text into `RAW-CONTEXT` scope vectors in the pgvector `research_chunks` table.

### 3. Deep Research (`RESEARCHING`)
Tavily web search enriches the topic with live results (ingested as `LOCAL`-scope vectors). The **Research Agent** (`gemini-2.5-flash`) retrieves all chunks via semantic search, produces refined `LOCAL` chunks vetted for factual accuracy.

### 4. Source Fact-Check (`FACT_CHECKING_RESEARCH`)
**MVP: Passthrough** — auto-advances to `SCRIPTING`. The Red Team at Step 6 catches issues downstream.

### 5. Script & Storyboard (`SCRIPTING`)
The **Copywriter Agent** (`gemini-1.5-pro`, temp=0.7) drafts a retention-optimized script + visual storyboard from the research context. On revision, the agent sees accumulated Red Team feedback.

### 6. Red Team Evaluation (`FACT_CHECKING_SCRIPT`)
The critical step. The **Red Team Agent** (`gemini-1.5-pro`, temp=0.0) uses `.with_structured_output()` to break the script into atomic claims and verifies each against the research sources:

- **SUPPORTED** → Script passes, claims persisted to `fact_check_claims` table with evidence references
- **UNSUPPORTED/CONTESTED** → Script sent back to Step 5 with detailed feedback. After 3 failures → `HUMAN_REVIEW_NEEDED`
- Human override available via `POST /api/v1/jobs/{id}/approve-script`

### 7. Asset Generation (`ASSET_GENERATION`)
**MVP: Mocked** — The **Asset Studio Agent** (`gemini-2.5-flash`) generates Veo/Lyria production prompts but returns a fake `s3://` URL. No real TTS, video rendering, or FFmpeg yet.

### 8. Completion (`COMPLETED`)
LOCAL-scope vector chunks are cleaned up. The final job state, scripts, audit trail, and asset metadata are available via the API.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/jobs/` | Create a new RenderJob (returns `202 Accepted`) |
| `GET` | `/api/v1/jobs/{id}` | Poll job status with full scripts, claims audit, and assets |
| `POST` | `/api/v1/jobs/{id}/approve-script` | Approve or reject script (human-in-the-loop) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI (async, Pydantic V2) |
| Database | PostgreSQL 16 + pgvector (HNSW index, `factory` schema) |
| ORM | SQLAlchemy 2 async (`asyncpg`) |
| Migrations | Alembic (sync via `psycopg2`) |
| AI Orchestration | LangChain + Google GenAI |
| Models | `gemini-2.5-flash` (research, assets), `gemini-1.5-pro` (copywriting, red team) |
| Embeddings | `models/gemini-embedding-001` (768-dim, pgvector HNSW with cosine) |
| Web Search | Tavily (`langchain-tavily`) |
| Background Queue | `asyncio.create_task` + `FOR UPDATE SKIP LOCKED` (no Celery/Redis) |
| Language | Python 3.11 |
| Linter | Ruff |

---

## Quick Start

```bash
# 1. Set up .env (see Environment section below)
cp .env.example .env

# 2. Start Postgres + pgAdmin + API
docker compose up -d

# 3. Run migrations (DB must be running)
alembic upgrade head

# 4. Or run API locally (outside Docker)
uvicorn app.main:app --reload
```

### Lint & Format

```bash
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
| `DATABASE_URL` | Async connection string, e.g. `postgresql+asyncpg://postgres:postgres@localhost:5432/content_factory` |
| `POSTGRES_USER` | Docker Compose DB user |
| `POSTGRES_DB` | Docker Compose DB name |
| `POSTGRES_PORT` | Docker Compose host port (default `5433`) |
| `PGADMIN_EMAIL` | pgAdmin login email |
| `PGADMIN_PASSWORD` | pgAdmin login password |

---

## Project Structure

```
app/
  main.py                  # FastAPI app + lifespan (starts/stops QueueWorker)
  core/config.py           # pydantic-settings, reads .env
  api/routes.py            # /api/v1/jobs/ endpoints
  db/
    models.py              # SQLAlchemy models (factory schema)
    session.py             # async engine + session factory
    crud.py                # query helpers + queue operations
  schemas/shorts.py        # Pydantic request/response models
  services/
    llm.py                 # LangChain + Gemini model/embedding wrappers
    vector_store.py        # pgvector ingestion & semantic search
    chunking.py            # Markdown text splitter
    web_search.py          # TavilySearchService
  workers/
    orchestrator.py        # Agentic state machine (one transition per call)
    queue_worker.py        # asyncio poll loop with SKIP LOCKED
    agents.py              # BaseAgent → Research, Copywriter, RedTeam, AssetStudio
    tasks.py               # Post-completion LOCAL chunk cleanup
```
