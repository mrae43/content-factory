# Integration Test — Orchestrator State Transitions

## 1. Scope & Rationale

### What We're Testing

The orchestrator (`app/workers/orchestrator.py`) is the **central state machine** that drives every RenderJob from `PENDING` → `COMPLETED`. It coordinates 6 transitions, 4 agents, vector store ingestion/search, web search, and CRUD operations. **Zero tests exist for it today.**

These are **integration tests** — we mock external services (LLM calls, web search, embedding API) but exercise real interactions between the orchestrator, CRUD functions, vector store logic, and DB session management. The tests verify that state transitions produce the correct side effects (DB writes, status changes, error logs) in the correct order.

### What We're NOT Testing

| Excluded | Why |
|:---------|:----|
| Agent LLM output quality | Covered by `@pytest.mark.agent` tests + `@pytest.mark.eval` (DeepEval) |
| Actual Gemini API calls | Mocked — agents return predefined `AgentResult` objects |
| Actual Tavily API calls | Mocked — `TavilySearchService.search` returns predefined results |
| Actual embedding API calls | Mocked — `get_embeddings`/`get_query_embeddings` return fake vectors |
| Real Postgres database | Using `mock_db_session` — we verify mock interactions, not SQL execution |
| QueueWorker poll loop | Already tested in `test_queue_worker.py` (13 tests) |
| Route HTTP layer | Already tested in `test_routes.py` (6 tests) |

---

## 2. State Machine Reference

```
PENDING ──────────────────────► RESEARCHING
                                  │
                           ┌──────┴──────┐
                           │  (web search │
                           │   + Research │
                           │    Agent)    │
                           └──────┬──────┘
                                  ▼
              FACT_CHECKING_RESEARCH (passthrough)
                                  │
                                  ▼
                             SCRIPTING ◄──────────────────┐
                                  │                        │
                           ┌──────┴──────┐                 │
                           │ Copywriter  │                 │
                           │   Agent     │                 │
                           └──────┬──────┘                 │
                                  ▼                        │
                      FACT_CHECKING_SCRIPT ───(reject)─────┘
                          │           │
                     (approved)   (max revs)
                          │           │
                          ▼           ▼
                   ASSET_GENERATION  HUMAN_REVIEW_NEEDED
                          │           ▲
                     (AssetStudio)    │ (escalate)
                          │           │
                          ▼           │
                      COMPLETED ──(cleanup)
                          ▲
                          │
                       FAILED ◄── (any exception)
```
---

## 3. Mocking Strategy

### Principle: Mock at the boundary, not the interior

We mock **external APIs** and **LLM calls** but let the orchestrator's internal logic (status dispatch, CRUD calls, evidence resolution) execute naturally. This catches interface defects between the orchestrator and its dependencies.

### What to Mock

| Dependency | Mock Target | How |
|:-----------|:-----------|:-----|
| **ResearchAgent** | `app.workers.orchestrator.ResearchAgent` | Patch class, return predefined `AgentResult` |
| **CopywriterAgent** | `app.workers.orchestrator.CopywriterAgent` | Patch class, return predefined `AgentResult` |
| **RedTeamAgent** | `app.workers.orchestrator.RedTeamAgent` | Patch class, return predefined `AgentResult` |
| **AssetStudioAgent** | `app.workers.orchestrator.AssetStudioAgent` | Patch class, return predefined `AgentResult` |
| **Web Search** | `app.workers.orchestrator._web_search_service` | Patch module-level singleton |
| **Vector Store** | `app.workers.orchestrator.ContentFactoryVectorStore` | Patch class, return `mock_vector_store` |
| **Chunking** | `app.workers.orchestrator.process_extraction_job` | Patch function, return predefined chunks |
| **CRUD** | NOT mocked | Uses `mock_db_session` — we verify calls on the mock session |
| **Settings** | `app.workers.orchestrator.settings` | Patch specific values when needed |

### Why Patch at `app.workers.orchestrator.*`

The orchestrator imports agents, vector store, chunking at the module level. Patching at `app.workers.orchestrator.ResearchAgent` intercepts the reference where it's used, not where it's defined. This is the standard Python mock pattern.

---

## 4. Test Cases

### 4.1 Transition: PENDING → RESEARCHING

**File:** `test_orchestrator_transitions.py`  
**Class:** `TestTransitionPending`

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 1 | `test_should_chunk_raw_text_and_ingest_as_raw_context` | Happy path: `pre_context` has `raw_text`, chunking returns 3 chunks | `process_extraction_job` called with job ID and raw_text; `vector_store.ingest_chunks` called with `scope="RAW-CONTEXT"`; `update_job_status` called with `RESEARCHING` |
| 2 | `test_should_handle_dict_pre_context` | `pre_context` is a dict with `raw_text` key | Extracts `raw_text` correctly from dict |
| 3 | `test_should_handle_string_pre_context` | `pre_context` is a plain string | Converts to string and passes to chunking |
| 4 | `test_should_transition_to_researching_even_with_no_chunks` | `process_extraction_job` returns empty list | `ingest_chunks` NOT called; `update_job_status` still called with `RESEARCHING` (orchestrator proceeds, doesn't fail) |
| 5 | `test_should_transition_to_failed_on_chunking_exception` | `process_extraction_job` raises exception | `log_error` called; status set to `FAILED` |

**Patches:** `process_extraction_job`, `ContentFactoryVectorStore`, `update_job_status`

---

### 4.2 Transition: RESEARCHING → FACT_CHECKING_RESEARCH

**Class:** `TestTransitionResearching`

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 6 | `test_should_search_web_ingest_and_run_research_agent` | Happy path: web search returns results, ResearchAgent succeeds | `_web_search_service.search` called with `job.topic`; web results ingested as `LOCAL` scope with metadata (`source`, `query`, `urls`, `search_depth`); `ResearchAgent` called with `{job_id, topic, vector_store}`; status → `FACT_CHECKING_RESEARCH` |
| 7 | `test_should_proceed_without_web_results` | Web search returns empty list | No web ingestion; ResearchAgent still called; status → `FACT_CHECKING_RESEARCH` |
| 8 | `test_should_filter_web_results_with_empty_content` | Web search returns results but some have empty `content` | Only valid results (non-empty content) are ingested |
| 9 | `test_should_fail_when_research_agent_returns_error` | ResearchAgent returns `AgentResult(status=ERROR)` | Exception raised → caught → `log_error` called; status → `FAILED` |
| 10 | `test_should_fail_when_research_agent_raises_exception` | ResearchAgent raises exception during `run()` | Exception caught → `log_error` called; status → `FAILED` |

**Patches:** `_web_search_service`, `ContentFactoryVectorStore`, `ResearchAgent`, `update_job_status`

---

### 4.3 Transition: FACT_CHECKING_RESEARCH → SCRIPTING (Passthrough)

**Class:** `TestTransitionFactCheckingResearch`

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 11 | `test_should_pass_through_to_scripting` | Job status is `FACT_CHECKING_RESEARCH` | `update_job_status` called directly with `SCRIPTING`; no agents created |

**Patches:** `update_job_status`  
**Note:** This is the simplest transition — pure passthrough, no agents involved.

---

### 4.4 Transition: SCRIPTING → FACT_CHECKING_SCRIPT

**Class:** `TestTransitionScripting`

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 12 | `test_should_run_copywriter_and_save_script` | Happy path: no previous script (first version) | `CopywriterAgent` created with `model_name="gemini-1.5-pro"`; agent called with `{job_id, topic, vector_store, feedback}`; `save_script` called with `version=1`; status → `FACT_CHECKING_SCRIPT` |
| 13 | `test_should_increment_version_for_revision` | Previous script exists with `version=2` | `get_latest_script` returns mock with `version=2`; `save_script` called with `version=3` |
| 14 | `test_should_pass_revision_feedback_to_agent` | Latest script has feedback in `feedback_history` | Feedback string extracted from last entry; passed in agent context |
| 15 | `test_should_handle_dict_feedback_history` | `feedback_history` contains dict entries like `{"source": "human_editor", "comment": "..."}` | Extracts `feedback` key from dict entry |
| 16 | `test_should_fail_when_copywriter_returns_error` | CopywriterAgent returns `AgentResult(status=ERROR)` | Exception raised → status → `FAILED` |

**Patches:** `CopywriterAgent`, `ContentFactoryVectorStore`, `get_latest_script`, `save_script`, `update_job_status`

---

### 4.5 Transition: FACT_CHECKING_SCRIPT → ASSET_GENERATION | SCRIPTING | HUMAN_REVIEW_NEEDED

**Class:** `TestTransitionFactCheckingScript`

This is the most complex transition — three possible outcomes with branching logic.

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 17 | `test_should_approve_when_all_claims_supported` | RedTeam returns `SUCCESS` with supported claims | `_resolve_evidence_refs` called; `save_fact_check_claims` called with script ID; `script.is_approved = True`; `db.commit` called; status → `ASSET_GENERATION` |
| 18 | `test_should_approve_with_empty_claims` | RedTeam returns `SUCCESS` with empty claims list | No claims persisted; script still approved; status → `ASSET_GENERATION` |
| 19 | `test_should_reject_and_loop_to_scripting` | RedTeam returns `REVISION_NEEDED`, version=1 < max_revisions=3 | Claims persisted; `append_script_feedback` called with reasoning; status → `SCRIPTING` |
| 20 | `test_should_escalate_after_max_revisions` | RedTeam returns `REVISION_NEEDED`, version=3 >= max_revisions=3 | No feedback appended; status → `HUMAN_REVIEW_NEEDED` |
| 21 | `test_should_escalate_after_max_revisions_custom` | RedTeam returns `REVISION_NEEDED`, version=2, max_revisions=2 | Patch `settings.max_red_team_revisions = 2`; status → `HUMAN_REVIEW_NEEDED` |
| 22 | `test_should_escalate_when_red_team_returns_escalate` | RedTeam returns `ESCALATE` | No claims persisted; status → `HUMAN_REVIEW_NEEDED` |
| 23 | `test_should_resolve_evidence_references` | Claims have `evidence_text`, vector store returns matches above threshold | `_resolve_evidence_refs` calls `semantic_search` for each claim's evidence_text; `evidence_references` populated with chunk IDs |
| 24 | `test_should_not_resolve_refs_for_empty_evidence` | Claim has empty `evidence_text` | `semantic_search` NOT called for that claim; `evidence_references` stays `[]` |

**Patches:** `RedTeamAgent`, `ContentFactoryVectorStore`, `get_latest_script`, `save_fact_check_claims`, `append_script_feedback`, `update_job_status`, `settings`

---

### 4.6 Transition: ASSET_GENERATION → COMPLETED

**Class:** `TestTransitionAssetGeneration`

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 25 | `test_should_set_video_url_and_complete` | Happy path: AssetStudioAgent returns mock `s3://` URL | `job.final_video_url` set to URL from payload; `db.commit` called; `update_job_status` called with `COMPLETED` |
| 26 | `test_should_not_transition_on_agent_failure` | AssetStudioAgent returns non-SUCCESS status | `job.final_video_url` remains `None`; status NOT updated (no else branch in code) |

**Patches:** `AssetStudioAgent`, `update_job_status`

---

### 4.7 Transition: COMPLETED (Cleanup)

**Class:** `TestTransitionCompleted`

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 27 | `test_should_cleanup_local_chunks_on_completion` | Job status is `COMPLETED` | `cleanup_local_research_chunks` called with `job.id` and `db` |

**Patches:** `cleanup_local_research_chunks`

---

### 4.8 Terminal States

**Class:** `TestTerminalStates`

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 28 | `test_should_not_transition_from_human_review_needed` | Job status is `HUMAN_REVIEW_NEEDED` | No agents created; no status changes; no DB writes |
| 29 | `test_should_not_transition_from_failed` | Job status is `FAILED` | Same — terminal, no action |

---

### 4.9 Error Handling (Cross-Cutting)

**Class:** `TestOrchestratorErrorHandling`

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 30 | `test_should_catch_any_exception_and_fail_job` | Any transition raises unexpected exception | `log_error` called with traceback; `update_job_status(FAILED)` called |
| 31 | `test_should_log_phase_in_error` | Exception in RESEARCHING phase | `log_error` called with `phase="RESEARCHING"` |
| 32 | `test_should_handle_unrecognized_status` | Job has an unexpected status string | No crash; no state change; error logged |

---

### 4.10 Multi-Step Integration Flows

**Class:** `TestOrchestratorMultiStep`

These tests call `execute_state_transition` multiple times to simulate the pipeline flowing through several states sequentially.

| # | Test Name | Scenario | Assertions |
|---|-----------|----------|------------|
| 33 | `test_full_happy_path_pending_to_completed` | Simulate all transitions in sequence: PENDING → RESEARCHING → FACT_CHECKING_RESEARCH → SCRIPTING → FACT_CHECKING_SCRIPT → ASSET_GENERATION → COMPLETED | After each call, verify status advanced correctly; after completion, `cleanup_local_research_chunks` called |
| 34 | `test_revision_loop_with_eventual_approval` | SCRIPTING → FACT_CHECKING_SCRIPT (reject) → SCRIPTING → FACT_CHECKING_SCRIPT (approve) → ASSET_GENERATION → COMPLETED | Version increments to 2 on second script; feedback appended on rejection; final script approved |
| 35 | `test_revision_loop_hits_max_and_escalates` | Three consecutive REVISION_NEEDED results | After 3rd rejection (version=3), status → `HUMAN_REVIEW_NEEDED`; pipeline stops |

---

## 5. Test Implementation Pattern

Each test follows the same structure:

```python
@pytest.mark.integration
async def test_should_chunk_raw_text_and_ingest_as_raw_context(
    mock_db_session, mock_job, mock_vector_store
):
    # ── Arrange ──
    mock_job.status = JobStatusEnum.PENDING
    mock_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

    with (
        patch("app.workers.orchestrator.process_extraction_job", new_callable=AsyncMock) as mock_chunking,
        patch("app.workers.orchestrator.ContentFactoryVectorStore", return_value=mock_vector_store),
        patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock) as mock_update,
    ):
        mock_chunking.return_value = mock_chunks

        # ── Act ──
        await execute_state_transition(mock_db_session, mock_job)

        # ── Assert ──
        mock_chunking.assert_awaited_once_with(str(mock_job.id), mock_job.pre_context["raw_text"])
        mock_vector_store.ingest_chunks.assert_awaited_once()
        ingest_call = mock_vector_store.ingest_chunks.call_args
        assert ingest_call.kwargs["scope"] == "RAW-CONTEXT"
        mock_update.assert_awaited_once_with(
            mock_db_session, mock_job.id, JobStatusEnum.RESEARCHING
        )
```

### Key Pattern Notes

1. **All tests are `@pytest.mark.integration`** — consistent with the test plan marker system
2. **All tests are `async def`** — orchestrator is fully async; `pytest-asyncio` with `asyncio_mode="auto"` handles this
3. **Patches target `app.workers.orchestrator.*`** — the module where symbols are used, not where they're defined
4. **`mock_db_session` from root conftest** — provides `commit`, `rollback`, `flush`, `execute`, `add`, `add_all`
5. **Assertions focus on side effects** — verify which functions were called, with what args, in what order
6. **No real DB, no real LLM** — deterministic, fast, repeatable

---

## 6. Patch Map (Quick Reference)

Every test will need a subset of these patches. This table helps identify what to mock per transition.

| Patch Target | Used In Transitions | Type |
|:-------------|:---------------------|:-----|
| `app.workers.orchestrator.process_extraction_job` | PENDING | AsyncMock |
| `app.workers.orchestrator.ContentFactoryVectorStore` | PENDING, RESEARCHING, SCRIPTING, FACT_CHECKING_SCRIPT | Mock returning `mock_vector_store` |
| `app.workers.orchestrator.update_job_status` | All transitions | AsyncMock |
| `app.workers.orchestrator.log_error` | Error scenarios | AsyncMock |
| `app.workers.orchestrator._web_search_service` | RESEARCHING | AsyncMock |
| `app.workers.orchestrator.ResearchAgent` | RESEARCHING | Mock class |
| `app.workers.orchestrator.CopywriterAgent` | SCRIPTING | Mock class |
| `app.workers.orchestrator.RedTeamAgent` | FACT_CHECKING_SCRIPT | Mock class |
| `app.workers.orchestrator.AssetStudioAgent` | ASSET_GENERATION | Mock class |
| `app.workers.orchestrator.get_latest_script` | SCRIPTING, FACT_CHECKING_SCRIPT | AsyncMock |
| `app.workers.orchestrator.save_script` | SCRIPTING | AsyncMock |
| `app.workers.orchestrator.append_script_feedback` | FACT_CHECKING_SCRIPT (revision) | AsyncMock |
| `app.workers.orchestrator.save_fact_check_claims` | FACT_CHECKING_SCRIPT | AsyncMock |
| `app.workers.orchestrator.cleanup_local_research_chunks` | COMPLETED | AsyncMock |
| `app.workers.orchestrator.settings` | FACT_CHECKING_SCRIPT (max_revisions) | Mock |

---

## 7. Agent Mocking Pattern

Since each agent's `__init__` calls `get_llm()` (which would fail without a real API key), we must patch the agent class itself and control `run()`:

```python
def _mock_agent_class(agent_result):
    """Create a mock agent class whose run() returns the given AgentResult."""
    mock_instance = AsyncMock()
    mock_instance.run = AsyncMock(return_value=agent_result)
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls
```

Usage in a test:

```python
result = AgentResult(
    status=AgentActionStatus.SUCCESS,
    payload={"chunks": ["chunk1", "chunk2"]},
    reasoning="Research complete",
    confidence_score=0.9,
)

with patch("app.workers.orchestrator.ResearchAgent", return_value=_mock_agent_instance(result)):
    await execute_state_transition(db, job)
```
---
## 8. Running the Tests

```bash
# Integration tests only
pytest -m integration -v

# Specific orchestrator tests
pytest tests/integration/test_orchestrator_transitions.py -v

# Single test class
pytest tests/integration/test_orchestrator_transitions.py::TestTransitionPending -v

# Single test
pytest tests/integration/test_orchestrator_transitions.py::TestTransitionPending::test_should_chunk_raw_text_and_ingest_as_raw_context -v
```

---

## 9. Known Risks & Mitigations

| Risk | Mitigation |
|:-----|:-----------|
| Agent `__init__` calls `get_llm()` which needs API key | Patch agent class at `app.workers.orchestrator.*` level, never instantiate real agents |
| `_web_search_service` is a module-level singleton | Patch `app.workers.orchestrator._web_search_service` directly |
| `update_job_status` commits the session | Mock it — we don't want real commits in integration tests |
| `save_fact_check_claims` uses `db.flush()` not `db.commit()` | `mock_db_session.flush` is already a no-op AsyncMock |
| `settings` is a module-level singleton | Patch specific attributes: `patch("app.workers.orchestrator.settings.max_red_team_revisions", 2)` |
| `ContentFactoryVectorStore.__init__` creates embedding instances | Patch the class, return `mock_vector_store` |
| Revision counter depends on `latest_script.version` | Control via `get_latest_script` mock returning a script with specific version |

---

## 10. What Comes After (Future Work)

| Next Step | Depends On |
|:----------|:-----------|
| Phase 4: DeepEval AI evaluation tests | This plan completes Phase 3 |
| Golden dataset population (20+ cases) | Eval tests need curated inputs |
| Real DB integration tests (Testcontainers) | When CI pipeline supports Docker services |
| Performance benchmarks on orchestrator | After integration tests are stable |
