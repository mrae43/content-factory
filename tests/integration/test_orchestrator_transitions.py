import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.workers.agents import AgentActionStatus, AgentResult
from app.schemas.shorts import JobStatusEnum
from app.workers.orchestrator import execute_state_transition
from tests.integration.conftest import _mock_agent_class


@pytest.mark.integration
class TestTransitionFactCheckingResearch:
    async def test_should_pass_through_to_scripting(self, mock_db_session, mock_job):
        mock_job.status = JobStatusEnum.FACT_CHECKING_RESEARCH

        with patch(
            "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
        ) as mock_update:
            await execute_state_transition(mock_db_session, mock_job)

            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.SCRIPTING
            )


@pytest.mark.integration
class TestTransitionCompleted:
    async def test_should_cleanup_local_chunks_on_completion(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.COMPLETED

        with patch(
            "app.workers.orchestrator.cleanup_local_research_chunks",
            new_callable=AsyncMock,
        ) as mock_cleanup:
            await execute_state_transition(mock_db_session, mock_job)

            mock_cleanup.assert_awaited_once_with(mock_job.id, mock_db_session)


@pytest.mark.integration
class TestTerminalStates:
    async def test_should_not_transition_from_human_review_needed(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.HUMAN_REVIEW_NEEDED

        with (
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_update.assert_not_awaited()
            mock_log.assert_not_awaited()

    async def test_should_not_transition_from_failed(self, mock_db_session, mock_job):
        mock_job.status = JobStatusEnum.FAILED

        with (
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_update.assert_not_awaited()
            mock_log.assert_not_awaited()


@pytest.mark.integration
class TestTransitionPending:
    async def test_should_chunk_raw_text_and_ingest_as_raw_context(
        self, mock_db_session, mock_job, mock_vector_store
    ):
        mock_job.status = JobStatusEnum.PENDING
        mock_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

        with (
            patch(
                "app.workers.orchestrator.process_extraction_job",
                new_callable=AsyncMock,
            ) as mock_chunking,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_chunking.return_value = mock_chunks

            await execute_state_transition(mock_db_session, mock_job)

            mock_chunking.assert_awaited_once_with(
                str(mock_job.id), mock_job.pre_context["raw_text"]
            )
            mock_vector_store.ingest_chunks.assert_awaited_once()
            ingest_call = mock_vector_store.ingest_chunks.call_args
            assert ingest_call.kwargs["scope"] == "RAW-CONTEXT"
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.RESEARCHING
            )

    async def test_should_handle_dict_pre_context(
        self, mock_db_session, mock_job, mock_vector_store
    ):
        mock_job.status = JobStatusEnum.PENDING
        mock_job.pre_context = {
            "raw_text": "Some custom raw text content",
            "source_urls": [],
        }

        with (
            patch(
                "app.workers.orchestrator.process_extraction_job",
                new_callable=AsyncMock,
            ) as mock_chunking,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
        ):
            mock_chunking.return_value = ["chunk"]

            await execute_state_transition(mock_db_session, mock_job)

            mock_chunking.assert_awaited_once_with(
                str(mock_job.id), "Some custom raw text content"
            )

    async def test_should_handle_string_pre_context(
        self, mock_db_session, mock_job, mock_vector_store
    ):
        mock_job.status = JobStatusEnum.PENDING
        mock_job.pre_context = "Just a plain string"

        with (
            patch(
                "app.workers.orchestrator.process_extraction_job",
                new_callable=AsyncMock,
            ) as mock_chunking,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
        ):
            mock_chunking.return_value = ["chunk"]

            await execute_state_transition(mock_db_session, mock_job)

            mock_chunking.assert_awaited_once_with(
                str(mock_job.id), "Just a plain string"
            )

    async def test_should_transition_to_researching_even_with_no_chunks(
        self, mock_db_session, mock_job, mock_vector_store
    ):
        mock_job.status = JobStatusEnum.PENDING

        with (
            patch(
                "app.workers.orchestrator.process_extraction_job",
                new_callable=AsyncMock,
            ) as mock_chunking,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_chunking.return_value = []

            await execute_state_transition(mock_db_session, mock_job)

            mock_vector_store.ingest_chunks.assert_not_awaited()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.RESEARCHING
            )

    async def test_should_transition_to_failed_on_chunking_exception(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.PENDING

        with (
            patch(
                "app.workers.orchestrator.process_extraction_job",
                new_callable=AsyncMock,
                side_effect=Exception("Chunking blew up"),
            ),
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            assert "Chunking blew up" in mock_log.call_args[0][2]
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )


@pytest.mark.integration
class TestTransitionResearching:
    async def test_should_search_web_ingest_and_run_research_agent(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        web_results = [
            {
                "content": "BRICS nations announced a new payment system.",
                "url": "https://example.com/brics",
            }
        ]
        result = agent_result_success(payload={"chunks": ["research chunk"]})

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.ResearchAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_web.search = AsyncMock(return_value=web_results)

            await execute_state_transition(mock_db_session, mock_job)

            mock_web.search.assert_awaited_once_with(mock_job.topic)
            ingest_calls = mock_vector_store.ingest_chunks.call_args_list
            web_ingest = [c for c in ingest_calls if c.kwargs.get("scope") == "LOCAL"]
            assert len(web_ingest) == 1
            meta = web_ingest[0].kwargs["meta"]
            assert meta["source"] == "web_search"
            assert meta["query"] == mock_job.topic
            assert meta["urls"] == ["https://example.com/brics"]
            assert meta["search_depth"] == "basic"
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FACT_CHECKING_RESEARCH
            )

    async def test_should_proceed_without_web_results(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        result = agent_result_success(payload={"chunks": ["chunk"]})

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.ResearchAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_web.search = AsyncMock(return_value=[])

            await execute_state_transition(mock_db_session, mock_job)

            local_ingests = [
                c
                for c in mock_vector_store.ingest_chunks.call_args_list
                if c.kwargs.get("scope") == "LOCAL"
            ]
            assert len(local_ingests) == 0
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FACT_CHECKING_RESEARCH
            )

    async def test_should_filter_web_results_with_empty_content(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        web_results = [
            {"content": "Valid content", "url": "https://a.com"},
            {"content": "", "url": "https://b.com"},
            {"content": "Also valid", "url": "https://c.com"},
        ]
        result = agent_result_success(payload={"chunks": ["chunk"]})

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.ResearchAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
        ):
            mock_web.search = AsyncMock(return_value=web_results)

            await execute_state_transition(mock_db_session, mock_job)

            local_ingests = [
                c
                for c in mock_vector_store.ingest_chunks.call_args_list
                if c.kwargs.get("scope") == "LOCAL"
            ]
            assert len(local_ingests) == 1
            assert local_ingests[0].kwargs["chunks"] == [
                "Valid content",
                "Also valid",
            ]

    async def test_should_fail_when_research_agent_returns_error(
        self, mock_db_session, mock_job, mock_vector_store
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        error_result = AgentResult(
            status=AgentActionStatus.ERROR,
            payload={},
            reasoning="No context found",
            confidence_score=0.0,
        )

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.ResearchAgent",
                return_value=_mock_agent_class(error_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_web.search = AsyncMock(return_value=[])

            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )

    async def test_should_fail_when_research_agent_raises_exception(
        self, mock_db_session, mock_job, mock_vector_store
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("API timeout"))

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch("app.workers.orchestrator.ResearchAgent", return_value=mock_agent),
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_web.search = AsyncMock(return_value=[])

            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )


@pytest.mark.integration
class TestTransitionScripting:
    async def test_should_run_copywriter_and_save_script(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.SCRIPTING
        result = agent_result_success(
            payload={"script_content": "New script content", "storyboard": []}
        )

        with (
            patch(
                "app.workers.orchestrator.CopywriterAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_script", new_callable=AsyncMock
            ) as mock_save,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_get_script.return_value = None

            await execute_state_transition(mock_db_session, mock_job)

            mock_save.assert_awaited_once_with(
                mock_db_session, mock_job.id, "New script content", 1
            )
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FACT_CHECKING_SCRIPT
            )

    async def test_should_increment_version_for_revision(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.SCRIPTING
        mock_script.version = 2
        result = agent_result_success(
            payload={"script_content": "Revised script", "storyboard": []}
        )

        with (
            patch(
                "app.workers.orchestrator.CopywriterAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_script", new_callable=AsyncMock
            ) as mock_save,
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_save.assert_awaited_once_with(
                mock_db_session, mock_job.id, "Revised script", 3
            )

    async def test_should_pass_revision_feedback_to_agent(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.SCRIPTING
        mock_script.feedback_history = ["Needs more data", "Add sources"]
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(
            return_value=agent_result_success(
                payload={"script_content": "Fixed script", "storyboard": []}
            )
        )

        with (
            patch(
                "app.workers.orchestrator.CopywriterAgent",
                return_value=mock_agent_instance,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch("app.workers.orchestrator.save_script", new_callable=AsyncMock),
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            call_kwargs = mock_agent_instance.run.call_args.kwargs
            assert call_kwargs["context"]["feedback"] == "Add sources"

    async def test_should_handle_dict_feedback_history(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.SCRIPTING
        mock_script.feedback_history = [
            {"source": "human_editor", "feedback": "Tone too aggressive"}
        ]
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(
            return_value=agent_result_success(
                payload={"script_content": "Toned down", "storyboard": []}
            )
        )

        with (
            patch(
                "app.workers.orchestrator.CopywriterAgent",
                return_value=mock_agent_instance,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch("app.workers.orchestrator.save_script", new_callable=AsyncMock),
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            call_kwargs = mock_agent_instance.run.call_args.kwargs
            assert call_kwargs["context"]["feedback"] == "Tone too aggressive"

    async def test_should_fail_when_copywriter_returns_error(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
    ):
        mock_job.status = JobStatusEnum.SCRIPTING
        error_result = AgentResult(
            status=AgentActionStatus.ERROR,
            payload={},
            reasoning="No research chunks found",
            confidence_score=0.0,
        )

        with (
            patch(
                "app.workers.orchestrator.CopywriterAgent",
                return_value=_mock_agent_class(error_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch("app.workers.orchestrator.get_latest_script", new_callable=AsyncMock),
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )


@pytest.mark.integration
class TestTransitionFactCheckingScript:
    async def test_should_approve_when_all_claims_supported(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        claims_data = [
            {
                "claim_text": "BRICS announced new system",
                "verdict": "SUPPORTED",
                "confidence": 0.95,
                "evidence_text": "BRICS nations announced...",
                "evidence_references": [str(uuid4())],
            }
        ]
        result = agent_result_success(
            payload={"claims": claims_data, "verdict": "SUPPORTED"}
        )

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ) as mock_save_claims,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_save_claims.assert_awaited_once_with(
                mock_db_session, mock_script.id, claims_data
            )
            assert mock_script.is_approved is True
            mock_db_session.commit.assert_awaited()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.ASSET_GENERATION
            )

    async def test_should_approve_with_empty_claims(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        result = agent_result_success(payload={"claims": [], "verdict": "SUPPORTED"})

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ) as mock_save_claims,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_save_claims.assert_not_awaited()
            assert mock_script.is_approved is True
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.ASSET_GENERATION
            )

    async def test_should_reject_and_loop_to_scripting(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_revision,
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        mock_script.version = 1
        claims_data = [
            {
                "claim_text": "GDP grew 15%",
                "verdict": "UNSUPPORTED",
                "confidence": 0.3,
                "evidence_text": "No evidence",
                "evidence_references": [],
            }
        ]
        result = agent_result_revision(
            payload={"claims": claims_data, "verdict": "UNSUPPORTED"},
            reasoning="Claims unsupported",
        )

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.append_script_feedback",
                new_callable=AsyncMock,
            ) as mock_append_feedback,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_append_feedback.assert_awaited_once_with(
                mock_db_session, mock_job.id, "Claims unsupported"
            )
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.SCRIPTING
            )

    async def test_should_escalate_after_max_revisions(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_revision,
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        mock_script.version = 3
        result = agent_result_revision(
            payload={"claims": [], "verdict": "UNSUPPORTED"},
            reasoning="Still bad",
        )

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.append_script_feedback",
                new_callable=AsyncMock,
            ) as mock_append_feedback,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_append_feedback.assert_not_awaited()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED
            )

    async def test_should_escalate_after_max_revisions_custom(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_revision,
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        mock_script.version = 2
        result = agent_result_revision(
            payload={"claims": [], "verdict": "UNSUPPORTED"},
            reasoning="Still bad",
        )

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.append_script_feedback",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
            patch("app.workers.orchestrator.settings") as mock_settings,
        ):
            mock_get_script.return_value = mock_script
            mock_settings.max_red_team_revisions = 2

            await execute_state_transition(mock_db_session, mock_job)

            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED
            )

    async def test_should_escalate_when_red_team_returns_escalate(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_escalate,
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        result = agent_result_escalate()

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ) as mock_save_claims,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_save_claims.assert_not_awaited()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.HUMAN_REVIEW_NEEDED
            )

    async def test_should_resolve_evidence_references(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        chunk_id = uuid4()
        mock_vector_store.semantic_search.return_value = [
            {
                "id": chunk_id,
                "content": "Evidence text",
                "similarity_score": 0.85,
            }
        ]
        claims_data = [
            {
                "claim_text": "Claim A",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence_text": "Evidence text",
            }
        ]
        result = agent_result_success(
            payload={"claims": claims_data, "verdict": "SUPPORTED"}
        )

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
            patch("app.workers.orchestrator.settings") as mock_settings,
        ):
            mock_get_script.return_value = mock_script
            mock_settings.similarity_threshold = 0.75

            await execute_state_transition(mock_db_session, mock_job)

            mock_vector_store.semantic_search.assert_awaited_once_with(
                query="Evidence text",
                job_id=mock_job.id,
                scopes=["RAW-CONTEXT", "LOCAL"],
                top_k=3,
            )
            assert claims_data[0]["evidence_references"] == [str(chunk_id)]

    async def test_should_not_resolve_refs_for_empty_evidence(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        claims_data = [
            {
                "claim_text": "Claim A",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "evidence_text": "",
            }
        ]
        result = agent_result_success(
            payload={"claims": claims_data, "verdict": "SUPPORTED"}
        )

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_vector_store.semantic_search.assert_not_awaited()
            assert claims_data[0]["evidence_references"] == []


@pytest.mark.integration
class TestTransitionAssetGeneration:
    async def test_should_set_video_url_and_complete(
        self,
        mock_db_session,
        mock_job,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.ASSET_GENERATION
        result = agent_result_success(
            payload={"video_url": "s3://factory/renders/test_rendered.mp4"}
        )

        with (
            patch(
                "app.workers.orchestrator.AssetStudioAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            assert mock_job.final_video_url == (
                "s3://factory/renders/test_rendered.mp4"
            )
            mock_db_session.commit.assert_awaited()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.COMPLETED
            )

    async def test_should_not_transition_on_agent_failure(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.ASSET_GENERATION
        error_result = AgentResult(
            status=AgentActionStatus.ERROR,
            payload={},
            reasoning="Asset generation failed",
            confidence_score=0.0,
        )

        with (
            patch(
                "app.workers.orchestrator.AssetStudioAgent",
                return_value=_mock_agent_class(error_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            assert mock_job.final_video_url is None
            mock_update.assert_not_awaited()


@pytest.mark.integration
class TestOrchestratorErrorHandling:
    async def test_should_catch_any_exception_and_fail_job(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.PENDING

        with (
            patch(
                "app.workers.orchestrator.process_extraction_job",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Unexpected"),
            ),
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            error_msg = mock_log.call_args[0][2]
            assert "Unexpected" in error_msg
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )

    async def test_should_log_phase_in_error(
        self, mock_db_session, mock_job, mock_vector_store
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("API error"))

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch("app.workers.orchestrator.ResearchAgent", return_value=mock_agent),
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
            patch("app.workers.orchestrator.update_job_status", new_callable=AsyncMock),
        ):
            mock_web.search = AsyncMock(return_value=[])

            await execute_state_transition(mock_db_session, mock_job)

            phase = mock_log.call_args[1].get("phase") or mock_log.call_args[0][3]
            assert phase == str(JobStatusEnum.RESEARCHING)

    async def test_should_handle_unrecognized_status(self, mock_db_session, mock_job):
        mock_job.status = "UNKNOWN_STATUS"

        with (
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
            patch(
                "app.workers.orchestrator.log_error", new_callable=AsyncMock
            ) as mock_log,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_update.assert_not_awaited()
            mock_log.assert_not_awaited()


@pytest.mark.integration
class TestOrchestratorMultiStep:
    async def test_full_happy_path_pending_to_completed(
        self, mock_db_session, mock_job, mock_vector_store, mock_script
    ):
        research_result = AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"chunks": ["research chunk"]},
            reasoning="Done",
            confidence_score=0.9,
        )
        copy_result = AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"script_content": "Script v1", "storyboard": []},
            reasoning="Done",
            confidence_score=0.9,
        )
        red_team_result = AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"claims": [], "verdict": "SUPPORTED"},
            reasoning="All good",
            confidence_score=0.95,
        )
        asset_result = AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"video_url": "s3://factory/renders/final.mp4"},
            reasoning="Done",
            confidence_score=0.9,
        )

        with (
            patch(
                "app.workers.orchestrator.process_extraction_job",
                new_callable=AsyncMock,
                return_value=["chunk1"],
            ),
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.ResearchAgent",
                return_value=_mock_agent_class(research_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.CopywriterAgent",
                return_value=_mock_agent_class(copy_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(red_team_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.AssetStudioAgent",
                return_value=_mock_agent_class(asset_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch("app.workers.orchestrator.save_script", new_callable=AsyncMock),
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.cleanup_local_research_chunks",
                new_callable=AsyncMock,
            ) as mock_cleanup,
        ):
            mock_web.search = AsyncMock(return_value=[])
            mock_get_script.return_value = mock_script

            transitions = [
                (JobStatusEnum.PENDING, JobStatusEnum.RESEARCHING),
                (JobStatusEnum.RESEARCHING, JobStatusEnum.FACT_CHECKING_RESEARCH),
                (
                    JobStatusEnum.FACT_CHECKING_RESEARCH,
                    JobStatusEnum.SCRIPTING,
                ),
                (JobStatusEnum.SCRIPTING, JobStatusEnum.FACT_CHECKING_SCRIPT),
                (
                    JobStatusEnum.FACT_CHECKING_SCRIPT,
                    JobStatusEnum.ASSET_GENERATION,
                ),
                (JobStatusEnum.ASSET_GENERATION, JobStatusEnum.COMPLETED),
            ]

            for current, expected in transitions:
                mock_job.status = current
                await execute_state_transition(mock_db_session, mock_job)

            assert mock_update.call_count == len(transitions)
            for i, (_, expected) in enumerate(transitions):
                assert mock_update.call_args_list[i].args == (
                    mock_db_session,
                    mock_job.id,
                    expected,
                )

            mock_job.status = JobStatusEnum.COMPLETED
            await execute_state_transition(mock_db_session, mock_job)
            mock_cleanup.assert_awaited_once_with(mock_job.id, mock_db_session)

    async def test_revision_loop_with_eventual_approval(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
    ):
        revision_result = AgentResult(
            status=AgentActionStatus.REVISION_NEEDED,
            payload={"claims": [], "verdict": "UNSUPPORTED"},
            reasoning="Needs work",
            confidence_score=0.5,
        )
        approved_result = AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"claims": [], "verdict": "SUPPORTED"},
            reasoning="All good now",
            confidence_score=0.9,
        )
        copy_result = AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"script_content": "Script", "storyboard": []},
            reasoning="Done",
            confidence_score=0.9,
        )
        asset_result = AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"video_url": "s3://factory/renders/final.mp4"},
            reasoning="Done",
            confidence_score=0.9,
        )

        script_v1 = MagicMock()
        script_v1.version = 1
        script_v1.id = uuid4()
        script_v1.content = "Script v1"
        script_v1.is_approved = False
        script_v1.feedback_history = []

        script_v2 = MagicMock()
        script_v2.version = 2
        script_v2.id = uuid4()
        script_v2.content = "Script v2"
        script_v2.is_approved = False
        script_v2.feedback_history = []

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(revision_result).return_value,
            ) as mock_red_team_cls,
            patch(
                "app.workers.orchestrator.CopywriterAgent",
                return_value=_mock_agent_class(copy_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.AssetStudioAgent",
                return_value=_mock_agent_class(asset_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch("app.workers.orchestrator.save_script", new_callable=AsyncMock),
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.append_script_feedback",
                new_callable=AsyncMock,
            ) as mock_append_feedback,
            patch("app.workers.orchestrator.settings") as mock_settings,
        ):
            mock_settings.max_red_team_revisions = 3
            mock_red_team_instance_reject = AsyncMock()
            mock_red_team_instance_reject.run = AsyncMock(return_value=revision_result)
            mock_red_team_instance_approve = AsyncMock()
            mock_red_team_instance_approve.run = AsyncMock(return_value=approved_result)
            mock_red_team_cls.side_effect = [
                mock_red_team_instance_reject,
                mock_red_team_instance_approve,
            ]

            mock_get_script.side_effect = [script_v1, script_v1, script_v1, script_v2]

            mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
            await execute_state_transition(mock_db_session, mock_job)

            mock_append_feedback.assert_awaited_once_with(
                mock_db_session, mock_job.id, "Needs work"
            )
            assert mock_update.call_args_list[0].args == (
                mock_db_session,
                mock_job.id,
                JobStatusEnum.SCRIPTING,
            )

            mock_job.status = JobStatusEnum.SCRIPTING
            await execute_state_transition(mock_db_session, mock_job)

            mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
            await execute_state_transition(mock_db_session, mock_job)

            assert script_v2.is_approved is True

    async def test_revision_loop_hits_max_and_escalates(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
    ):
        revision_result = AgentResult(
            status=AgentActionStatus.REVISION_NEEDED,
            payload={"claims": [], "verdict": "UNSUPPORTED"},
            reasoning="Still bad",
            confidence_score=0.3,
        )

        scripts = []
        for v in [1, 2, 3]:
            s = MagicMock()
            s.version = v
            s.id = uuid4()
            s.content = f"Script v{v}"
            s.is_approved = False
            s.feedback_history = []
            scripts.append(s)

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(revision_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ) as mock_update,
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.append_script_feedback",
                new_callable=AsyncMock,
            ) as mock_append_feedback,
        ):
            for i, script in enumerate(scripts):
                mock_get_script.return_value = script
                mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT

                await execute_state_transition(mock_db_session, mock_job)

            assert mock_update.call_args_list[-1].args == (
                mock_db_session,
                mock_job.id,
                JobStatusEnum.HUMAN_REVIEW_NEEDED,
            )

            total_feedback_calls = mock_append_feedback.call_count
            assert total_feedback_calls == 2

            total_updates = mock_update.call_count
            assert total_updates == 3
