import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.schemas.shorts import JobStatusEnum
from app.workers.orchestrator import execute_state_transition


@pytest.mark.integration
class TestTransitionResearchingSourceTypes:
    async def test_ingests_both_web_search_and_url_extract_with_distinct_source_types(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        mock_job.title = "BRICS 2025"
        mock_job.source_urls = ["https://example.com/source1", "https://example.com/source2"]

        web_results = [
            {"content": "Web search content about BRICS", "url": "https://news.example.com/article"},
        ]
        extracted_results = [
            {"content": "Extracted content from source URL", "url": "https://example.com/source1"},
            {"content": "More extracted content", "url": "https://example.com/source2"},
        ]

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ),
        ):
            mock_web.search = AsyncMock(return_value=web_results)
            mock_web.extract = AsyncMock(return_value=extracted_results)

            await execute_state_transition(mock_db_session, mock_job)

            ingest_calls = mock_vector_store.ingest_chunks.call_args_list
            assert len(ingest_calls) == 2

            web_ingest = [c for c in ingest_calls if c.kwargs.get("meta", {}).get("source_type") == "WEB_SEARCH"]
            url_extract_ingest = [c for c in ingest_calls if c.kwargs.get("meta", {}).get("source_type") == "URL_EXTRACT"]

            assert len(web_ingest) == 1
            assert web_ingest[0].kwargs["chunks"] == ["Web search content about BRICS"]
            assert web_ingest[0].kwargs["meta"]["urls"] == ["https://news.example.com/article"]

            assert len(url_extract_ingest) == 1
            assert len(url_extract_ingest[0].kwargs["chunks"]) == 2
            assert url_extract_ingest[0].kwargs["meta"]["urls"] == ["https://example.com/source1", "https://example.com/source2"]

    async def test_url_extract_not_called_when_no_source_urls(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        mock_job.title = "BRICS 2025"
        mock_job.source_urls = []

        web_results = [
            {"content": "Web search content", "url": "https://news.example.com"},
        ]

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ),
        ):
            mock_web.search = AsyncMock(return_value=web_results)
            mock_web.extract = AsyncMock()

            await execute_state_transition(mock_db_session, mock_job)

            mock_web.extract.assert_not_awaited()
            ingest_calls = mock_vector_store.ingest_chunks.call_args_list
            web_ingest = [c for c in ingest_calls if c.kwargs.get("meta", {}).get("source_type") == "WEB_SEARCH"]
            assert len(web_ingest) == 1

    async def test_url_extract_skipped_when_empty_results(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
    ):
        mock_job.status = JobStatusEnum.RESEARCHING
        mock_job.title = "BRICS 2025"
        mock_job.source_urls = ["https://example.com/source"]

        web_results = [
            {"content": "Web content", "url": "https://news.example.com"},
        ]

        with (
            patch("app.workers.orchestrator._web_search_service") as mock_web,
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workers.orchestrator.update_job_status", new_callable=AsyncMock
            ),
        ):
            mock_web.search = AsyncMock(return_value=web_results)
            mock_web.extract = AsyncMock(return_value=[])

            await execute_state_transition(mock_db_session, mock_job)

            mock_web.extract.assert_awaited_once_with(["https://example.com/source"])
            ingest_calls = mock_vector_store.ingest_chunks.call_args_list
            url_extract_ingest = [c for c in ingest_calls if c.kwargs.get("meta", {}).get("source_type") == "URL_EXTRACT"]
            assert len(url_extract_ingest) == 0