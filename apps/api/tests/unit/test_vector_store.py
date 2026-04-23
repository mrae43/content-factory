import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.services.vector_store import ContentFactoryVectorStore


@pytest.fixture
def job_id():
    return uuid4()


@pytest.fixture
def mock_db_session():
    db = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.add_all = MagicMock()
    return db


def _make_chunk_mock(chunk_id=None, job_id=None, content="text", meta=None):
    chunk = MagicMock()
    chunk.id = chunk_id or uuid4()
    chunk.job_id = job_id or uuid4()
    chunk.content = content
    chunk.meta = meta or {"scope": "LOCAL", "version": "1.0"}
    return chunk


def _create_store(mock_db_session):
    with (
        patch("app.services.vector_store.get_query_embeddings") as mock_get_qemb,
        patch("app.services.vector_store.get_embeddings") as mock_get_emb,
    ):
        mock_embedder = AsyncMock()
        mock_embedder.aembed_documents = AsyncMock(
            return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768]
        )
        mock_get_emb.return_value = mock_embedder

        mock_query_embedder = AsyncMock()
        mock_query_embedder.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock_get_qemb.return_value = mock_query_embedder

        store = ContentFactoryVectorStore(mock_db_session)
        store.embedder = mock_embedder
        store.query_embedder = mock_query_embedder

    return store


@pytest.mark.unit
class TestIngestChunks:
    async def test_should_return_zero_and_skip_db_when_chunks_empty(
        self, mock_db_session, job_id
    ):
        store = _create_store(mock_db_session)

        result = await store.ingest_chunks(job_id, [])

        assert result == 0
        mock_db_session.add_all.assert_not_called()
        mock_db_session.commit.assert_not_awaited()

    async def test_should_embed_and_ingest_all_chunks(self, mock_db_session, job_id):
        store = _create_store(mock_db_session)
        chunks = [
            "BRICS collective GDP grew 3.2% in 2024.",
            "China's GDP growth was 5.2% in Q3 2024.",
            "The New Development Bank approved $3.2 billion.",
        ]

        result = await store.ingest_chunks(job_id, chunks)

        assert result == 3
        store.embedder.aembed_documents.assert_awaited_once_with(chunks)
        mock_db_session.add_all.assert_called_once()
        added = mock_db_session.add_all.call_args[0][0]
        assert len(added) == 3
        mock_db_session.commit.assert_awaited_once()

    async def test_should_set_default_meta_scope_and_version(
        self, mock_db_session, job_id
    ):
        store = _create_store(mock_db_session)

        await store.ingest_chunks(job_id, ["chunk text"])

        added = mock_db_session.add_all.call_args[0][0]
        assert added[0].meta == {"scope": "LOCAL", "version": "1.0"}

    async def test_should_merge_custom_meta_into_defaults(
        self, mock_db_session, job_id
    ):
        store = _create_store(mock_db_session)

        await store.ingest_chunks(
            job_id,
            ["chunk text"],
            meta={"source": "tavily"},
        )

        added = mock_db_session.add_all.call_args[0][0]
        assert added[0].meta == {
            "scope": "LOCAL",
            "version": "1.0",
            "source": "tavily",
        }


@pytest.mark.unit
class TestSemanticSearch:
    def _setup_search_results(self, mock_db_session, rows):
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        mock_db_session.execute.return_value = mock_result

    async def test_should_return_filtered_results_above_threshold(
        self, mock_db_session, job_id
    ):
        store = _create_store(mock_db_session)

        chunk_a = _make_chunk_mock(job_id=job_id, content="result A")
        chunk_b = _make_chunk_mock(job_id=job_id, content="result B")
        chunk_c = _make_chunk_mock(job_id=job_id, content="result C")
        rows = [
            (chunk_a, 0.92),
            (chunk_b, 0.78),
            (chunk_c, 0.60),
        ]
        self._setup_search_results(mock_db_session, rows)

        with patch("app.services.vector_store.settings") as mock_settings:
            mock_settings.similarity_threshold = 0.75
            results = await store.semantic_search("test query", job_id=job_id)

        assert len(results) == 2
        assert results[0]["content"] == "result A"
        assert results[0]["similarity_score"] == 0.92
        assert results[1]["content"] == "result B"
        assert results[1]["similarity_score"] == 0.78
        assert "id" in results[0]
        assert "meta" in results[0]
        assert "job_id" in results[0]

    async def test_should_return_empty_when_all_results_below_threshold(
        self, mock_db_session, job_id
    ):
        store = _create_store(mock_db_session)

        chunk = _make_chunk_mock(job_id=job_id)
        rows = [(chunk, 0.40), (chunk, 0.50)]
        self._setup_search_results(mock_db_session, rows)

        with patch("app.services.vector_store.settings") as mock_settings:
            mock_settings.similarity_threshold = 0.75
            results = await store.semantic_search("test query", job_id=job_id)

        assert results == []

    async def test_should_use_scopes_over_scope_when_both_provided(
        self, mock_db_session, job_id
    ):
        store = _create_store(mock_db_session)
        self._setup_search_results(mock_db_session, [])

        with patch("app.services.vector_store.settings") as mock_settings:
            mock_settings.similarity_threshold = 0.75
            await store.semantic_search(
                "test query",
                job_id=job_id,
                scope="A",
                scopes=["B", "C"],
            )

        call_args = mock_db_session.execute.call_args
        compiled = str(call_args[0][0].compile(compile_kwargs={"literal_binds": True}))
        assert "B" in compiled or "IN" in compiled.upper()

    async def test_should_filter_by_single_scope_when_no_scopes(
        self, mock_db_session, job_id
    ):
        store = _create_store(mock_db_session)
        self._setup_search_results(mock_db_session, [])

        with patch("app.services.vector_store.settings") as mock_settings:
            mock_settings.similarity_threshold = 0.75
            await store.semantic_search(
                "test query",
                job_id=job_id,
                scope="RAW-CONTEXT",
            )

        mock_db_session.execute.assert_awaited_once()

    async def test_should_use_settings_threshold_when_none_passed(
        self, mock_db_session, job_id
    ):
        store = _create_store(mock_db_session)

        chunk_high = _make_chunk_mock(job_id=job_id, content="high")
        chunk_mid = _make_chunk_mock(job_id=job_id, content="mid")
        rows = [(chunk_high, 0.76), (chunk_mid, 0.74)]
        self._setup_search_results(mock_db_session, rows)

        with patch("app.services.vector_store.settings") as mock_settings:
            mock_settings.similarity_threshold = 0.75
            results = await store.semantic_search(
                "test query",
                job_id=job_id,
            )

        assert len(results) == 1
        assert results[0]["content"] == "high"
