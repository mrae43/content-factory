import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4


TEST_DATABASE_URL = (
    "postgresql+asyncpg://postgres:postgres@localhost:5432/content_factory_test"
)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.semantic_search.return_value = [
        {
            "id": str(uuid4()),
            "content": "Test chunk content about BRICS GDP growth.",
            "meta": {"scope": "RAW-CONTEXT", "version": "1.0"},
            "job_id": str(uuid4()),
            "similarity_score": 0.92,
        }
    ]
    store.ingest_chunks.return_value = 1
    return store


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    return llm


@pytest.fixture
def mock_web_search_service():
    service = AsyncMock()
    service.search.return_value = [
        {
            "content": "BRICS nations announced a new payment system in 2025.",
            "url": "https://example.com/brics",
        }
    ]
    return service


@pytest.fixture
def sample_job_payload():
    return {
        "topic": "BRICS De-dollarization 2025",
        "pre_context": {
            "raw_text": "The BRICS nations announced a new payment system...",
            "source_urls": [],
            "target_audience": "Investors",
            "guardrail_strictness": "High",
        },
        "strict_compliance_mode": True,
    }


@pytest.fixture
def sample_research_chunks():
    return [
        "BRICS collective GDP grew 3.2% in 2024 according to IMF data.",
        "China's GDP growth was 5.2% in Q3 2024.",
        "The New Development Bank approved $3.2 billion in loans for infrastructure.",
    ]


@pytest.fixture
def sample_claim_data():
    return [
        {
            "claim_text": "GDP grew 15%",
            "verdict": "UNSUPPORTED",
            "confidence": 0.3,
            "evidence_text": "No GDP data found supporting 15% growth",
            "evidence_references": [],
        },
        {
            "claim_text": "BRICS announced a new payment system",
            "verdict": "SUPPORTED",
            "confidence": 0.95,
            "evidence_text": "BRICS nations announced a new payment system in 2025.",
            "evidence_references": [],
        },
    ]


@pytest.fixture
def mock_settings():
    with patch("app.core.config.Settings") as mock_cls:
        instance = MagicMock()
        instance.gemini_api_key = "test-gemini-key"
        instance.tavily_api_key = "test-tavily-key"
        instance.synthid_watermark_enabled = True
        instance.max_red_team_revisions = 3
        instance.similarity_threshold = 0.75
        instance.worker_poll_interval_seconds = 5
        instance.worker_lock_timeout_minutes = 15
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    return session


@pytest.fixture
def job_id():
    return uuid4()


@pytest.fixture
def override_settings():
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-gemini-key",
            "TAVILY_API_KEY": "test-tavily-key",
        },
    ):
        yield
