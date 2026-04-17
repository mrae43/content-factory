import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.workers.agents import AgentActionStatus, AgentResult
from app.schemas.shorts import JobStatusEnum


@pytest.fixture
def mock_job():
    job = MagicMock()
    job.id = uuid4()
    job.topic = "BRICS De-dollarization 2025"
    job.pre_context = {
        "raw_text": "The BRICS nations announced a new payment system..."
    }
    job.status = JobStatusEnum.PENDING
    job.final_video_url = None
    job.error_log = None
    return job


@pytest.fixture
def agent_result_success():
    def _make(payload: dict, reasoning: str = "OK", confidence: float = 0.9):
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload=payload,
            reasoning=reasoning,
            confidence_score=confidence,
        )

    return _make


@pytest.fixture
def agent_result_revision():
    def _make(payload: dict, reasoning: str = "Claims unsupported"):
        return AgentResult(
            status=AgentActionStatus.REVISION_NEEDED,
            payload=payload,
            reasoning=reasoning,
            confidence_score=0.5,
        )

    return _make


@pytest.fixture
def agent_result_escalate():
    def _make(reasoning: str = "No research sources available"):
        return AgentResult(
            status=AgentActionStatus.ESCALATE,
            payload={},
            reasoning=reasoning,
            confidence_score=0.0,
        )

    return _make


@pytest.fixture
def mock_script():
    script = MagicMock()
    script.id = uuid4()
    script.job_id = uuid4()
    script.version = 1
    script.content = "Did you know BRICS is reshaping global finance?"
    script.is_approved = False
    script.feedback_history = []
    return script


def _mock_agent_class(agent_result):
    mock_instance = AsyncMock()
    mock_instance.run = AsyncMock(return_value=agent_result)
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls
