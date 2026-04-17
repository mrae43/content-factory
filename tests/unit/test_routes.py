import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.api.routes import router
from app.db.session import get_db
from app.schemas.shorts import JobStatusEnum


def make_mock_job(**overrides):
    defaults = {
        "id": uuid4(),
        "topic": "BRICS De-dollarization 2025",
        "status": JobStatusEnum.PENDING,
        "strict_compliance_mode": True,
        "final_video_url": None,
        "error_log": None,
        "scripts": [],
        "assets": [],
        "created_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    job = MagicMock()
    for key, value in defaults.items():
        setattr(job, key, value)
    return job


def make_mock_script(**overrides):
    defaults = {
        "id": uuid4(),
        "version": 1,
        "content": "Test script content",
        "is_approved": False,
        "feedback_history": [],
        "claims": [],
        "created_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    script = MagicMock()
    for key, value in defaults.items():
        setattr(script, key, value)
    return script


def make_mock_claim(**overrides):
    defaults = {
        "id": uuid4(),
        "claim_text": "Test claim",
        "verdict": "SUPPORTED",
        "confidence": 0.95,
        "evidence_references": [],
    }
    defaults.update(overrides)
    claim = MagicMock()
    for key, value in defaults.items():
        setattr(claim, key, value)
    return claim


def make_mock_asset(**overrides):
    defaults = {
        "id": uuid4(),
        "asset_type": "VISUAL_VEO",
        "url_or_path": "s3://bucket/video.mp4",
        "render_meta": {},
        "created_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    asset = MagicMock()
    for key, value in defaults.items():
        setattr(asset, key, value)
    return asset


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def test_app(mock_db):
    app = FastAPI()
    app.include_router(router)

    async def _override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest.fixture
async def client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.unit
class TestCreateRenderJob:
    async def test_should_return_202_with_correct_response(
        self, client, mock_db, sample_job_payload
    ):
        mock_job = make_mock_job(topic=sample_job_payload["topic"])
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = mock_job
        mock_db.execute.return_value = mock_result

        resp = await client.post("/api/v1/jobs/", json=sample_job_payload)

        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "PENDING"
        assert data["topic"] == "BRICS De-dollarization 2025"
        assert data["strict_compliance_mode"] is True
        assert "id" in data
        assert data["final_video_url"] is None
        assert data["error_log"] is None
        assert data["scripts"] == []
        assert data["assets"] == []
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()


@pytest.mark.unit
class TestGetRenderJob:
    async def test_should_return_200_with_full_audit_trail(self, client, mock_db):
        claim = make_mock_claim()
        script = make_mock_script(claims=[claim])
        asset = make_mock_asset()
        job = make_mock_job(
            status=JobStatusEnum.FACT_CHECKING_SCRIPT,
            scripts=[script],
            assets=[asset],
        )

        mock_result = MagicMock()
        mock_result.unique.return_value.scalar_one_or_none.return_value = job
        mock_db.execute.return_value = mock_result

        resp = await client.get(f"/api/v1/jobs/{job.id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == str(job.id)
        assert data["status"] == "FACT_CHECKING_SCRIPT"
        assert len(data["scripts"]) == 1
        assert data["scripts"][0]["version"] == 1
        assert len(data["scripts"][0]["claims"]) == 1
        assert data["scripts"][0]["claims"][0]["verdict"] == "SUPPORTED"
        assert len(data["assets"]) == 1
        assert data["assets"][0]["asset_type"] == "VISUAL_VEO"

    async def test_should_return_404_when_job_not_found(self, client, mock_db):
        mock_result = MagicMock()
        mock_result.unique.return_value.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        resp = await client.get(f"/api/v1/jobs/{uuid4()}")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


@pytest.mark.unit
class TestApproveScript:
    async def test_should_transition_to_asset_generation_on_approve(
        self, client, mock_db
    ):
        script = make_mock_script()
        job = make_mock_job(
            status=JobStatusEnum.HUMAN_REVIEW_NEEDED,
            scripts=[script],
        )

        mock_result_1 = MagicMock()
        mock_result_1.unique.return_value.scalar_one_or_none.return_value = job
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one.return_value = job
        mock_db.execute.side_effect = [mock_result_1, mock_result_2]

        resp = await client.post(
            f"/api/v1/jobs/{job.id}/approve-script",
            json={"is_approved": True},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "ASSET_GENERATION"
        assert script.is_approved is True
        mock_db.commit.assert_awaited()

    async def test_should_transition_to_scripting_on_reject_with_feedback(
        self, client, mock_db
    ):
        script = make_mock_script(feedback_history=[])
        job = make_mock_job(
            status=JobStatusEnum.FACT_CHECKING_SCRIPT,
            scripts=[script],
        )

        mock_result_1 = MagicMock()
        mock_result_1.unique.return_value.scalar_one_or_none.return_value = job
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one.return_value = job
        mock_db.execute.side_effect = [mock_result_1, mock_result_2]

        resp = await client.post(
            f"/api/v1/jobs/{job.id}/approve-script",
            json={"is_approved": False, "human_feedback": "Needs more sources"},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "SCRIPTING"
        assert len(script.feedback_history) == 1
        assert script.feedback_history[0]["source"] == "human_editor"
        assert script.feedback_history[0]["comment"] == "Needs more sources"

    async def test_should_return_400_when_job_in_wrong_status(self, client, mock_db):
        job = make_mock_job(status=JobStatusEnum.PENDING)

        mock_result = MagicMock()
        mock_result.unique.return_value.scalar_one_or_none.return_value = job
        mock_db.execute.return_value = mock_result

        resp = await client.post(
            f"/api/v1/jobs/{job.id}/approve-script",
            json={"is_approved": True},
        )

        assert resp.status_code == 400
        assert "cannot approve script" in resp.json()["detail"].lower()

    async def test_should_return_400_when_no_scripts_exist(self, client, mock_db):
        job = make_mock_job(
            status=JobStatusEnum.HUMAN_REVIEW_NEEDED,
            scripts=[],
        )

        mock_result = MagicMock()
        mock_result.unique.return_value.scalar_one_or_none.return_value = job
        mock_db.execute.return_value = mock_result

        resp = await client.post(
            f"/api/v1/jobs/{job.id}/approve-script",
            json={"is_approved": True},
        )

        assert resp.status_code == 400
        assert "no script" in resp.json()["detail"].lower()
