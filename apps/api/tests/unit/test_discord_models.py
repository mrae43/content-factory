import pytest
from uuid import uuid4

from app.db.discord_models import ScriptJob, FormatJob
from app.schemas.shorts import ScriptJobStatusEnum, FormatJobStatusEnum


@pytest.mark.unit
class TestScriptJobModel:
    def test_creates_with_defaults(self):
        job_id = uuid4()
        job = ScriptJob(
            id=job_id,
            title="Test Script",
            status=ScriptJobStatusEnum.PENDING.value,
        )
        assert job.id == job_id
        assert job.title == "Test Script"
        assert job.status == ScriptJobStatusEnum.PENDING.value
        assert job.user_reference is None
        assert job.source_urls is None
        assert job.story_directives is None
        assert job.refined_context is None
        assert job.assembled_context is None
        assert job.script_content is None
        assert job.claims is None
        assert job.working_memory is None
        assert job.hedge_index is None
        assert job.error_log is None
        assert job.locked_at is None
        assert job.locked_by is None
        assert job.id is not None

    def test_status_enum_values(self):
        assert ScriptJobStatusEnum.PENDING == "PENDING"
        assert ScriptJobStatusEnum.RESEARCHING == "RESEARCHING"
        assert ScriptJobStatusEnum.RETRIEVAL == "RETRIEVAL"
        assert ScriptJobStatusEnum.SCRIPTING == "SCRIPTING"
        assert ScriptJobStatusEnum.FACT_CHECKING_SCRIPT == "FACT_CHECKING_SCRIPT"
        assert ScriptJobStatusEnum.COMPLETED == "COMPLETED"
        assert ScriptJobStatusEnum.FAILED == "FAILED"
        assert ScriptJobStatusEnum.HUMAN_REVIEW_NEEDED == "HUMAN_REVIEW_NEEDED"

    def test_can_set_status(self):
        job = ScriptJob(title="Test", status=ScriptJobStatusEnum.SCRIPTING.value)
        assert job.status == ScriptJobStatusEnum.SCRIPTING.value

    def test_can_store_working_memory(self):
        job = ScriptJob(title="Test", working_memory={"discord_thread_id": "12345"})
        assert job.working_memory["discord_thread_id"] == "12345"

    def test_can_store_error_log(self):
        job = ScriptJob(
            title="Test",
            error_log={"pipeline": {"message": "boom", "timestamp": "t"}},
        )
        assert job.error_log["pipeline"]["message"] == "boom"


@pytest.mark.unit
class TestFormatJobModel:
    def test_creates_with_defaults(self):
        job_id = uuid4()
        fmt = FormatJob(
            id=job_id,
            source_job_id=uuid4(),
            title="Test Format",
            platform="instagram",
            format_type="carousel",
            status=FormatJobStatusEnum.PENDING.value,
        )
        assert fmt.id == job_id
        assert fmt.title == "Test Format"
        assert fmt.platform == "instagram"
        assert fmt.format_type == "carousel"
        assert fmt.status == FormatJobStatusEnum.PENDING.value
        assert fmt.script_content is None
        assert fmt.claims is None
        assert fmt.refined_context is None
        assert fmt.story_directives is None
        assert fmt.hedge_index is None
        assert fmt.epistemic_ledger is None
        assert fmt.format_payload is None
        assert fmt.final_video_url is None
        assert fmt.error_log is None
        assert fmt.locked_at is None
        assert fmt.locked_by is None

    def test_status_enum_values(self):
        assert FormatJobStatusEnum.PENDING == "PENDING"
        assert FormatJobStatusEnum.FORMATTING == "FORMATTING"
        assert FormatJobStatusEnum.ASSET_GENERATION == "ASSET_GENERATION"
        assert FormatJobStatusEnum.COMPLETED == "COMPLETED"
        assert FormatJobStatusEnum.FAILED == "FAILED"
        assert FormatJobStatusEnum.HUMAN_REVIEW_NEEDED == "HUMAN_REVIEW_NEEDED"

    def test_can_set_all_snapshot_fields(self):
        job = FormatJob(
            source_job_id=uuid4(),
            title="Test",
            platform="twitter",
            format_type="video",
            script_content="script content",
            claims=[{"claim_text": "test", "verdict": "SUPPORTED"}],
            refined_context="refined context",
            story_directives={"tone": "analytical"},
            hedge_index=[{"claim_text": "test", "verdict": "UNCERTAIN"}],
            epistemic_ledger={"key": "value"},
        )
        assert job.script_content == "script content"
        assert len(job.claims) == 1
        assert job.refined_context == "refined context"
        assert job.story_directives["tone"] == "analytical"
        assert len(job.hedge_index) == 1
        assert job.epistemic_ledger["key"] == "value"

    def test_can_store_format_payload(self):
        job = FormatJob(
            source_job_id=uuid4(),
            title="Test",
            platform="linkedin",
            format_type="blog",
            format_payload={"BLOG": {"status": "SUCCESS", "content": "# Title"}},
        )
        assert job.format_payload["BLOG"]["status"] == "SUCCESS"

    def test_can_store_video_url(self):
        job = FormatJob(
            source_job_id=uuid4(),
            title="Test",
            platform="youtube",
            format_type="video",
            final_video_url="https://storage.example.com/video.mp4",
        )
        assert job.final_video_url == "https://storage.example.com/video.mp4"
