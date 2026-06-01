import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.schemas.shorts import FormatJobStatusEnum


@pytest.fixture
def mock_format_job():
    job = MagicMock()
    job.id = uuid4()
    job.title = "Test"
    job.platform = "instagram"
    job.format_type = "carousel"
    job.script_content = "Test script content"
    job.claims = [{"claim_text": "test", "verdict": "SUPPORTED"}]
    job.refined_context = "Refined context"
    job.story_directives = {"tone": "analytical"}
    job.hedge_index = []
    job.epistemic_ledger = {}
    job.format_payload = None
    job.final_video_url = None
    job.error_log = None
    job.status = FormatJobStatusEnum.PENDING
    return job


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.mark.unit
class TestExecuteFormatStateTransition:
    async def test_pending_transitions_to_formatting(self, mock_db, mock_format_job):
        mock_format_job.status = FormatJobStatusEnum.PENDING

        with (
            patch(
                "app.workers.format_orchestrator._transition_formatting",
                new=AsyncMock(),
            ) as mock_transition,
            patch(
                "app.workers.format_orchestrator._transition_asset_generation",
                new=AsyncMock(),
            ),
        ):
            from app.workers.format_orchestrator import execute_format_state_transition

            await execute_format_state_transition(mock_db, mock_format_job)

            mock_transition.assert_awaited_once_with(mock_db, mock_format_job)

    async def test_asset_generation_transitions_correctly(
        self, mock_db, mock_format_job
    ):
        mock_format_job.status = FormatJobStatusEnum.ASSET_GENERATION
        mock_format_job.format_payload = {
            "CAROUSEL": {"status": "SUCCESS", "payload": {"slides": []}}
        }

        with (
            patch(
                "app.workers.format_orchestrator._transition_formatting",
                new=AsyncMock(),
            ),
            patch(
                "app.workers.format_orchestrator._transition_asset_generation",
                new=AsyncMock(),
            ) as mock_asset,
        ):
            from app.workers.format_orchestrator import execute_format_state_transition

            await execute_format_state_transition(mock_db, mock_format_job)

            mock_asset.assert_awaited_once_with(mock_db, mock_format_job)

    async def test_terminal_state_does_nothing(self, mock_db, mock_format_job):
        mock_format_job.status = FormatJobStatusEnum.COMPLETED

        with (
            patch(
                "app.workers.format_orchestrator._transition_formatting",
                new=AsyncMock(),
            ),
            patch(
                "app.workers.format_orchestrator._transition_asset_generation",
                new=AsyncMock(),
            ),
            patch(
                "app.workers.format_orchestrator.logger",
            ) as mock_logger,
        ):
            from app.workers.format_orchestrator import execute_format_state_transition

            await execute_format_state_transition(mock_db, mock_format_job)

            mock_logger.warning.assert_called_once()

    async def test_exception_sets_failed(self, mock_db, mock_format_job):
        mock_format_job.status = FormatJobStatusEnum.PENDING

        with (
            patch(
                "app.workers.format_orchestrator._transition_formatting",
                new=AsyncMock(side_effect=Exception("boom")),
            ),
            patch(
                "app.workers.format_orchestrator.log_format_job_error",
                new=AsyncMock(),
            ) as mock_log_error,
            patch(
                "app.workers.format_orchestrator.update_format_job_status",
                new=AsyncMock(),
            ) as mock_update,
        ):
            from app.workers.format_orchestrator import execute_format_state_transition

            await execute_format_state_transition(mock_db, mock_format_job)

            mock_log_error.assert_awaited_once()
            mock_update.assert_awaited_with(
                mock_db, mock_format_job.id, FormatJobStatusEnum.FAILED
            )


@pytest.mark.unit
class TestBuildFormatContent:
    def test_builds_blog_format(self):
        from app.workers.format_orchestrator import _build_format_content

        payload = {
            "title": "Test Blog",
            "sections": [
                {"heading": "Intro", "body": "This is the intro."},
                {"heading": "Body", "body": "This is the body."},
            ],
        }
        result = _build_format_content("BLOG", payload)
        assert "# Test Blog" in result
        assert "## Intro" in result
        assert "This is the body." in result

    def test_builds_carousel_format(self):
        from app.workers.format_orchestrator import _build_format_content

        payload = {
            "thread_title": "Test Thread",
            "slides": [
                {"slide_number": 1, "text": "First slide"},
                {"slide_number": 2, "text": "Second slide"},
            ],
        }
        result = _build_format_content("CAROUSEL", payload)
        assert "# Test Thread" in result
        assert "**Slide 1**" in result
        assert "Second slide" in result

    def test_builds_video_format(self):
        from app.workers.format_orchestrator import _build_format_content

        payload = {
            "title": "Test Video",
            "scenes": [
                {
                    "scene_number": 1,
                    "narration_text": "Intro narration",
                    "visual_prompt": "World map",
                    "audio_cue": "Tension build",
                },
            ],
        }
        result = _build_format_content("VIDEO", payload)
        assert "# Test Video" in result
        assert "### Scene 1" in result
        assert "Intro narration" in result

    def test_returns_title_for_unknown_format(self):
        from app.workers.format_orchestrator import _build_format_content

        result = _build_format_content("UNKNOWN", {"title": "Fallback Title"})
        assert result == "Fallback Title"


@pytest.mark.unit
class TestNextStatusAfterFormatting:
    def test_video_triggers_asset_generation(self):
        from app.workers.format_orchestrator import _next_status_after_formatting
        from app.schemas.shorts import FormatTypeEnum

        result = _next_status_after_formatting([FormatTypeEnum.VIDEO])
        assert result == FormatJobStatusEnum.ASSET_GENERATION

    def test_carousel_triggers_asset_generation(self):
        from app.workers.format_orchestrator import _next_status_after_formatting
        from app.schemas.shorts import FormatTypeEnum

        result = _next_status_after_formatting([FormatTypeEnum.CAROUSEL])
        assert result == FormatJobStatusEnum.ASSET_GENERATION

    def test_blog_only_completes(self):
        from app.workers.format_orchestrator import _next_status_after_formatting
        from app.schemas.shorts import FormatTypeEnum

        result = _next_status_after_formatting([FormatTypeEnum.BLOG])
        assert result == FormatJobStatusEnum.COMPLETED
