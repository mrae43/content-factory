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

    def test_short_triggers_asset_generation(self):
        from app.workers.format_orchestrator import _next_status_after_formatting
        from app.schemas.shorts import FormatTypeEnum

        result = _next_status_after_formatting([FormatTypeEnum.SHORT])
        assert result == FormatJobStatusEnum.ASSET_GENERATION

    def test_short_and_video_triggers_asset_generation(self):
        from app.workers.format_orchestrator import _next_status_after_formatting
        from app.schemas.shorts import FormatTypeEnum

        result = _next_status_after_formatting(
            [FormatTypeEnum.SHORT, FormatTypeEnum.VIDEO]
        )
        assert result == FormatJobStatusEnum.ASSET_GENERATION


@pytest.mark.unit
class TestTransitionFormattingShort:
    async def test_short_formatter_harness_created(self, mock_db, mock_format_job):
        mock_format_job.format_type = "short"
        mock_format_job.platform = "tiktok"
        mock_format_job.story_directives = {"tone": "analytical"}

        with (
            patch("app.workers.format_orchestrator.AgentHarness") as mock_harness_cls,
            patch(
                "app.workers.format_orchestrator.update_format_job_status",
                new=AsyncMock(),
            ),
            patch(
                "app.workers.format_orchestrator.update_format_job_format_payload",
                new=AsyncMock(),
            ),
            patch("app.workers.format_orchestrator.ShortFormatterAgent"),
        ):
            mock_harness = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.payload = {"scenes": []}
            mock_result.attempts = 1
            mock_result.error_log = None
            mock_result.escalated = False
            mock_harness.run_with_harness = AsyncMock(return_value=mock_result)
            mock_harness_cls.return_value = mock_harness

            from app.workers.format_orchestrator import _transition_formatting

            await _transition_formatting(mock_db, mock_format_job)

            mock_harness_cls.assert_called()


@pytest.mark.unit
class TestTransitionShortAssetGeneration:
    async def test_success_writes_merged_payload_and_composition(
        self, mock_db, mock_format_job
    ):
        mock_format_job.format_type = "short"
        mock_format_job.platform = "tiktok"
        mock_format_job.format_payload = {
            "SHORT": {
                "status": "SUCCESS",
                "payload": {"scenes": [{"scene_number": 1}]},
            }
        }

        visual_result = MagicMock()
        visual_result.success = True
        visual_result.payload = {
            "updated_format_payload": {"scenes": [{"scene_number": 1}], "visual": "ok"}
        }

        voice_result = MagicMock()
        voice_result.success = True
        voice_result.payload = {
            "voiceover_url": "https://s3/voice.mp3",
            "vocal_alignment_url": "https://s3/align.json",
        }

        with (
            patch(
                "app.workers.format_orchestrator._run_short_visual_asset",
                new=AsyncMock(return_value=visual_result),
            ),
            patch(
                "app.workers.format_orchestrator._run_short_voiceover",
                new=AsyncMock(return_value=voice_result),
            ),
            patch(
                "app.workers.format_orchestrator.update_format_job_format_payload",
                new=AsyncMock(),
            ) as mock_update_payload,
            patch(
                "app.workers.format_orchestrator.update_format_job_status",
                new=AsyncMock(),
            ) as mock_update_status,
        ):
            from app.workers.format_orchestrator import (
                _transition_short_asset_generation,
            )

            await _transition_short_asset_generation(mock_db, mock_format_job)

            mock_update_payload.assert_awaited_once()
            mock_update_status.assert_awaited_with(
                mock_db, mock_format_job.id, FormatJobStatusEnum.COMPOSITION
            )

    async def test_taskgroup_failure_raises_and_logs(self, mock_db, mock_format_job):
        mock_format_job.format_type = "short"
        mock_format_job.platform = "tiktok"
        mock_format_job.format_payload = {
            "SHORT": {"status": "SUCCESS", "payload": {"scenes": []}}
        }

        with (
            patch(
                "app.workers.format_orchestrator._run_short_visual_asset",
                new=AsyncMock(side_effect=RuntimeError("visual boom")),
            ),
            patch(
                "app.workers.format_orchestrator._run_short_voiceover",
                new=AsyncMock(side_effect=RuntimeError("voice boom")),
            ),
            patch(
                "app.workers.format_orchestrator.log_format_job_error",
                new=AsyncMock(),
            ) as mock_log_error,
        ):
            from app.workers.format_orchestrator import (
                _transition_short_asset_generation,
            )

            with pytest.raises(Exception):
                await _transition_short_asset_generation(mock_db, mock_format_job)

            mock_log_error.assert_awaited_once()

    async def test_visual_fails_raises_exception(self, mock_db, mock_format_job):
        mock_format_job.format_type = "short"
        mock_format_job.platform = "tiktok"
        mock_format_job.format_payload = {
            "SHORT": {"status": "SUCCESS", "payload": {"scenes": []}}
        }

        visual_result = MagicMock()
        visual_result.success = False
        visual_result.payload = {}

        voice_result = MagicMock()
        voice_result.success = True
        voice_result.payload = {
            "voiceover_url": "https://s3/voice.mp3",
            "vocal_alignment_url": "https://s3/align.json",
        }

        with (
            patch(
                "app.workers.format_orchestrator._run_short_visual_asset",
                new=AsyncMock(return_value=visual_result),
            ),
            patch(
                "app.workers.format_orchestrator._run_short_voiceover",
                new=AsyncMock(return_value=voice_result),
            ),
        ):
            from app.workers.format_orchestrator import (
                _transition_short_asset_generation,
            )

            with pytest.raises(Exception, match="SHORT asset generation failed"):
                await _transition_short_asset_generation(mock_db, mock_format_job)


@pytest.mark.unit
class TestTransitionComposition:
    async def test_success_sets_video_url_and_completed(self, mock_db, mock_format_job):
        mock_format_job.format_type = "short"
        mock_format_job.platform = "tiktok"
        mock_format_job.format_payload = {
            "SHORT": {
                "status": "SUCCESS",
                "payload": {
                    "scenes": [],
                    "voiceover_url": "https://s3/voice.mp3",
                    "vocal_alignment_url": "https://s3/align.json",
                },
            }
        }

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.payload = {"final_video_url": "https://s3/final.mp4"}

        with (
            patch("app.workers.format_orchestrator.AgentHarness") as mock_harness_cls,
            patch(
                "app.workers.format_orchestrator.update_format_job_video_url",
                new=AsyncMock(),
            ) as mock_update_url,
            patch(
                "app.workers.format_orchestrator.update_format_job_status",
                new=AsyncMock(),
            ) as mock_update_status,
        ):
            mock_harness = MagicMock()
            mock_harness.run_with_harness = AsyncMock(return_value=mock_result)
            mock_harness_cls.return_value = mock_harness

            from app.workers.format_orchestrator import _transition_composition

            await _transition_composition(mock_db, mock_format_job)

            mock_update_url.assert_awaited_with(
                mock_db, mock_format_job.id, "https://s3/final.mp4"
            )
            mock_update_status.assert_awaited_with(
                mock_db, mock_format_job.id, FormatJobStatusEnum.COMPLETED
            )

    async def test_failure_logs_error_and_sets_failed(self, mock_db, mock_format_job):
        mock_format_job.format_type = "short"
        mock_format_job.platform = "tiktok"
        mock_format_job.format_payload = {
            "SHORT": {
                "status": "SUCCESS",
                "payload": {
                    "scenes": [],
                    "voiceover_url": "https://s3/voice.mp3",
                    "vocal_alignment_url": "https://s3/align.json",
                },
            }
        }

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_log = ["Composition failed"]

        with (
            patch("app.workers.format_orchestrator.AgentHarness") as mock_harness_cls,
            patch(
                "app.workers.format_orchestrator.log_format_job_error",
                new=AsyncMock(),
            ) as mock_log_error,
            patch(
                "app.workers.format_orchestrator.update_format_job_status",
                new=AsyncMock(),
            ) as mock_update_status,
        ):
            mock_harness = MagicMock()
            mock_harness.run_with_harness = AsyncMock(return_value=mock_result)
            mock_harness_cls.return_value = mock_harness

            from app.workers.format_orchestrator import _transition_composition

            await _transition_composition(mock_db, mock_format_job)

            mock_log_error.assert_awaited_once()
            mock_update_status.assert_awaited_with(
                mock_db, mock_format_job.id, FormatJobStatusEnum.FAILED
            )


@pytest.mark.unit
class TestExecuteFormatStateTransitionComposition:
    async def test_composition_routes_to_transition_composition(
        self, mock_db, mock_format_job
    ):
        mock_format_job.status = FormatJobStatusEnum.COMPOSITION

        with (
            patch(
                "app.workers.format_orchestrator._transition_composition",
                new=AsyncMock(),
            ) as mock_composition,
        ):
            from app.workers.format_orchestrator import execute_format_state_transition

            await execute_format_state_transition(mock_db, mock_format_job)

            mock_composition.assert_awaited_once_with(mock_db, mock_format_job)

    async def test_short_asset_generation_dispatches(self, mock_db, mock_format_job):
        mock_format_job.status = FormatJobStatusEnum.ASSET_GENERATION
        mock_format_job.format_type = "short"

        with (
            patch(
                "app.workers.format_orchestrator._transition_short_asset_generation",
                new=AsyncMock(),
            ) as mock_short_asset,
        ):
            from app.workers.format_orchestrator import execute_format_state_transition

            await execute_format_state_transition(mock_db, mock_format_job)

            mock_short_asset.assert_awaited_once_with(mock_db, mock_format_job)
