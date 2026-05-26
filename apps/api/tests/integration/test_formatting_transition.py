import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.workers.agents import AgentActionStatus, AgentResult
from app.workers.harness import HarnessResult
from app.schemas.shorts import JobStatusEnum, FormatTypeEnum
from app.workers.orchestrator import execute_state_transition
from tests.integration.conftest import _mock_agent_class


def _harness_success(payload: dict, fmt_type: str = "blog") -> HarnessResult:
    return HarnessResult(
        success=True,
        format_type=fmt_type,
        payload=payload,
        attempts=1,
    )


def _harness_failure(fmt_type: str = "blog") -> HarnessResult:
    return HarnessResult(
        success=False,
        format_type=fmt_type,
        error_log=["Validation failed"],
        attempts=3,
    )


def _mock_harness(result: HarnessResult):
    harness = AsyncMock()
    harness.run_with_harness = AsyncMock(return_value=result)
    return harness


@pytest.mark.integration
class TestTransitionFormattingVideo:
    async def test_should_route_to_formatting_from_fact_check(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.FACT_CHECKING_SCRIPT
        mock_job.format_type = "video"
        mock_job.story_directives = {"guardrail_strictness": "Low"}
        mock_script = MagicMock()
        mock_script.id = uuid4()
        mock_script.version = 1
        mock_script.is_approved = False
        mock_script.feedback_history = []

        red_team_result = AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"claims": [], "verdict": "SUPPORTED"},
            reasoning="All good",
            confidence_score=0.95,
        )

        with (
            patch(
                "app.workers.orchestrator.RedTeamAgent",
                return_value=_mock_agent_class(red_team_result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ContentFactoryVectorStore",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.get_latest_script",
                new_callable=AsyncMock,
                return_value=mock_script,
            ),
            patch(
                "app.workers.orchestrator.save_fact_check_claims",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FORMATTING
            )


@pytest.mark.integration
class TestTransitionFormattingBlog:
    async def test_should_run_blog_formatter_and_complete(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.FORMATTING
        mock_job.format_type = "blog"
        mock_job.refined_context = "BRICS GDP grew 3.2%"
        mock_job.platform = ""

        mock_script = MagicMock()
        mock_script.id = uuid4()
        mock_script.version = 1
        mock_script.content = "Script content"

        harness_result = _harness_success(
            {"title": "Blog Title", "_format": "blog"}, "blog"
        )

        with (
            patch(
                "app.workers.orchestrator.get_latest_script",
                new_callable=AsyncMock,
                return_value=mock_script,
            ),
            patch(
                "app.workers.orchestrator.get_script_claims",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.workers.orchestrator.FormatterHarness",
                return_value=_mock_harness(harness_result),
            ),
            patch(
                "app.workers.orchestrator.BlogFormatterAgent",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.BlogValidator",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.save_format_script",
                new_callable=AsyncMock,
            ) as mock_save,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_save.assert_awaited_once()
            save_call = mock_save.call_args
            assert save_call.kwargs["format_type"] == "BLOG"
            assert save_call.kwargs["is_approved"] is True
            assert save_call.kwargs["format_payload"] == harness_result.payload
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.COMPLETED
            )


@pytest.mark.integration
class TestTransitionFormattingCarousel:
    async def test_should_run_carousel_formatter_and_complete(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.FORMATTING
        mock_job.format_type = "carousel"
        mock_job.refined_context = "BRICS GDP grew 3.2%"
        mock_job.platform = "twitter"

        mock_script = MagicMock()
        mock_script.id = uuid4()
        mock_script.version = 1
        mock_script.content = "Script content"

        harness_result = _harness_success(
            {"thread_title": "Thread Title", "_format": "carousel"}, "carousel"
        )

        with (
            patch(
                "app.workers.orchestrator.get_latest_script",
                new_callable=AsyncMock,
                return_value=mock_script,
            ),
            patch(
                "app.workers.orchestrator.get_script_claims",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.workers.orchestrator.FormatterHarness",
                return_value=_mock_harness(harness_result),
            ),
            patch(
                "app.workers.orchestrator.CarouselFormatterAgent",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.CarouselValidator",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.save_format_script",
                new_callable=AsyncMock,
            ) as mock_save,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_save.assert_awaited_once()
            save_call = mock_save.call_args
            assert save_call.kwargs["format_type"] == "CAROUSEL"
            assert save_call.kwargs["is_approved"] is True
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.ASSET_GENERATION
            )


@pytest.mark.integration
class TestTransitionFormattingAll:
    async def test_should_run_all_formatters_in_parallel_then_asset_gen(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.FORMATTING
        mock_job.format_type = "all"
        mock_job.refined_context = "BRICS GDP grew 3.2%"
        mock_job.platform = ""

        mock_script = MagicMock()
        mock_script.id = uuid4()
        mock_script.version = 1
        mock_script.content = "Script content"

        blog_result = _harness_success({"title": "Blog", "_format": "blog"}, "blog")
        carousel_result = _harness_success(
            {"thread_title": "Thread", "_format": "carousel"}, "carousel"
        )
        video_result = _harness_success(
            {"visual_style": "Cinematic", "_format": "video"}, "video"
        )

        call_count = 0

        def mock_harness_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_harness(blog_result)
            elif call_count == 2:
                return _mock_harness(carousel_result)
            return _mock_harness(video_result)

        with (
            patch(
                "app.workers.orchestrator.get_latest_script",
                new_callable=AsyncMock,
                return_value=mock_script,
            ),
            patch(
                "app.workers.orchestrator.get_script_claims",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.workers.orchestrator.FormatterHarness",
                side_effect=mock_harness_factory,
            ),
            patch(
                "app.workers.orchestrator.BlogFormatterAgent",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.BlogValidator",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.CarouselFormatterAgent",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.CarouselValidator",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.VideoFormatterAgent",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.VideoValidator",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.save_format_script",
                new_callable=AsyncMock,
            ) as mock_save,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            assert mock_save.call_count == 3
            fmt_types = [c.kwargs["format_type"] for c in mock_save.call_args_list]
            assert "BLOG" in fmt_types
            assert "CAROUSEL" in fmt_types
            assert "VIDEO" in fmt_types
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.ASSET_GENERATION
            )


@pytest.mark.integration
class TestTransitionFormattingFailure:
    async def test_should_raise_when_all_formatters_fail(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.FORMATTING
        mock_job.format_type = "blog"
        mock_job.refined_context = "Context"
        mock_job.platform = ""

        mock_script = MagicMock()
        mock_script.id = uuid4()
        mock_script.version = 1
        mock_script.content = "Script"

        harness_result = _harness_failure("blog")

        with (
            patch(
                "app.workers.orchestrator.get_latest_script",
                new_callable=AsyncMock,
                return_value=mock_script,
            ),
            patch(
                "app.workers.orchestrator.get_script_claims",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.workers.orchestrator.FormatterHarness",
                return_value=_mock_harness(harness_result),
            ),
            patch(
                "app.workers.orchestrator.BlogFormatterAgent",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.BlogValidator",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.save_format_script",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.log_error",
                new_callable=AsyncMock,
            ) as mock_log,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )

    async def test_should_raise_when_no_script_found(self, mock_db_session, mock_job):
        mock_job.status = JobStatusEnum.FORMATTING
        mock_job.format_type = "blog"

        with (
            patch(
                "app.workers.orchestrator.get_latest_script",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.workers.orchestrator.log_error",
                new_callable=AsyncMock,
            ) as mock_log,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )


@pytest.mark.integration
class TestTransitionFormattingVideoOnly:
    async def test_should_run_video_formatter_and_go_to_asset_generation(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.FORMATTING
        mock_job.format_type = "video"
        mock_job.refined_context = "BRICS GDP grew 3.2%"
        mock_job.platform = ""

        mock_script = MagicMock()
        mock_script.id = uuid4()
        mock_script.version = 1
        mock_script.content = "Script content"

        harness_result = _harness_success(
            {"visual_style": "Cinematic", "_format": "video"}, "video"
        )

        with (
            patch(
                "app.workers.orchestrator.get_latest_script",
                new_callable=AsyncMock,
                return_value=mock_script,
            ),
            patch(
                "app.workers.orchestrator.get_script_claims",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.workers.orchestrator.FormatterHarness",
                return_value=_mock_harness(harness_result),
            ),
            patch(
                "app.workers.orchestrator.VideoFormatterAgent",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.VideoValidator",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.save_format_script",
                new_callable=AsyncMock,
            ) as mock_save,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            mock_save.assert_awaited_once()
            save_call = mock_save.call_args
            assert save_call.kwargs["format_type"] == "VIDEO"
            assert save_call.kwargs["is_approved"] is True
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.ASSET_GENERATION
            )


@pytest.mark.integration
class TestTransitionAssetGenerationMissingVideoScript:
    async def test_should_fail_when_no_video_script_with_payload(
        self, mock_db_session, mock_job
    ):
        mock_job.status = JobStatusEnum.ASSET_GENERATION
        mock_job.format_type = "video"

        with (
            patch(
                "app.workers.orchestrator.get_latest_format_script",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
            patch(
                "app.workers.orchestrator.log_error",
                new_callable=AsyncMock,
            ) as mock_log,
        ):
            await execute_state_transition(mock_db_session, mock_job)

            assert mock_get_script.await_count == 2
            mock_get_script.assert_any_await(mock_db_session, mock_job.id, "VIDEO")
            mock_get_script.assert_any_await(mock_db_session, mock_job.id, "CAROUSEL")
            mock_log.assert_awaited_once()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )


@pytest.mark.integration
class TestNextStatusAfterFactCheck:
    @pytest.mark.parametrize(
        "format_type,expected",
        [
            ("video", JobStatusEnum.FORMATTING),
            ("blog", JobStatusEnum.FORMATTING),
            ("carousel", JobStatusEnum.FORMATTING),
            ("all", JobStatusEnum.FORMATTING),
        ],
    )
    def test_should_route_correctly(self, format_type, expected):
        from app.schemas.shorts import next_status_after_fact_check

        result = next_status_after_fact_check(format_type)
        assert result == expected


@pytest.mark.integration
class TestNextStatusAfterFormatting:
    @pytest.mark.parametrize(
        "resolved_formats,expected",
        [
            ([FormatTypeEnum.BLOG], JobStatusEnum.COMPLETED),
            ([FormatTypeEnum.CAROUSEL], JobStatusEnum.ASSET_GENERATION),
            ([FormatTypeEnum.VIDEO], JobStatusEnum.ASSET_GENERATION),
            (
                [FormatTypeEnum.VIDEO, FormatTypeEnum.BLOG],
                JobStatusEnum.ASSET_GENERATION,
            ),
        ],
    )
    def test_should_route_correctly(self, resolved_formats, expected):
        from app.workers.orchestrator import _next_status_after_formatting

        result = _next_status_after_formatting(resolved_formats)
        assert result == expected
