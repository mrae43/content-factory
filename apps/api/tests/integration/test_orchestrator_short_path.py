import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.workers.harness import HarnessResult
from app.schemas.shorts import JobStatusEnum, FormatTypeEnum
from app.workers.orchestrator import execute_state_transition
from tests.integration.conftest import _mock_agent_class


def _make_harness_result(success: bool, payload: dict):
    return HarnessResult(
        success=success,
        format_type="short",
        payload=payload,
        attempts=1,
    )


@pytest.fixture
def short_format_script():
    script = MagicMock()
    script.id = uuid4()
    script.job_id = uuid4()
    script.format_payload = {
        "format": "short",
        "version": 1,
        "scenes": [
            {
                "scene_number": 1,
                "narration_text": "Welcome to the show.",
                "visual_prompt": "Bright studio background",
                "asset_type": "video_clip",
                "target_duration_seconds": 5.0,
            },
            {
                "scene_number": 2,
                "narration_text": "Here is the main point.",
                "visual_prompt": "Graph trending up",
                "asset_type": "ken_burns",
                "kb_motion": "zoom_in",
                "target_duration_seconds": 5.0,
            },
        ],
        "target_total_duration": 30.0,
        "visual_style": "Cinematic",
        "audio_direction": "Upbeat",
        "music_mood": "synthwave_hype",
        "voice_id": "test-voice",
        "subtitle_preset": "CENTER_POP_YELLOW",
    }
    script.content = "Short script content"
    return script


@pytest.mark.integration
class TestShortPathStateTransitions:
    async def test_formatting_to_asset_generation_for_short(
        self,
        mock_db_session,
        mock_job,
        mock_vector_store,
        mock_script,
        agent_result_success,
    ):
        mock_job.status = JobStatusEnum.FORMATTING
        mock_job.platform = "tiktok"
        mock_job.format_type = "short"
        mock_job.story_directives = {"voice_id": "test-voice", "loopable": True}

        short_payload = {
            "format": "short",
            "version": 1,
            "scenes": [
                {
                    "scene_number": 1,
                    "narration_text": "Welcome to the show.",
                    "visual_prompt": "Bright studio background",
                    "asset_type": "video_clip",
                    "target_duration_seconds": 5.0,
                }
            ],
            "target_total_duration": 30.0,
            "visual_style": "Cinematic",
            "audio_direction": "Upbeat",
            "music_mood": "synthwave_hype",
            "voice_id": "test-voice",
            "subtitle_preset": "CENTER_POP_YELLOW",
        }

        result = agent_result_success(payload=short_payload)

        with (
            patch(
                "app.workers.orchestrator.ShortFormatterAgent",
                return_value=_mock_agent_class(result).return_value,
            ),
            patch(
                "app.workers.orchestrator.ShortValidator",
                return_value=MagicMock(
                    validate=MagicMock(
                        return_value=MagicMock(
                            valid=True, validated_payload=short_payload
                        )
                    )
                ),
            ),
            patch(
                "app.workers.orchestrator.AgentHarness",
                return_value=MagicMock(
                    run_with_harness=AsyncMock(
                        return_value=_make_harness_result(True, short_payload)
                    )
                ),
            ),
            patch(
                "app.workers.orchestrator.get_latest_script", new_callable=AsyncMock
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.save_format_script",
                new_callable=AsyncMock,
            ) as mock_save_format,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            mock_get_script.return_value = mock_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_save_format.assert_awaited_once()
            args = mock_save_format.await_args
            assert args.kwargs["format_type"] == "SHORT"
            assert args.kwargs["format_payload"] == short_payload
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.ASSET_GENERATION
            )

    async def test_asset_generation_to_composition_for_short(
        self,
        mock_db_session,
        mock_job,
        short_format_script,
    ):
        mock_job.status = JobStatusEnum.ASSET_GENERATION
        mock_job.platform = "tiktok"
        mock_job.format_type = "short"

        visual_result = _make_harness_result(
            True,
            {
                "scene_urls": [
                    {
                        "scene_number": 1,
                        "url": "s3://factory/shorts/scene_1.mp4",
                        "asset_type": "video_clip",
                    },
                    {
                        "scene_number": 2,
                        "url": "s3://factory/shorts/scene_2.png",
                        "asset_type": "ken_burns",
                    },
                ],
                "updated_format_payload": {
                    **short_format_script.format_payload,
                    "scenes": [
                        {
                            **short_format_script.format_payload["scenes"][0],
                            "video_url": "s3://factory/shorts/scene_1.mp4",
                        },
                        {
                            **short_format_script.format_payload["scenes"][1],
                            "image_url": "s3://factory/shorts/scene_2.png",
                        },
                    ],
                },
            },
        )
        voiceover_result = _make_harness_result(
            True,
            {
                "voiceover_url": "s3://factory/shorts/voiceover.mp3",
                "vocal_alignment_url": "s3://factory/shorts/alignment.json",
                "duration_seconds": 12.5,
            },
        )

        with (
            patch(
                "app.workers.orchestrator.get_latest_format_script",
                new_callable=AsyncMock,
            ) as mock_get_format_script,
            patch(
                "app.workers.orchestrator.AgentHarness",
                return_value=MagicMock(
                    run_with_harness=AsyncMock(
                        side_effect=[visual_result, voiceover_result]
                    )
                ),
            ),
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            mock_get_format_script.side_effect = [
                None,  # video
                None,  # carousel
                short_format_script,  # short
            ]

            await execute_state_transition(mock_db_session, mock_job)

            mock_db_session.commit.assert_awaited()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.COMPOSITION
            )

    async def test_composition_to_completed_for_short(
        self,
        mock_db_session,
        mock_job,
        short_format_script,
    ):
        mock_job.status = JobStatusEnum.COMPOSITION
        mock_job.platform = "tiktok"
        mock_job.format_type = "short"

        short_format_script.format_payload["voiceover_url"] = (
            "s3://factory/shorts/voiceover.mp3"
        )
        short_format_script.format_payload["vocal_alignment_url"] = (
            "s3://factory/shorts/alignment.json"
        )

        composer_result = _make_harness_result(
            True,
            {
                "final_video_url": "s3://factory/shorts/final_output.mp4",
            },
        )

        with (
            patch(
                "app.workers.orchestrator.get_latest_format_script",
                new_callable=AsyncMock,
            ) as mock_get_format_script,
            patch(
                "app.workers.orchestrator.AgentHarness",
                return_value=MagicMock(
                    run_with_harness=AsyncMock(return_value=composer_result)
                ),
            ),
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            mock_get_format_script.return_value = short_format_script

            await execute_state_transition(mock_db_session, mock_job)

            assert mock_job.final_video_url == "s3://factory/shorts/final_output.mp4"
            mock_db_session.commit.assert_awaited()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.COMPLETED
            )

    async def test_short_on_tiktok_does_not_enter_video_path(
        self,
        mock_db_session,
        mock_job,
        short_format_script,
    ):
        mock_job.status = JobStatusEnum.ASSET_GENERATION
        mock_job.platform = "tiktok"
        mock_job.format_type = "short"

        visual_result = _make_harness_result(
            True,
            {
                "scene_urls": [
                    {
                        "scene_number": 1,
                        "url": "s3://factory/shorts/scene_1.mp4",
                        "asset_type": "video_clip",
                    }
                ],
                "updated_format_payload": short_format_script.format_payload,
            },
        )
        voiceover_result = _make_harness_result(
            True,
            {
                "voiceover_url": "s3://factory/shorts/voiceover.mp3",
                "vocal_alignment_url": "s3://factory/shorts/alignment.json",
                "duration_seconds": 10.0,
            },
        )

        with (
            patch(
                "app.workers.orchestrator.get_latest_format_script",
                new_callable=AsyncMock,
            ) as mock_get_format_script,
            patch(
                "app.workers.orchestrator.AgentHarness",
                return_value=MagicMock(
                    run_with_harness=AsyncMock(
                        side_effect=[visual_result, voiceover_result]
                    )
                ),
            ),
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            mock_get_format_script.side_effect = [
                None,  # video
                None,  # carousel
                short_format_script,  # short
            ]

            await execute_state_transition(mock_db_session, mock_job)

            args = mock_update.await_args
            assert args.args[2] == JobStatusEnum.COMPOSITION

    async def test_short_on_instagram_does_not_enter_video_path(
        self,
        mock_db_session,
        mock_job,
        short_format_script,
    ):
        mock_job.status = JobStatusEnum.ASSET_GENERATION
        mock_job.platform = "instagram"
        mock_job.format_type = "short"

        visual_result = _make_harness_result(
            True,
            {
                "scene_urls": [
                    {
                        "scene_number": 1,
                        "url": "s3://factory/shorts/scene_1.mp4",
                        "asset_type": "video_clip",
                    }
                ],
                "updated_format_payload": short_format_script.format_payload,
            },
        )
        voiceover_result = _make_harness_result(
            True,
            {
                "voiceover_url": "s3://factory/shorts/voiceover.mp3",
                "vocal_alignment_url": "s3://factory/shorts/alignment.json",
                "duration_seconds": 10.0,
            },
        )

        with (
            patch(
                "app.workers.orchestrator.get_latest_format_script",
                new_callable=AsyncMock,
            ) as mock_get_format_script,
            patch(
                "app.workers.orchestrator.AgentHarness",
                return_value=MagicMock(
                    run_with_harness=AsyncMock(
                        side_effect=[visual_result, voiceover_result]
                    )
                ),
            ),
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            mock_get_format_script.side_effect = [
                None,  # video
                None,  # carousel
                short_format_script,  # short
            ]

            await execute_state_transition(mock_db_session, mock_job)

            args = mock_update.await_args
            assert args.args[2] == JobStatusEnum.COMPOSITION

    async def test_composition_fails_when_missing_voiceover(
        self,
        mock_db_session,
        mock_job,
        short_format_script,
    ):
        mock_job.status = JobStatusEnum.COMPOSITION
        mock_job.platform = "tiktok"
        mock_job.format_type = "short"

        # Missing voiceover_url in payload
        short_format_script.format_payload.pop("voiceover_url", None)
        short_format_script.format_payload["vocal_alignment_url"] = (
            "s3://factory/shorts/alignment.json"
        )

        with (
            patch(
                "app.workers.orchestrator.get_latest_format_script",
                new_callable=AsyncMock,
            ) as mock_get_format_script,
            patch(
                "app.workers.orchestrator.log_error",
                new_callable=AsyncMock,
            ) as mock_log,
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            mock_get_format_script.return_value = short_format_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )

    async def test_composition_fails_on_composer_error(
        self,
        mock_db_session,
        mock_job,
        short_format_script,
    ):
        mock_job.status = JobStatusEnum.COMPOSITION
        mock_job.platform = "tiktok"
        mock_job.format_type = "short"

        short_format_script.format_payload["voiceover_url"] = (
            "s3://factory/shorts/voiceover.mp3"
        )
        short_format_script.format_payload["vocal_alignment_url"] = (
            "s3://factory/shorts/alignment.json"
        )

        composer_result = _make_harness_result(
            False,
            {},
        )
        composer_result.error_log = ["FFmpeg failed"]

        with (
            patch(
                "app.workers.orchestrator.get_latest_format_script",
                new_callable=AsyncMock,
            ) as mock_get_format_script,
            patch(
                "app.workers.orchestrator.AgentHarness",
                return_value=MagicMock(
                    run_with_harness=AsyncMock(return_value=composer_result)
                ),
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
            mock_get_format_script.return_value = short_format_script

            await execute_state_transition(mock_db_session, mock_job)

            mock_log.assert_awaited_once()
            mock_update.assert_awaited_once_with(
                mock_db_session, mock_job.id, JobStatusEnum.FAILED
            )

    async def test_full_short_happy_path(
        self,
        mock_db_session,
        mock_job,
        short_format_script,
    ):
        mock_job.platform = "tiktok"
        mock_job.format_type = "short"
        mock_job.story_directives = {"voice_id": "test-voice", "loopable": True}

        short_payload = short_format_script.format_payload

        formatter_harness = _make_harness_result(True, short_payload)
        visual_harness = _make_harness_result(
            True,
            {
                "scene_urls": [
                    {
                        "scene_number": 1,
                        "url": "s3://factory/shorts/scene_1.mp4",
                        "asset_type": "video_clip",
                    }
                ],
                "updated_format_payload": short_payload,
            },
        )
        voiceover_harness = _make_harness_result(
            True,
            {
                "voiceover_url": "s3://factory/shorts/voiceover.mp3",
                "vocal_alignment_url": "s3://factory/shorts/alignment.json",
                "duration_seconds": 10.0,
            },
        )
        composer_harness = _make_harness_result(
            True,
            {"final_video_url": "s3://factory/shorts/final.mp4"},
        )

        with (
            patch(
                "app.workers.orchestrator.AgentHarness",
                return_value=MagicMock(
                    run_with_harness=AsyncMock(
                        side_effect=[
                            formatter_harness,
                            visual_harness,
                            voiceover_harness,
                            composer_harness,
                        ]
                    )
                ),
            ),
            patch(
                "app.workers.orchestrator.ShortFormatterAgent",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.ShortValidator",
                return_value=MagicMock(),
            ),
            patch(
                "app.workers.orchestrator.get_latest_script",
                new_callable=AsyncMock,
            ) as mock_get_script,
            patch(
                "app.workers.orchestrator.get_latest_format_script",
                new_callable=AsyncMock,
            ) as mock_get_format_script,
            patch(
                "app.workers.orchestrator.save_format_script",
                new_callable=AsyncMock,
            ),
            patch(
                "app.workers.orchestrator.update_job_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            mock_get_script.return_value = MagicMock(
                id=uuid4(),
                version=1,
                content="Test script",
                is_approved=True,
                feedback_history=[],
            )
            mock_get_format_script.side_effect = [
                None,  # video
                None,  # carousel
                short_format_script,  # short (asset gen)
                short_format_script,  # short (composition)
            ]

            # FORMATTING -> ASSET_GENERATION
            mock_job.status = JobStatusEnum.FORMATTING
            await execute_state_transition(mock_db_session, mock_job)
            assert mock_update.call_args.args[2] == JobStatusEnum.ASSET_GENERATION

            # ASSET_GENERATION -> COMPOSITION
            mock_job.status = JobStatusEnum.ASSET_GENERATION
            await execute_state_transition(mock_db_session, mock_job)
            assert mock_update.call_args.args[2] == JobStatusEnum.COMPOSITION

            # COMPOSITION -> COMPLETED
            short_format_script.format_payload["voiceover_url"] = (
                "s3://factory/shorts/voiceover.mp3"
            )
            short_format_script.format_payload["vocal_alignment_url"] = (
                "s3://factory/shorts/alignment.json"
            )
            mock_job.status = JobStatusEnum.COMPOSITION
            await execute_state_transition(mock_db_session, mock_job)
            assert mock_update.call_args.args[2] == JobStatusEnum.COMPLETED
            assert mock_job.final_video_url == "s3://factory/shorts/final.mp4"


@pytest.mark.integration
class TestNextStatusAfterFormatting:
    def test_short_returns_asset_generation(self):
        from app.workers.orchestrator import _next_status_after_formatting

        result = _next_status_after_formatting([FormatTypeEnum.SHORT])
        assert result == JobStatusEnum.ASSET_GENERATION

    def test_short_with_blog_returns_asset_generation(self):
        from app.workers.orchestrator import _next_status_after_formatting

        result = _next_status_after_formatting(
            [FormatTypeEnum.SHORT, FormatTypeEnum.BLOG]
        )
        assert result == JobStatusEnum.ASSET_GENERATION

    def test_blog_only_completes(self):
        from app.workers.orchestrator import _next_status_after_formatting

        result = _next_status_after_formatting([FormatTypeEnum.BLOG])
        assert result == JobStatusEnum.COMPLETED


@pytest.mark.integration
class TestBuildFormatContentShort:
    def test_builds_short_format(self):
        from app.workers.orchestrator import _build_format_content

        payload = {
            "scenes": [
                {
                    "scene_number": 1,
                    "narration_text": "Intro narration",
                    "visual_prompt": "World map",
                    "asset_type": "video_clip",
                },
                {
                    "scene_number": 2,
                    "narration_text": "Main point",
                    "visual_prompt": "Graph",
                    "asset_type": "ken_burns",
                },
            ]
        }
        result = _build_format_content("SHORT", payload)
        assert "### Scene 1" in result
        assert "Intro narration" in result
        assert "video_clip" in result
        assert "### Scene 2" in result
        assert "ken_burns" in result
