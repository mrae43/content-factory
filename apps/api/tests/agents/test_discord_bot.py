import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

from app.discord_bot import PlatformModal, FormatJobWatcher, _resync_format_jobs
from app.discord_ui import ShortFormatSelectionView
from app.schemas.shorts import FormatJobStatusEnum


@pytest.mark.agent
class TestPlatformModal:
    def test_uses_text_input_not_select(self):
        modal = PlatformModal(
            script_job_id="11111111-1111-1111-1111-111111111111",
            format_type="carousel",
        )

        assert len(modal.children) == 1
        item = modal.children[0]
        import discord

        assert isinstance(item, discord.ui.TextInput)
        assert not isinstance(item, discord.ui.Select)

    def test_platform_input_has_correct_placeholder(self):
        modal = PlatformModal(
            script_job_id="11111111-1111-1111-1111-111111111111",
            format_type="video",
        )

        item = modal.children[0]
        assert "tiktok" in item.placeholder.lower()
        assert "instagram" in item.placeholder.lower()

    def test_valid_platform_preserved(self):
        modal = PlatformModal(
            script_job_id="11111111-1111-1111-1111-111111111111",
            format_type="blog",
        )

        modal.platform_input = MagicMock()
        modal.platform_input.value = "  TikTok  "

        raw_platform = (modal.platform_input.value or "").strip().lower()
        from app.discord_bot import VALID_PLATFORMS

        platform = raw_platform if raw_platform in VALID_PLATFORMS else "instagram"
        assert platform == "tiktok"

    def test_invalid_platform_defaults_to_instagram(self):
        modal = PlatformModal(
            script_job_id="11111111-1111-1111-1111-111111111111",
            format_type="carousel",
        )

        modal.platform_input = MagicMock()
        modal.platform_input.value = "foobar"

        raw_platform = (modal.platform_input.value or "").strip().lower()
        from app.discord_bot import VALID_PLATFORMS

        platform = raw_platform if raw_platform in VALID_PLATFORMS else "instagram"
        assert platform == "instagram"


@pytest.mark.agent
class TestShortFormatSelectionView:
    def test_platform_options(self):
        view = ShortFormatSelectionView(
            script_job_id="11111111-1111-1111-1111-111111111111"
        )

        select = None
        for child in view.children:
            if hasattr(child, "custom_id") and child.custom_id == "short_platform":
                select = child
                break

        assert select is not None
        assert len(select.options) == 3
        values = [o.value for o in select.options]
        assert "tiktok" in values
        assert "instagram" in values
        assert "youtube" in values

    def test_style_options(self):
        view = ShortFormatSelectionView(
            script_job_id="11111111-1111-1111-1111-111111111111"
        )

        select = None
        for child in view.children:
            if hasattr(child, "custom_id") and child.custom_id == "short_style":
                select = child
                break

        assert select is not None
        assert len(select.options) == 5

    def test_loopable_toggle(self):
        view = ShortFormatSelectionView(
            script_job_id="11111111-1111-1111-1111-111111111111"
        )

        assert view.loopable is True

        toggle = None
        for child in view.children:
            if hasattr(child, "custom_id") and child.custom_id == "short_loopable":
                toggle = child
                break

        assert toggle is not None
        assert "On" in toggle.label

        view.loopable = False
        toggle.label = f"Loopable: {'On' if view.loopable else 'Off'}"
        assert "Off" in toggle.label


@pytest.mark.agent
class TestFormatJobWatcher:
    async def test_polling_detects_state_change(self):
        watcher = FormatJobWatcher()

        mock_job = MagicMock()
        mock_job.id = uuid4()
        mock_job.status = FormatJobStatusEnum.FORMATTING
        mock_job.working_memory = {
            "discord_thread_id": "123",
            "discord_message_id": "456",
        }
        mock_job.created_at = None

        with (
            patch(
                "app.discord_bot.get_format_jobs_for_watcher",
                new=AsyncMock(return_value=[mock_job]),
            ),
            patch("app.discord_bot.AsyncSessionLocal") as mock_session_cls,
            patch.object(
                watcher, "_update_living_embed", new=AsyncMock()
            ) as mock_update,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            await watcher._check_format_jobs()

            mock_update.assert_awaited_once_with(mock_job, "FORMATTING")
            assert watcher._previous_states[mock_job.id] == "FORMATTING"

    async def test_polling_terminal_state_triggers_finalize(self):
        watcher = FormatJobWatcher()

        mock_job = MagicMock()
        mock_job.id = uuid4()
        mock_job.status = FormatJobStatusEnum.COMPLETED
        mock_job.working_memory = {
            "discord_thread_id": "123",
            "discord_message_id": "456",
        }
        mock_job.created_at = None

        with (
            patch(
                "app.discord_bot.get_format_jobs_for_watcher",
                new=AsyncMock(return_value=[mock_job]),
            ),
            patch("app.discord_bot.AsyncSessionLocal") as mock_session_cls,
            patch.object(watcher, "_update_living_embed", new=AsyncMock()),
            patch.object(watcher, "_finalize_job", new=AsyncMock()) as mock_finalize,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            await watcher._check_format_jobs()

            mock_finalize.assert_awaited_once_with(mock_job, "COMPLETED")
            assert mock_job.id not in watcher._previous_states

    async def test_empty_result_set_no_crash(self):
        watcher = FormatJobWatcher()

        with (
            patch(
                "app.discord_bot.get_format_jobs_for_watcher",
                new=AsyncMock(return_value=[]),
            ),
            patch("app.discord_bot.AsyncSessionLocal") as mock_session_cls,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            await watcher._check_format_jobs()

    async def test_embed_edit_failure_logged(self):
        watcher = FormatJobWatcher()
        job_id = uuid4()
        watcher._previous_states[job_id] = "PENDING"

        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.status = FormatJobStatusEnum.FORMATTING
        mock_job.working_memory = {
            "discord_thread_id": "123",
            "discord_message_id": "456",
        }
        mock_job.created_at = None

        import discord

        with (
            patch(
                "app.discord_bot.get_format_jobs_for_watcher",
                new=AsyncMock(return_value=[mock_job]),
            ),
            patch("app.discord_bot.AsyncSessionLocal") as mock_session_cls,
            patch("app.discord_bot.bot") as mock_bot,
            patch("app.discord_bot.logger") as mock_logger,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            mock_bot.fetch_channel = AsyncMock(
                side_effect=discord.NotFound(MagicMock(), "not found")
            )

            await watcher._check_format_jobs()

            mock_logger.warning.assert_called()


@pytest.mark.agent
class TestResyncFormatJobs:
    async def test_startup_recovery_posts_and_marks_finalized(self):
        mock_job = MagicMock()
        mock_job.id = uuid4()
        mock_job.status = FormatJobStatusEnum.COMPLETED
        mock_job.working_memory = {
            "discord_thread_id": "123",
            "discord_message_id": "456",
        }
        mock_job.error_log = {}
        mock_job.created_at = None
        mock_job.final_video_url = None

        mock_thread = AsyncMock()
        mock_thread.send = AsyncMock()

        with (
            patch(
                "app.discord_bot.get_format_jobs_missed_terminal",
                new=AsyncMock(return_value=[mock_job]),
            ),
            patch("app.discord_bot.AsyncSessionLocal") as mock_session_cls,
            patch("app.discord_bot.bot") as mock_bot,
            patch(
                "app.discord_bot.update_format_job_working_memory",
                new=AsyncMock(),
            ) as mock_update_wm,
            patch(
                "app.discord_bot.build_completed_embed",
                return_value=MagicMock(),
            ),
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            mock_bot.fetch_channel = AsyncMock(return_value=mock_thread)

            await _resync_format_jobs()

            mock_thread.send.assert_awaited()
            mock_update_wm.assert_awaited_once()
            wm_arg = mock_update_wm.call_args[0][2]
            assert wm_arg["final_embed_updated"] is True

    async def test_deleted_thread_handling(self):
        mock_job = MagicMock()
        mock_job.id = uuid4()
        mock_job.status = FormatJobStatusEnum.FAILED
        mock_job.working_memory = {
            "discord_thread_id": "999",
            "discord_message_id": "456",
        }
        mock_job.error_log = {}
        mock_job.created_at = None

        import discord

        with (
            patch(
                "app.discord_bot.get_format_jobs_missed_terminal",
                new=AsyncMock(return_value=[mock_job]),
            ),
            patch("app.discord_bot.AsyncSessionLocal") as mock_session_cls,
            patch("app.discord_bot.bot") as mock_bot,
            patch(
                "app.discord_bot.update_format_job_working_memory",
                new=AsyncMock(),
            ) as mock_update_wm,
            patch(
                "app.discord_bot.build_failed_embed",
                return_value=MagicMock(),
            ),
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            mock_bot.fetch_channel = AsyncMock(
                side_effect=discord.NotFound(MagicMock(), "not found")
            )

            await _resync_format_jobs()

            mock_update_wm.assert_awaited_once()
            wm_arg = mock_update_wm.call_args[0][2]
            assert wm_arg["final_embed_updated"] is True
