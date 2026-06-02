import pytest
from unittest.mock import MagicMock

from app.discord_bot import PlatformModal


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

        # Simulate on_submit extracting the platform
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
