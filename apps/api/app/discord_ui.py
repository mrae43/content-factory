"""Discord UI components for SHORT format selection."""

import logging
from uuid import UUID

import discord

from app.db.session import AsyncSessionLocal
from app.db.script_crud import get_script_job
from app.db.format_crud import (
    create_format_job,
    update_format_job_working_memory,
    reset_format_job_to_composition,
)
from app.services.short_config import (
    DEFAULT_VOICE_MAP,
    DEFAULT_SUBTITLE_PRESET_MAP,
    PLATFORM_ASPECT_RATIOS_SHORT,
)
from app.discord_embeds import build_format_embed

logger = logging.getLogger(__name__)

SHORT_PLATFORMS = {
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "youtube": "YouTube Shorts",
}

VISUAL_STYLE_THEMES = [
    ("cinematic", "Cinematic", "Dramatic lighting, wide shots, film grain"),
    ("minimalist", "Minimalist", "Clean, sparse, neutral colors"),
    ("newsroom", "Newsroom", "Studio lighting, talking-head, infographics"),
    ("documentary", "Documentary", "Natural lighting, handheld, raw footage"),
    ("dynamic", "Dynamic", "Fast cuts, vibrant colors, motion graphics"),
]


class ShortFormatSelectionView(discord.ui.View):
    """View with platform select, visual style select, loopable toggle, confirm."""

    def __init__(self, script_job_id: UUID):
        super().__init__(timeout=300)
        self.script_job_id = script_job_id
        self.loopable = True
        self._platform = "tiktok"
        self._visual_style_theme = "cinematic"

    # ── Platform Select ──────────────────────────────────────────────

    @discord.ui.select(
        placeholder="Choose platform...",
        options=[
            discord.SelectOption(
                label="TikTok",
                value="tiktok",
                description="9:16 vertical video",
                emoji="🎵",
            ),
            discord.SelectOption(
                label="Instagram",
                value="instagram",
                description="4:5 vertical video",
                emoji="📸",
            ),
            discord.SelectOption(
                label="YouTube Shorts",
                value="youtube",
                description="9:16 vertical video",
                emoji="▶️",
            ),
        ],
        custom_id="short_platform",
    )
    async def platform_select(
        self, interaction: discord.Interaction, select: discord.ui.Select
    ):
        self._platform = select.values[0]
        await interaction.response.defer()

    # ── Visual Style Theme Select ────────────────────────────────────

    @discord.ui.select(
        placeholder="Choose visual style...",
        options=[
            discord.SelectOption(label=label, value=value, description=desc)
            for value, label, desc in VISUAL_STYLE_THEMES
        ],
        custom_id="short_style",
    )
    async def style_select(
        self, interaction: discord.Interaction, select: discord.ui.Select
    ):
        self._visual_style_theme = select.values[0]
        await interaction.response.defer()

    # ── Loopable Toggle ──────────────────────────────────────────────

    @discord.ui.button(
        label="Loopable: On",
        style=discord.ButtonStyle.primary,
        custom_id="short_loopable",
    )
    async def loopable_toggle(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.loopable = not self.loopable
        button.label = f"Loopable: {'On' if self.loopable else 'Off'}"
        button.style = (
            discord.ButtonStyle.primary
            if self.loopable
            else discord.ButtonStyle.secondary
        )
        await interaction.response.edit_message(view=self)

    # ── Confirm Button ───────────────────────────────────────────────

    @discord.ui.button(
        label="Confirm & Generate",
        style=discord.ButtonStyle.success,
        custom_id="short_confirm",
        row=3,
    )
    async def confirm_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        await interaction.response.defer(ephemeral=False)

        platform = self._platform
        voice_id = DEFAULT_VOICE_MAP.get(platform, "")
        aspect = PLATFORM_ASPECT_RATIOS_SHORT.get(platform, (1080, 1920))
        subtitle = DEFAULT_SUBTITLE_PRESET_MAP.get(platform, "CENTER_POP_YELLOW")

        try:
            async with AsyncSessionLocal() as db:
                job = await get_script_job(db, self.script_job_id)
                if not job:
                    await interaction.followup.send(
                        "❌ Script job not found.",
                        ephemeral=True,
                    )
                    return

                # Enrich story_directives
                enriched = dict(job.story_directives or {})
                enriched.update(
                    {
                        "voice_id": voice_id,
                        "loopable": self.loopable,
                        "visual_style_theme": self._visual_style_theme,
                        "aspect_ratio": f"{aspect[0]}x{aspect[1]}",
                        "subtitle_preset": subtitle,
                    }
                )

                snapshot_data = {
                    "title": job.title,
                    "script_content": job.script_content,
                    "claims": job.claims,
                    "refined_context": job.refined_context,
                    "story_directives": enriched,
                    "hedge_index": job.hedge_index,
                    "epistemic_ledger": (
                        job.working_memory.get("epistemic_ledger")
                        if job.working_memory
                        else None
                    ),
                }

                # Step 1: Insert FormatJob with empty working_memory
                fmt_job = await create_format_job(
                    db,
                    source_job_id=self.script_job_id,
                    platform=platform,
                    format_type="short",
                    snapshot_data=snapshot_data,
                    working_memory={},
                )

                # Step 2: Create Discord thread + post initial embed
                thread = await interaction.channel.create_thread(
                    name=f"🎬｜short-gen-{str(fmt_job.id)[:8]}",
                    type=discord.ChannelType.public_thread,
                    auto_archive_duration=60,
                )
                embed = build_format_embed(fmt_job, elapsed_seconds=0)
                msg = await thread.send(embed=embed)

                # Step 3: Update working_memory with thread/message IDs
                await update_format_job_working_memory(
                    db,
                    fmt_job.id,
                    {
                        "discord_thread_id": str(thread.id),
                        "discord_message_id": str(msg.id),
                    },
                )

                await interaction.followup.send(
                    f"✅ **Short video job created!**\n"
                    f"Track progress in {thread.mention}",
                )

        except Exception as exc:
            logger.exception("Failed to create SHORT format job")
            await interaction.followup.send(
                f"❌ Failed to create Short job: {exc}",
                ephemeral=True,
            )


class RetryCompositionButton(discord.ui.Button):
    """Button to retry composition from a failed state."""

    def __init__(self, format_job_id: UUID):
        super().__init__(
            label="Retry Composition",
            style=discord.ButtonStyle.primary,
            custom_id=f"retry_composition_{format_job_id}",
        )
        self.format_job_id = format_job_id

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        try:
            async with AsyncSessionLocal() as db:
                job = await reset_format_job_to_composition(db, self.format_job_id)
                if not job:
                    await interaction.followup.send(
                        "❌ Format job not found.",
                        ephemeral=True,
                    )
                    return

                # Edit the Living Embed to show retry state
                embed = build_format_embed(job, elapsed_seconds=0)
                embed.title = "🔄 Retrying composition..."
                await interaction.edit_original_response(embed=embed, view=None)

                await interaction.followup.send(
                    "🔄 **Composition retry queued!**\n"
                    "The QueueWorker will pick it up shortly.",
                    ephemeral=True,
                )
        except Exception as exc:
            logger.exception("Retry composition failed")
            await interaction.followup.send(
                f"❌ Retry failed: {exc}",
                ephemeral=True,
            )
