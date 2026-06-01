"""
Discord bot for Content Factory.
Standalone process: python -m app.discord_bot

Slash commands:
  /script <title> [user_reference] [source_urls]
    Runs the full script content pipeline inline and posts results.
"""

import asyncio
import logging
from uuid import UUID

import discord
from discord.ext import commands

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.script_crud import (
    create_script_job,
    get_script_job,
    get_stuck_script_jobs,
)
from app.db.format_crud import create_format_job
from app.services.script_pipeline import ScriptPipelineRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)
GUILD = discord.Object(id=settings.discord_guild_id)

# Limit concurrent pipeline runs to avoid OOM
_script_semaphore = asyncio.Semaphore(3)


# ── Progress Notifier (Discord implementation) ──────────────────────────


class DiscordProgressNotifier:
    def __init__(self, thread: discord.Thread):
        self._thread = thread

    async def notify(self, message: str) -> None:
        await self._thread.send(message)


# ── Slash Command: /script ──────────────────────────────────────────────


@bot.tree.command(
    guild=GUILD,
    name="script",
    description="Generate a script on any topic with optional reference material",
)
async def script(
    interaction: discord.Interaction,
    title: str,
    user_reference: str = "",
    source_urls: str = "",
):
    await interaction.response.defer(ephemeral=False)

    parsed_urls = (
        [u.strip() for u in source_urls.split(",") if u.strip()] if source_urls else []
    )
    bot.loop.create_task(
        _run_script_pipeline(interaction, title, user_reference, parsed_urls)
    )


async def _run_script_pipeline(
    interaction: discord.Interaction,
    title: str,
    user_reference: str,
    source_urls: list[str],
):
    async with _script_semaphore:
        try:
            job = None
            thread = None
            async with AsyncSessionLocal() as db:
                job = await create_script_job(
                    db,
                    title=title,
                    user_reference=user_reference,
                    source_urls=source_urls,
                )

                msg = await interaction.followup.send(
                    f"🎬 **Starting script generation: *{title}***",
                    wait=True,
                )
                thread = await msg.create_thread(
                    name=f"Script: {title[:90]}",
                    auto_archive_duration=60,
                )

                working_memory = dict(job.working_memory or {})
                working_memory["discord_thread_id"] = str(thread.id)
                job.working_memory = working_memory
                await db.commit()

                notifier = DiscordProgressNotifier(thread)
                runner = ScriptPipelineRunner(db, job.id, notifier)
                await runner.run()

                job = await get_script_job(db, job.id)

            if job and thread:
                if job.status.value == "COMPLETED":
                    await _post_completion(thread, job)
                elif job.status.value == "HUMAN_REVIEW_NEEDED":
                    await thread.send(
                        "⚠️ **Pipeline escalated** — human review is required."
                    )
                elif job.status.value == "FAILED":
                    error_log = job.error_log or {}
                    error_msg = (
                        list(error_log.values())[0].get("message", "Unknown error")
                        if error_log
                        else "Unknown error"
                    )
                    await thread.send(f"❌ **Pipeline failed**: {error_msg[:2000]}")
        except Exception:
            logger.exception("Script pipeline crashed")
            try:
                await interaction.followup.send(
                    "❌ An unexpected error occurred while generating the script.",
                    ephemeral=True,
                )
            except Exception:
                pass


async def _post_completion(thread: discord.Thread, job) -> None:
    script_preview = (job.script_content or "")[:500]
    claims = job.claims or []
    total = len(claims)
    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")

    embed = discord.Embed(
        title="✅ Script Complete",
        description=f"**{job.title}**",
        color=discord.Color.green(),
    )
    embed.add_field(
        name="Script Preview", value=script_preview or "No content", inline=False
    )
    embed.add_field(name="Claims", value=f"{supported}/{total} verified", inline=True)
    embed.set_footer(text="Select a format below to create your output")

    await thread.send(embed=embed)

    view = FormatSelectionView(job.id)
    await thread.send("**Choose a format:**", view=view)


# ── Format Selection View ───────────────────────────────────────────────


class FormatSelectionView(discord.ui.View):
    def __init__(self, script_job_id: UUID):
        super().__init__(timeout=300)
        self.script_job_id = script_job_id

    @discord.ui.button(
        label="Create Carousel",
        style=discord.ButtonStyle.primary,
        custom_id="format_carousel",
    )
    async def carousel_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        await interaction.response.send_modal(
            PlatformModal(self.script_job_id, "carousel")
        )

    @discord.ui.button(
        label="Create Video",
        style=discord.ButtonStyle.secondary,
        custom_id="format_video",
    )
    async def video_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        await interaction.response.send_modal(
            PlatformModal(self.script_job_id, "video")
        )

    @discord.ui.button(
        label="Create Blog",
        style=discord.ButtonStyle.secondary,
        custom_id="format_blog",
    )
    async def blog_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        await interaction.response.send_modal(PlatformModal(self.script_job_id, "blog"))


# ── Platform Modal ──────────────────────────────────────────────────────


PLATFORM_CHOICES = [
    discord.SelectOption(
        label="TikTok", value="tiktok", description="Short-form vertical video"
    ),
    discord.SelectOption(
        label="YouTube", value="youtube", description="Long-form horizontal video"
    ),
    discord.SelectOption(
        label="Instagram", value="instagram", description="Square carousel"
    ),
    discord.SelectOption(
        label="Twitter / X", value="twitter", description="Wide carousel"
    ),
    discord.SelectOption(
        label="LinkedIn", value="linkedin", description="Professional carousel"
    ),
]


class PlatformModal(discord.ui.Modal):
    def __init__(self, script_job_id: UUID, format_type: str):
        super().__init__(title=f"Create {format_type.title()}")
        self.script_job_id = script_job_id
        self.format_type = format_type

        self.platform = discord.ui.Select(
            placeholder="Choose a platform...",
            options=PLATFORM_CHOICES,
            min_values=1,
            max_values=1,
        )
        self.add_item(self.platform)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)
        platform = self.platform.values[0] if self.platform.values else "instagram"

        try:
            async with AsyncSessionLocal() as db:
                job = await get_script_job(db, self.script_job_id)
                if not job:
                    await interaction.followup.send(
                        "❌ Script job not found. It may have been deleted.",
                        ephemeral=True,
                    )
                    return

                snapshot_data = {
                    "title": job.title,
                    "script_content": job.script_content,
                    "claims": job.claims,
                    "refined_context": job.refined_context,
                    "story_directives": job.story_directives,
                    "hedge_index": job.hedge_index,
                    "epistemic_ledger": None,
                }

                fmt_job = await create_format_job(
                    db,
                    source_job_id=self.script_job_id,
                    platform=platform,
                    format_type=self.format_type,
                    snapshot_data=snapshot_data,
                )

                await interaction.followup.send(
                    f"✅ **{self.format_type.title()}** job created for **{platform}**!\n"
                    f"Job ID: `{fmt_job.id}`\n"
                    f"The QueueWorker will process it shortly.",
                )
        except Exception as exc:
            logger.exception("Failed to create format job")
            await interaction.followup.send(
                f"❌ Failed to create format job: {exc}",
                ephemeral=True,
            )


# ── Bot Lifecycle ───────────────────────────────────────────────────────


async def setup_hook():
    guild_cmds = bot.tree.get_commands(guild=GUILD)
    logger.info("Guild-local tree has %d commands", len(guild_cmds))
    for cmd in guild_cmds:
        logger.info("  - /%s", cmd.name)
    if not guild_cmds:
        logger.warning("No guild-local commands — check guild= arg on decorator")
        return
    try:
        synced = await bot.tree.sync(guild=GUILD)
        logger.info(
            "Synced %d commands to guild %s", len(synced), settings.discord_guild_id
        )
    except discord.Forbidden:
        logger.error(
            "Sync failed with 403 Forbidden. The bot lacks 'applications.commands' "
            "OAuth2 scope. Re-invite with both 'bot' and 'applications.commands' scopes."
        )
    except Exception as exc:
        logger.error("Sync failed: %s", exc)


bot.setup_hook = setup_hook


@bot.event
async def on_ready():
    logger.info("Bot logged in as %s", bot.user)
    for g in bot.guilds:
        logger.info("  - %s (%s)", g.name, g.id)

    await recover_stuck_script_jobs()


async def recover_stuck_script_jobs():
    """Release stale locks and resume any stuck script pipelines on startup."""
    logger.info("Checking for stuck script jobs...")
    async with AsyncSessionLocal() as db:
        stuck = await get_stuck_script_jobs(db, timeout_minutes=15)
        logger.info("Found %d stuck script jobs", len(stuck))
        for job in stuck:
            job.locked_at = None
            job.locked_by = None
            await db.commit()
            logger.info(
                "Released lock on stuck ScriptJob %s (status=%s)",
                job.id,
                job.status.value,
            )


async def main():
    if not settings.discord_token or not settings.discord_guild_id:
        logger.error("DISCORD_TOKEN and DISCORD_GUILD_ID must be set in .env")
        return
    async with bot:
        await bot.start(settings.discord_token)


if __name__ == "__main__":
    asyncio.run(main())
