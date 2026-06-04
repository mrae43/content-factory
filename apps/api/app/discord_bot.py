"""
Discord bot for Content Factory.
Standalone process: python -m app.discord_bot

Slash commands:
  /script <title> [user_reference] [source_urls]
    Runs the full script content pipeline inline and posts results.
"""

import asyncio
import io
import logging
from datetime import datetime, timezone
from uuid import UUID

import aiohttp
import discord
from discord.ext import commands

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.script_crud import (
    create_script_job,
    get_script_job,
    get_stuck_script_jobs,
    log_script_job_error,
    update_script_job_status,
)
from app.db.format_crud import (
    create_format_job,
    get_format_jobs_for_watcher,
    get_format_jobs_missed_terminal,
    update_format_job_working_memory,
)
from app.schemas.shorts import ScriptJobStatusEnum
from app.services.script_pipeline import ScriptPipelineRunner
from app.discord_embeds import (
    build_format_embed,
    build_completed_embed,
    build_failed_embed,
)
from app.discord_ui import ShortFormatSelectionView, RetryCompositionButton

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
                if job.status == ScriptJobStatusEnum.COMPLETED:
                    await _post_completion(thread, job)
                elif job.status == ScriptJobStatusEnum.HUMAN_REVIEW_NEEDED:
                    await thread.send(
                        "⚠️ **Pipeline escalated** — human review is required."
                    )
                elif job.status == ScriptJobStatusEnum.FAILED:
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

    @discord.ui.button(
        label="Create Short",
        style=discord.ButtonStyle.success,
        custom_id="format_short",
    )
    async def short_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        view = ShortFormatSelectionView(self.script_job_id)
        await interaction.response.send_message(
            "**Create Short Video**\nSelect platform, style, and options:",
            view=view,
            ephemeral=False,
        )


# ── Platform Modal ──────────────────────────────────────────────────────

VALID_PLATFORMS = {"tiktok", "youtube", "instagram", "twitter", "linkedin"}


class PlatformModal(discord.ui.Modal):
    def __init__(self, script_job_id: UUID, format_type: str):
        super().__init__(title=f"Create {format_type.title()}")
        self.script_job_id = script_job_id
        self.format_type = format_type

        self.platform_input = discord.ui.TextInput(
            label="Platform",
            placeholder="tiktok, youtube, instagram, twitter, linkedin",
            min_length=1,
            max_length=20,
            required=True,
        )
        self.add_item(self.platform_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)
        raw_platform = (self.platform_input.value or "").strip().lower()
        platform = raw_platform if raw_platform in VALID_PLATFORMS else "instagram"

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
                    format_type=self.format_type,
                    snapshot_data=snapshot_data,
                    working_memory={},
                )

                # Step 2: Create Discord thread + post initial embed
                thread = await interaction.channel.create_thread(
                    name=f"🎬｜{self.format_type}-gen-{str(fmt_job.id)[:8]}",
                    type=discord.ChannelType.public_thread,
                    auto_archive_duration=60,
                )
                embed = build_format_embed(
                    fmt_job,
                    elapsed_seconds=0,
                    format_label=self.format_type.title(),
                )
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
                    f"✅ **{self.format_type.title()}** job created for **{platform}**!\n"
                    f"Track progress in {thread.mention}",
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
    await _resync_format_jobs()

    global _format_job_watcher
    _format_job_watcher = FormatJobWatcher()
    await _format_job_watcher.start()


async def _safe_resume(runner: ScriptPipelineRunner, job_id: UUID) -> None:
    """Resume a recovered pipeline, guarding the bot event loop."""
    async with _script_semaphore:
        try:
            await runner.run()
        except Exception:
            logger.exception("Crash recovery failed for ScriptJob %s", job_id)


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
                getattr(job.status, "value", str(job.status)),
            )

            thread_id = None
            if job.working_memory:
                thread_id = job.working_memory.get("discord_thread_id")

            if thread_id:
                try:
                    thread = await bot.fetch_channel(int(thread_id))
                    notifier = DiscordProgressNotifier(thread)
                    runner = ScriptPipelineRunner(db, job.id, notifier)
                    bot.loop.create_task(_safe_resume(runner, job.id))
                    logger.info("Resuming ScriptJob %s in thread %s", job.id, thread_id)
                except discord.NotFound:
                    logger.warning(
                        "Thread %s deleted for ScriptJob %s", thread_id, job.id
                    )
                    await log_script_job_error(
                        db,
                        job.id,
                        f"Discord thread {thread_id} was deleted during bot downtime",
                        "crash_recovery",
                    )
                    await update_script_job_status(
                        db, job.id, ScriptJobStatusEnum.FAILED
                    )
            else:
                logger.warning("No thread_id for ScriptJob %s; cannot resume", job.id)
                await log_script_job_error(
                    db,
                    job.id,
                    "Missing discord_thread_id in working_memory; cannot resume after crash",
                    "crash_recovery",
                )
                await update_script_job_status(db, job.id, ScriptJobStatusEnum.FAILED)


class FormatJobWatcher:
    """Background task that polls format_jobs and updates Living Embeds."""

    def __init__(self):
        self._previous_states: dict[UUID, str] = {}
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("FormatJob Watcher started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("FormatJob Watcher stopped")

    async def _poll_loop(self):
        while self._running:
            try:
                await self._check_format_jobs()
            except Exception:
                logger.exception("FormatJob watcher poll error")
            await asyncio.sleep(5)

    async def _check_format_jobs(self):
        async with AsyncSessionLocal() as db:
            jobs = await get_format_jobs_for_watcher(db)

            for job in jobs:
                prev = self._previous_states.get(job.id)
                current = (
                    job.status.value
                    if hasattr(job.status, "value")
                    else str(job.status)
                )

                if prev != current:
                    await self._update_living_embed(job, current)
                    self._previous_states[job.id] = current

                # Terminal states
                if current in ("COMPLETED", "FAILED"):
                    await self._finalize_job(job, current)
                    self._previous_states.pop(job.id, None)

    async def _update_living_embed(self, job, status: str) -> None:
        wm = job.working_memory or {}
        thread_id = wm.get("discord_thread_id")
        message_id = wm.get("discord_message_id")
        if not thread_id or not message_id:
            return

        try:
            thread = await bot.fetch_channel(int(thread_id))
            msg = await thread.fetch_message(int(message_id))

            elapsed = None
            if job.created_at:
                elapsed = (datetime.now(timezone.utc) - job.created_at).total_seconds()

            embed = build_format_embed(job, elapsed_seconds=elapsed)
            await msg.edit(embed=embed)
        except discord.NotFound:
            logger.warning("Thread/message not found for FormatJob %s", job.id)
        except Exception:
            logger.exception("Failed to update embed for FormatJob %s", job.id)

    async def _finalize_job(self, job, status: str) -> None:
        wm = job.working_memory or {}
        thread_id = wm.get("discord_thread_id")
        if not thread_id:
            return

        try:
            thread = await bot.fetch_channel(int(thread_id))

            elapsed = None
            if job.created_at:
                elapsed = (datetime.now(timezone.utc) - job.created_at).total_seconds()

            if status == "COMPLETED":
                embed = build_completed_embed(job, elapsed_seconds=elapsed)
                await thread.send(embed=embed)

                # Attempt file upload (fallback to link if >25MB)
                if job.final_video_url:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(job.final_video_url) as resp:
                                if (
                                    resp.status == 200
                                    and int(resp.headers.get("Content-Length", 0))
                                    < 25 * 1024 * 1024
                                ):
                                    data = await resp.read()
                                    await thread.send(
                                        file=discord.File(
                                            io.BytesIO(data),
                                            filename=f"short-{str(job.id)[:8]}.mp4",
                                        )
                                    )
                    except Exception as exc:
                        logger.warning(
                            "Failed to upload video for FormatJob %s: %s",
                            job.id,
                            exc,
                        )
            elif status == "FAILED":
                embed = build_failed_embed(job, elapsed_seconds=elapsed)
                view = discord.ui.View()
                has_composition_error = any(
                    "COMPOSITION" in k.upper() for k in (job.error_log or {}).keys()
                )
                if has_composition_error:
                    view.add_item(RetryCompositionButton(job.id))
                await thread.send(embed=embed, view=view)

            # Mark as finalized
            wm["final_embed_updated"] = True
            async with AsyncSessionLocal() as db:
                await update_format_job_working_memory(db, job.id, wm)

        except discord.NotFound:
            logger.warning(
                "Thread %s not found for FormatJob %s; marking finalized",
                thread_id,
                job.id,
            )
            wm["final_embed_updated"] = True
            async with AsyncSessionLocal() as db:
                await update_format_job_working_memory(db, job.id, wm)
        except Exception:
            logger.exception("Failed to finalize FormatJob %s", job.id)


# Create watcher instance
_format_job_watcher: FormatJobWatcher | None = None


async def _resync_format_jobs():
    """Post final result embeds for FormatJobs missed during downtime."""
    logger.info("Checking for missed terminal FormatJobs...")
    async with AsyncSessionLocal() as db:
        missed = await get_format_jobs_missed_terminal(db)
        logger.info("Found %d missed terminal FormatJobs", len(missed))

        for job in missed:
            wm = job.working_memory or {}
            thread_id = wm.get("discord_thread_id")
            if not thread_id:
                continue

            try:
                thread = await bot.fetch_channel(int(thread_id))
                status = (
                    job.status.value
                    if hasattr(job.status, "value")
                    else str(job.status)
                )

                elapsed = None
                if job.created_at:
                    elapsed = (
                        datetime.now(timezone.utc) - job.created_at
                    ).total_seconds()

                if status == "FAILED":
                    embed = build_failed_embed(job, elapsed_seconds=elapsed)
                    view = discord.ui.View()
                    has_composition_error = any(
                        "COMPOSITION" in k.upper() for k in (job.error_log or {}).keys()
                    )
                    if has_composition_error:
                        view.add_item(RetryCompositionButton(job.id))
                    await thread.send(embed=embed, view=view)
                else:
                    embed = build_completed_embed(job, elapsed_seconds=elapsed)
                    await thread.send(embed=embed)

                wm["final_embed_updated"] = True
                await update_format_job_working_memory(db, job.id, wm)
                logger.info("Re-synced FormatJob %s to thread %s", job.id, thread_id)

            except discord.NotFound:
                logger.warning(
                    "Thread %s not found for FormatJob %s; marking finalized",
                    thread_id,
                    job.id,
                )
                wm["final_embed_updated"] = True
                await update_format_job_working_memory(db, job.id, wm)
            except Exception:
                logger.exception("Failed to re-sync FormatJob %s", job.id)


async def main():
    if not settings.discord_token or not settings.discord_guild_id:
        logger.error("DISCORD_TOKEN and DISCORD_GUILD_ID must be set in .env")
        return
    async with bot:
        try:
            await bot.start(settings.discord_token)
        finally:
            global _format_job_watcher
            if _format_job_watcher:
                await _format_job_watcher.stop()


if __name__ == "__main__":
    asyncio.run(main())
