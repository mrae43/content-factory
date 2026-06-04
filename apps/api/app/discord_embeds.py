"""Embed builders for FormatJob Living Embed."""

from typing import Optional

import discord

from app.db.discord_models import FormatJob


def build_format_embed(
    format_job: FormatJob,
    elapsed_seconds: Optional[float] = None,
    format_label: str = "Short Video",
) -> discord.Embed:
    """Build a single Living Embed for the given FormatJob state."""
    status = format_job.status
    if hasattr(status, "value"):
        status = status.value

    status_labels = {
        "PENDING": ("🕐 Pending", discord.Color.light_grey(), "1/4"),
        "FORMATTING": ("🎨 Formatting...", discord.Color.blue(), "2/4"),
        "ASSET_GENERATION": ("🎬 Generating Assets...", discord.Color.gold(), "3/4"),
        "COMPOSITION": ("🎞️ Composing Video...", discord.Color.purple(), "4/4"),
        "COMPLETED": ("✅ Completed", discord.Color.green(), "4/4"),
        "FAILED": ("❌ Failed", discord.Color.red(), "—"),
        "HUMAN_REVIEW_NEEDED": ("⚠️ Human Review", discord.Color.orange(), "—"),
    }

    label, color, progress = status_labels.get(
        status, ("❓ Unknown", discord.Color.default(), "—")
    )

    embed = discord.Embed(
        title=f"{format_label} — {label}",
        color=color,
    )
    embed.add_field(
        name="Platform",
        value=format_job.platform or "Unknown",
        inline=True,
    )
    embed.add_field(
        name="Format",
        value=format_job.format_type or "Unknown",
        inline=True,
    )
    embed.add_field(
        name="Progress",
        value=progress,
        inline=True,
    )

    if elapsed_seconds is not None:
        minutes = int(elapsed_seconds) // 60
        seconds = int(elapsed_seconds) % 60
        if minutes > 0:
            embed.add_field(
                name="Duration",
                value=f"{minutes}m {seconds}s",
                inline=True,
            )
        else:
            embed.add_field(
                name="Duration",
                value=f"{seconds}s",
                inline=True,
            )

    return embed


def build_completed_embed(
    format_job: FormatJob,
    elapsed_seconds: Optional[float] = None,
) -> discord.Embed:
    """Build a completion embed with video URL."""
    embed = build_format_embed(format_job, elapsed_seconds)
    embed.title = "✅ Short Video Complete"
    if format_job.final_video_url:
        embed.add_field(
            name="Video URL",
            value=format_job.final_video_url,
            inline=False,
        )
    return embed


def build_failed_embed(
    format_job: FormatJob,
    elapsed_seconds: Optional[float] = None,
) -> discord.Embed:
    """Build a failure embed with error summary."""
    embed = build_format_embed(format_job, elapsed_seconds)
    embed.title = "❌ Short Video Failed"

    error_log = format_job.error_log or {}
    if error_log:
        last_phase = list(error_log.keys())[-1]
        last_error = error_log[last_phase]
        embed.add_field(
            name=f"Error ({last_phase})",
            value=last_error.get("message", "Unknown error")[:1000],
            inline=False,
        )

    return embed
