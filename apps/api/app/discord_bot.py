"""
Discord bot for Content Factory.
Standalone process: python -m app.discord_bot
"""
import asyncio
import logging

import discord
from discord.ext import commands

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()

bot = commands.Bot(command_prefix="!", intents=intents)
GUILD = discord.Object(id=settings.discord_guild_id)


@bot.tree.command(
    guild=GUILD, name="ping", description="Check if the bot is alive"
)
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("pong!")


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


async def main():
    async with bot:
        await bot.start(settings.discord_token)


if __name__ == "__main__":
    asyncio.run(main())
