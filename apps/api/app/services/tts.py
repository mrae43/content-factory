from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import aiohttp
from pydantic import BaseModel

from app.core.config import settings
from app.services.tools import Tool

logger = logging.getLogger("factory.tts")

_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"


class TTSResult(BaseModel):
    """Result of a text-to-speech generation."""

    audio_bytes: bytes
    vocal_alignment_data: list[dict]
    duration_seconds: float


class TTSProvider(ABC):
    """Abstract base for TTS providers."""

    @abstractmethod
    async def generate_voiceover(self, text: str, voice_id: str) -> TTSResult:
        """Generate voiceover audio and word-level alignment data."""
        ...


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs TTS implementation using aiohttp."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_voiceover(self, text: str, voice_id: str) -> TTSResult:
        url = f"{_ELEVENLABS_BASE_URL}/v1/text-to-speech/{voice_id}/with-timestamps"
        payload = {
            "text": text,
            "model_id": settings.tts_model_id,
        }
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=(),
                        status=resp.status,
                        message=body[:500],
                        headers=resp.headers,
                    )
                data = await resp.json()

        audio_base64 = data.get("audio_base64", "")
        if not audio_base64:
            raise ValueError("ElevenLabs response missing audio_base64")

        audio_bytes = base64.b64decode(audio_base64)

        alignment = data.get("alignment", {})
        characters = alignment.get("characters", [])
        start_times = alignment.get("character_start_times_seconds", [])
        end_times = alignment.get("character_end_times_seconds", [])

        vocal_alignment_data = _chars_to_word_timestamps(
            characters, start_times, end_times
        )

        duration_seconds = max(end_times) if end_times else 0.0

        return TTSResult(
            audio_bytes=audio_bytes,
            vocal_alignment_data=vocal_alignment_data,
            duration_seconds=duration_seconds,
        )


def _chars_to_word_timestamps(
    characters: list[str],
    start_times: list[float],
    end_times: list[float],
) -> list[dict]:
    """Convert ElevenLabs character-level timestamps to word-level timestamps."""
    words: list[dict] = []
    current_word_chars: list[str] = []
    current_word_starts: list[float] = []
    current_word_ends: list[float] = []

    for char, start, end in zip(characters, start_times, end_times):
        if char == " ":
            if current_word_chars:
                words.append(
                    {
                        "word": "".join(current_word_chars),
                        "start": current_word_starts[0],
                        "end": current_word_ends[-1],
                    }
                )
                current_word_chars = []
                current_word_starts = []
                current_word_ends = []
        else:
            current_word_chars.append(char)
            current_word_starts.append(start)
            current_word_ends.append(end)

    if current_word_chars:
        words.append(
            {
                "word": "".join(current_word_chars),
                "start": current_word_starts[0],
                "end": current_word_ends[-1],
            }
        )

    return words


# ---------------------------------------------------------------------------
# Provider registry (mirrors VIDEO_GEN_PROVIDERS pattern)
# ---------------------------------------------------------------------------

TTS_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "elevenlabs": {
        "class": ElevenLabsTTS,
        "api_key_attr": "tts_api_key",
    },
}

_DEFAULT_TTS_PROVIDER = "elevenlabs"


def _resolve_tts_provider(provider_name: str = "") -> tuple[str, dict]:
    if not provider_name:
        provider_name = _DEFAULT_TTS_PROVIDER
    config = TTS_PROVIDERS.get(provider_name)
    if config is None:
        raise ValueError(
            f"Unknown TTS provider '{provider_name}'. "
            f"Available: {', '.join(TTS_PROVIDERS)}."
        )
    return provider_name, config


_tts_provider_cache: Dict[str, TTSProvider] = {}


def get_tts_provider(provider_name: str = "") -> TTSProvider:
    key = provider_name or _DEFAULT_TTS_PROVIDER
    if key not in _tts_provider_cache:
        _, config = _resolve_tts_provider(key)
        api_key = getattr(settings, config["api_key_attr"], None)
        _tts_provider_cache[key] = config["class"](api_key=api_key)
    return _tts_provider_cache[key]


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


def make_generate_voiceover_tool() -> Tool:
    async def _generate(text: str, voice_id: str) -> dict:
        provider = get_tts_provider(settings.tts_provider)
        result = await provider.generate_voiceover(text=text, voice_id=voice_id)
        return {
            "audio_bytes": result.audio_bytes,
            "vocal_alignment_data": result.vocal_alignment_data,
            "duration_seconds": result.duration_seconds,
        }

    return Tool(
        name="generate_voiceover",
        description="Generate a voiceover audio file and word-level timestamps via the configured TTS provider.",
        callable=_generate,
        permissions={"ShortVoiceoverAgent", "*"},
    )


def make_get_alignment_tool() -> Tool:
    async def _get_alignment(voiceover_id: str = "") -> dict:
        # ElevenLabs returns alignment data inline with generation.
        # This tool is a no-op for ElevenLabs; future providers may implement
        # a separate alignment retrieval endpoint.
        return {
            "message": (
                "Alignment data is returned inline by generate_voiceover. "
                "No separate alignment retrieval is needed for the current provider."
            ),
            "vocal_alignment_data": [],
        }

    return Tool(
        name="get_alignment",
        description="Retrieve word-level vocal alignment data for a voiceover. (No-op for ElevenLabs — data is inline.)",
        callable=_get_alignment,
        permissions={"ShortVoiceoverAgent", "*"},
    )
