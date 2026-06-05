import base64
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from app.services.tts import (
    ElevenLabsTTS,
    TTSProvider,
    TTSResult,
    _chars_to_word_timestamps,
    get_tts_provider,
    TTS_PROVIDERS,
)


def _dummy_b64_audio(size: int = 2048) -> str:
    return base64.b64encode(b"\x00" * size).decode("ascii")


def _make_elevenlabs_response(
    audio_b64: str = "",
    characters: list[str] | None = None,
    char_starts: list[float] | None = None,
    char_ends: list[float] | None = None,
):
    return {
        "audio_base64": audio_b64 or _dummy_b64_audio(),
        "alignment": {
            "characters": characters or [],
            "character_start_times_seconds": char_starts or [],
            "character_end_times_seconds": char_ends or [],
        },
    }


@pytest.mark.unit
class TestCharsToWordTimestamps:
    def test_simple_sentence(self):
        characters = list("Hello world")
        starts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ends = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        result = _chars_to_word_timestamps(characters, starts, ends)
        assert len(result) == 2
        assert result[0]["word"] == "Hello"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 0.5
        assert result[1]["word"] == "world"
        assert result[1]["start"] == 0.6
        assert result[1]["end"] == 1.1

    def test_trailing_space_ignored(self):
        characters = list("Hi ")
        starts = [0.0, 0.1, 0.2]
        ends = [0.1, 0.2, 0.3]
        result = _chars_to_word_timestamps(characters, starts, ends)
        assert len(result) == 1
        assert result[0]["word"] == "Hi"

    def test_empty_input(self):
        result = _chars_to_word_timestamps([], [], [])
        assert result == []

    def test_multiple_spaces(self):
        characters = list("A  B")
        starts = [0.0, 0.1, 0.2, 0.3]
        ends = [0.1, 0.2, 0.3, 0.4]
        result = _chars_to_word_timestamps(characters, starts, ends)
        assert len(result) == 2
        assert result[0]["word"] == "A"
        assert result[1]["word"] == "B"

    def test_single_word_no_space(self):
        characters = list("Hey")
        starts = [0.0, 0.1, 0.2]
        ends = [0.1, 0.2, 0.3]
        result = _chars_to_word_timestamps(characters, starts, ends)
        assert len(result) == 1
        assert result[0]["word"] == "Hey"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 0.3


@pytest.mark.unit
class TestElevenLabsTTSInit:
    def test_init_stores_api_key(self):
        tts = ElevenLabsTTS(api_key="test-key-123")
        assert tts.api_key == "test-key-123"


@pytest.mark.unit
class TestElevenLabsTTSSuccess:
    @pytest.fixture
    def tts(self):
        return ElevenLabsTTS(api_key="test-key")

    def _make_response(self, json_data: dict, status: int = 200):
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=json_data)
        if status < 400:
            mock_resp.raise_for_status = MagicMock()
        else:
            mock_resp.raise_for_status = MagicMock(
                side_effect=aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=status,
                ),
            )
        return mock_resp

    def _make_post_cm(self, mock_response):
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_response)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    @pytest.mark.asyncio
    async def test_generate_voiceover_success(self, tts):
        resp_data = _make_elevenlabs_response(
            characters=list("Test audio"),
            char_starts=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            char_ends=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        resp = self._make_response(resp_data)
        cm = self._make_post_cm(resp)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            session = MagicMock()
            session.post = MagicMock(return_value=cm)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = session

            result = await tts.generate_voiceover(
                text="Test audio", voice_id="voice123"
            )

        assert isinstance(result, TTSResult)
        assert result.audio_bytes is not None
        assert len(result.audio_bytes) > 0
        assert result.duration_seconds == 1.0
        assert len(result.vocal_alignment_data) == 2
        assert result.vocal_alignment_data[0]["word"] == "Test"
        assert result.vocal_alignment_data[1]["word"] == "audio"

    @pytest.mark.asyncio
    async def test_generate_voiceover_passes_correct_payload(self, tts):
        resp_data = _make_elevenlabs_response(
            characters=list("Hi"),
            char_starts=[0.0, 0.1],
            char_ends=[0.1, 0.2],
        )
        resp = self._make_response(resp_data)
        cm = self._make_post_cm(resp)

        with patch("aiohttp.ClientSession") as mock_session_cls, \
             patch("app.services.tts.settings") as mock_settings:
            mock_settings.tts_model_id = "eleven_flash_v2_5"
            session = MagicMock()
            session.post = MagicMock(return_value=cm)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = session

            await tts.generate_voiceover(text="Hello world", voice_id="voice456")

            call_kwargs = session.post.call_args[1]
            assert call_kwargs["json"]["text"] == "Hello world"
            assert call_kwargs["json"]["model_id"] == "eleven_flash_v2_5"
            assert call_kwargs["headers"]["xi-api-key"] == "test-key"
            assert call_kwargs["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_generate_voiceover_empty_alignment(self, tts):
        resp_data = _make_elevenlabs_response()
        resp = self._make_response(resp_data)
        cm = self._make_post_cm(resp)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            session = MagicMock()
            session.post = MagicMock(return_value=cm)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = session

            result = await tts.generate_voiceover(text="Test", voice_id="voice1")

        assert result.duration_seconds == 0.0
        assert result.vocal_alignment_data == []


@pytest.mark.unit
class TestElevenLabsTTSError:
    @pytest.fixture
    def tts(self):
        return ElevenLabsTTS(api_key="test-key")

    def _make_error_response(self, status: int):
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.text = AsyncMock(return_value='{"error":"bad request"}')
        mock_resp.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=status,
                message="bad request",
            ),
        )
        return mock_resp

    def _make_post_cm(self, mock_response):
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_response)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, tts):
        resp = self._make_error_response(401)
        cm = self._make_post_cm(resp)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            session = MagicMock()
            session.post = MagicMock(return_value=cm)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = session

            with pytest.raises(aiohttp.ClientResponseError):
                await tts.generate_voiceover(text="Test", voice_id="voice1")

    @pytest.mark.asyncio
    async def test_raises_on_missing_audio_base64(self, tts):
        resp_data = {
            "audio_base64": "",
            "alignment": {
                "characters": [],
                "character_start_times_seconds": [],
                "character_end_times_seconds": [],
            },
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=resp_data)
        mock_resp.raise_for_status = MagicMock()
        cm = self._make_post_cm(mock_resp)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            session = MagicMock()
            session.post = MagicMock(return_value=cm)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = session

            with pytest.raises(ValueError, match="missing audio_base64"):
                await tts.generate_voiceover(text="Test", voice_id="voice1")


@pytest.mark.unit
class TestTTSProviderRegistry:
    def test_registry_has_elevenlabs(self):
        assert "elevenlabs" in TTS_PROVIDERS
        assert TTS_PROVIDERS["elevenlabs"]["class"] == ElevenLabsTTS
        assert TTS_PROVIDERS["elevenlabs"]["api_key_attr"] == "tts_api_key"

    def test_get_provider_returns_instance(self):
        with patch("app.services.tts.settings") as mock_settings:
            mock_settings.tts_api_key = "test-key"
            provider = get_tts_provider("elevenlabs")
            assert isinstance(provider, TTSProvider)
            assert isinstance(provider, ElevenLabsTTS)

    def test_get_provider_caches_instance(self):
        with patch("app.services.tts.settings") as mock_settings:
            mock_settings.tts_api_key = "test-key"
            provider1 = get_tts_provider("elevenlabs")
            provider2 = get_tts_provider("elevenlabs")
            assert provider1 is provider2

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            get_tts_provider("unknown_provider")
