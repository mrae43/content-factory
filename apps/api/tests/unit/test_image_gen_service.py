import base64
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from app.services.image_gen import (
    ImageGenerationService,
    ImageGenResult,
    _enrich_prompt,
    _get_dimensions,
    _STYLE_ENRICHMENT,
    DEFAULT_DIMENSIONS,
    PLATFORM_DIMENSIONS,
)


def _dummy_b64(size: int = 2048) -> str:
    return base64.b64encode(b"x" * size).decode("ascii")


class _MockSession:
    """Minimal session mock that supports async with and has a callable .post.

    Avoids AsyncMock attribute auto-creation issues (where session.post
    becomes an AsyncMock whose calls return coroutines instead of the
    intended async-context-manager).
    """

    def __init__(self, post_return):
        self.post = MagicMock(return_value=post_return)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.unit
class TestImageGenHelpers:
    def test_enrich_prompt_appends_style(self):
        result = _enrich_prompt("A graph showing GDP growth")
        assert result.startswith("A graph showing GDP growth. ")
        assert _STYLE_ENRICHMENT in result

    def test_get_dimensions_known_platform(self):
        for platform, (w, h) in PLATFORM_DIMENSIONS.items():
            assert _get_dimensions(platform) == (w, h)

    def test_get_dimensions_unknown_platform_returns_default(self):
        assert _get_dimensions("unknown") == DEFAULT_DIMENSIONS

    def test_get_dimensions_empty_string_returns_default(self):
        assert _get_dimensions("") == DEFAULT_DIMENSIONS


@pytest.mark.unit
class TestImageGenResult:
    def test_success_result(self):
        result = ImageGenResult(
            success=True,
            image_bytes=b"data",
            width=1080,
            height=1350,
            prompt_used="test prompt",
        )
        assert result.success is True
        assert result.image_bytes == b"data"
        assert result.failure_reason is None

    def test_failure_result(self):
        result = ImageGenResult(
            success=False,
            failure_reason="API error",
            prompt_used="test prompt",
        )
        assert result.success is False
        assert result.image_bytes is None
        assert result.failure_reason == "API error"


@pytest.mark.unit
class TestImageGenerationServiceInit:
    def test_defaults_from_settings(self):
        with patch("app.services.image_gen.settings") as mock_settings:
            mock_settings.image_model = "default-model"
            mock_settings.image_gen_max_retries = 3
            mock_settings.image_gen_timeout_seconds = 30
            service = ImageGenerationService()
            assert service.model == "default-model"
            assert service.max_retries == 3
            assert service.timeout == 30

    def test_explicit_overrides(self):
        with patch("app.services.image_gen.settings"):
            service = ImageGenerationService(
                model="custom-model",
                max_retries=5,
                timeout_seconds=60,
            )
            assert service.model == "custom-model"
            assert service.max_retries == 5
            assert service.timeout == 60


@pytest.mark.unit
class TestImageGenerationService:
    @pytest.fixture
    def service(self):
        return ImageGenerationService(
            model="test-model",
            max_retries=2,
            timeout_seconds=30,
        )

    @pytest.fixture
    def mock_aiohttp(self):
        with patch("aiohttp.ClientSession") as mock_cls:
            yield mock_cls

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

    def _make_session(self, post_cm):
        return _MockSession(post_cm)

    def _make_success_response(self, b64_str=None, width=1080, height=1350):
        data = {
            "data": [
                {"b64_json": b64_str or _dummy_b64(), "width": width, "height": height}
            ]
        }
        return self._make_response(data)

    # --- Success ---

    @pytest.mark.asyncio
    async def test_generate_success(self, service, mock_aiohttp):
        resp = self._make_success_response()
        cm = self._make_post_cm(resp)
        session = self._make_session(cm)
        mock_aiohttp.return_value = session

        result = await service.generate(
            "A chart showing GDP growth", platform="instagram"
        )

        assert result.success is True
        assert result.image_bytes is not None
        assert len(result.image_bytes) >= 1024
        assert result.width == 1080
        assert result.height == 1350
        assert result.prompt_used is not None
        assert _STYLE_ENRICHMENT in result.prompt_used

    @pytest.mark.asyncio
    async def test_generate_lenient_when_no_dimensions_returned(
        self, service, mock_aiohttp
    ):
        resp = self._make_response(
            {"data": [{"b64_json": _dummy_b64()}]},
        )
        cm = self._make_post_cm(resp)
        session = self._make_session(cm)
        mock_aiohttp.return_value = session

        result = await service.generate("A chart", platform="instagram")

        assert result.success is True
        assert result.width == 1080
        assert result.height == 1350

    @pytest.mark.asyncio
    async def test_generate_passes_correct_payload(self, service, mock_aiohttp):
        resp = self._make_success_response()
        cm = self._make_post_cm(resp)
        session = self._make_session(cm)
        mock_aiohttp.return_value = session

        await service.generate("Test visual", platform="linkedin")

        mock_aiohttp.assert_called_once()
        call_kwargs = session.post.call_args[1]
        assert call_kwargs["json"]["model"] == "test-model"
        assert call_kwargs["json"]["width"] == 1080
        assert call_kwargs["json"]["height"] == 1350
        assert call_kwargs["json"]["steps"] == 4
        assert call_kwargs["json"]["n"] == 1
        assert call_kwargs["json"]["response_format"] == "b64_json"
        assert "Bearer" in call_kwargs["headers"]["Authorization"]
        assert call_kwargs["headers"]["Content-Type"] == "application/json"

    # --- Retry / error recovery ---

    @pytest.mark.asyncio
    async def test_generate_retries_on_timeout_then_succeeds(
        self, service, mock_aiohttp
    ):
        fail_cm = AsyncMock()
        fail_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        fail_cm.__aexit__ = AsyncMock(return_value=None)

        resp = self._make_success_response()
        success_cm = self._make_post_cm(resp)

        session_fail = self._make_session(fail_cm)
        session_ok = self._make_session(success_cm)
        mock_aiohttp.side_effect = [session_fail, session_ok]

        result = await service.generate("Test", platform="instagram")

        assert result.success is True
        assert mock_aiohttp.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_retries_on_http_error_then_succeeds(
        self, service, mock_aiohttp
    ):
        fail_resp = self._make_response({"error": "server error"}, status=500)
        fail_cm = self._make_post_cm(fail_resp)

        resp = self._make_success_response()
        success_cm = self._make_post_cm(resp)

        session_fail = self._make_session(fail_cm)
        session_ok = self._make_session(success_cm)
        mock_aiohttp.side_effect = [session_fail, session_ok]

        result = await service.generate("Test", platform="instagram")

        assert result.success is True
        assert mock_aiohttp.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_retries_on_connection_error_then_succeeds(
        self, service, mock_aiohttp
    ):
        fail_cm = AsyncMock()
        fail_cm.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Connection refused"),
        )
        fail_cm.__aexit__ = AsyncMock(return_value=None)

        resp = self._make_success_response()
        success_cm = self._make_post_cm(resp)

        session_fail = self._make_session(fail_cm)
        session_ok = self._make_session(success_cm)
        mock_aiohttp.side_effect = [session_fail, session_ok]

        result = await service.generate("Test", platform="instagram")

        assert result.success is True
        assert mock_aiohttp.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_all_retries_exhausted(self, service, mock_aiohttp):
        fail_cm = AsyncMock()
        fail_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        fail_cm.__aexit__ = AsyncMock(return_value=None)

        session = self._make_session(fail_cm)
        mock_aiohttp.return_value = session

        result = await service.generate("Test", platform="instagram")

        assert result.success is False
        assert result.failure_reason is not None
        assert "All 2" in result.failure_reason or "2 attempts" in result.failure_reason
        assert result.image_bytes is None
        assert mock_aiohttp.call_count == 2

    # --- Response validation failures ---

    @pytest.mark.asyncio
    async def test_generate_empty_data_array(self, service, mock_aiohttp):
        resp = self._make_response({"data": []})
        cm = self._make_post_cm(resp)
        session = self._make_session(cm)
        mock_aiohttp.return_value = session

        result = await service.generate("Test", platform="instagram")

        assert result.success is False
        assert "empty" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_generate_missing_b64_json(self, service, mock_aiohttp):
        resp = self._make_response({"data": [{"width": 1080}]})
        cm = self._make_post_cm(resp)
        session = self._make_session(cm)
        mock_aiohttp.return_value = session

        result = await service.generate("Test", platform="instagram")

        assert result.success is False
        assert "b64_json" in result.failure_reason

    @pytest.mark.asyncio
    async def test_generate_image_too_small(self, service, mock_aiohttp):
        tiny_b64 = _dummy_b64(512)
        resp = self._make_response(
            {"data": [{"b64_json": tiny_b64, "width": 1080, "height": 1350}]},
        )
        cm = self._make_post_cm(resp)
        session = self._make_session(cm)
        mock_aiohttp.return_value = session

        result = await service.generate("Test", platform="instagram")

        assert result.success is False
        assert (
            "1024" in result.failure_reason or "small" in result.failure_reason.lower()
        )

    @pytest.mark.asyncio
    async def test_generate_dimension_mismatch(self, service, mock_aiohttp):
        resp = self._make_success_response(width=1920, height=1080)
        cm = self._make_post_cm(resp)
        session = self._make_session(cm)
        mock_aiohttp.return_value = session

        result = await service.generate("Test", platform="instagram")

        assert result.success is False
        assert "dimension" in result.failure_reason.lower()

    # --- Direct _validate_response tests ---

    def test_validate_response_success(self, service):
        data = {"data": [{"b64_json": _dummy_b64(), "width": 1080, "height": 1350}]}
        result = service._validate_response(data, 1080, 1350, "prompt")
        assert result.success is True
        assert result.width == 1080
        assert result.height == 1350

    def test_validate_response_empty_data(self, service):
        result = service._validate_response({}, 1080, 1350, "prompt")
        assert result.success is False
        assert "empty" in result.failure_reason.lower()

    def test_validate_response_data_not_a_list(self, service):
        result = service._validate_response(
            {"data": "not_a_list"}, 1080, 1350, "prompt"
        )
        assert result.success is False

    def test_validate_response_data_empty_list(self, service):
        result = service._validate_response({"data": []}, 1080, 1350, "prompt")
        assert result.success is False

    def test_validate_response_missing_b64_json(self, service):
        data = {"data": [{"width": 1080, "height": 1350}]}
        result = service._validate_response(data, 1080, 1350, "prompt")
        assert result.success is False
        assert "b64_json" in result.failure_reason

    def test_validate_response_dimension_mismatch(self, service):
        data = {"data": [{"b64_json": _dummy_b64(), "width": 1920, "height": 1080}]}
        result = service._validate_response(data, 1080, 1350, "prompt")
        assert result.success is False
        assert "dimension" in result.failure_reason.lower()

    def test_validate_response_lenient_when_no_dims_returned(self, service):
        data = {"data": [{"b64_json": _dummy_b64()}]}
        result = service._validate_response(data, 1080, 1350, "prompt")
        assert result.success is True
