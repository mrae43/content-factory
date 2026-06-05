import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest

from app.services.video_gen import (
    KlingVideoGen,
    TogetherVideoGen,
    VideoGenProvider,
    VideoGenResult,
    _generate_kling_jwt,
    _map_duration_to_kling,
    get_video_gen_provider,
    make_generate_video_tool,
    make_poll_video_tool,
    VIDEO_GEN_PROVIDERS,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_aiohttp_resp(status: int = 200, json_data: dict | None = None):
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data or {})
    if status >= 400:
        from aiohttp import ClientResponseError

        mock_resp.raise_for_status = MagicMock(
            side_effect=ClientResponseError(
                request_info=MagicMock(), history=(), status=status
            )
        )
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _make_post_cm(mock_response):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _make_get_cm(mock_response):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


# ── Test data ───────────────────────────────────────────────────────────────────

TEST_ACCESS_KEY = "test-access-key-12345"
TEST_SECRET_KEY = "test-secret-key-67890"
TEST_BASE_URL = "https://api.klingai.com/v1"


# ═══════════════════════════════════════════════════════════════════════════════
# _generate_kling_jwt
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestKlingJWTGeneration:
    def test_returns_valid_jwt_with_correct_claims(self):
        token = _generate_kling_jwt(TEST_ACCESS_KEY, TEST_SECRET_KEY)
        assert isinstance(token, str)
        assert token.count(".") == 2  # three parts

        decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=["HS256"])
        assert decoded["iss"] == TEST_ACCESS_KEY
        assert decoded["nbf"] <= int(time.time())
        assert decoded["exp"] > int(time.time())

    def test_jwt_expiry_is_30_minutes(self):
        token = _generate_kling_jwt(TEST_ACCESS_KEY, TEST_SECRET_KEY)
        decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=["HS256"])
        assert decoded["exp"] - decoded["nbf"] == 1805  # 1800 + 5s nbf buffer

    def test_different_keys_produce_different_tokens(self):
        t1 = _generate_kling_jwt("key1", "secret1")
        t2 = _generate_kling_jwt("key2", "secret2")
        assert t1 != t2

    def test_wrong_secret_fails_verification(self):
        token = _generate_kling_jwt(TEST_ACCESS_KEY, TEST_SECRET_KEY)
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(token, "wrong-secret", algorithms=["HS256"])


# ═══════════════════════════════════════════════════════════════════════════════
# _map_duration_to_kling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestKlingDurationMapping:
    def test_three_seconds_maps_to_5(self):
        assert _map_duration_to_kling(3.0) == "5"

    def test_seven_seconds_maps_to_5(self):
        assert _map_duration_to_kling(7.0) == "5"

    def test_seven_point_one_maps_to_10(self):
        assert _map_duration_to_kling(7.1) == "10"

    def test_eight_seconds_maps_to_10(self):
        assert _map_duration_to_kling(8.0) == "10"

    def test_fifteen_seconds_maps_to_10(self):
        assert _map_duration_to_kling(15.0) == "10"

    def test_zero_seconds_maps_to_5(self):
        assert _map_duration_to_kling(0.0) == "5"

    def test_negative_seconds_maps_to_5(self):
        assert _map_duration_to_kling(-1.0) == "5"


# ═══════════════════════════════════════════════════════════════════════════════
# KlingVideoGen._auth_headers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestKlingVideoGenAuthHeaders:
    def test_returns_bearer_token(self):
        provider = KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)
        headers = provider._auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["Content-Type"] == "application/json"

    def test_caches_jwt_token(self):
        provider = KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)
        h1 = provider._auth_headers()
        h2 = provider._auth_headers()
        assert h1["Authorization"] == h2["Authorization"]

    def test_refreshes_jwt_when_expired(self):
        provider = KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)
        # Set expiry in the past
        provider._jwt_expires_at = time.time() - 1
        provider._jwt_token = "old-token"
        headers = provider._auth_headers()
        assert headers["Authorization"] != "Bearer old-token"

    def test_refreshes_jwt_within_five_minute_buffer(self):
        provider = KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)
        # Set expiry 4 minutes from now (within 5 min buffer → should refresh)
        provider._jwt_expires_at = time.time() + 240
        provider._jwt_token = "stale-token"
        headers = provider._auth_headers()
        assert headers["Authorization"] != "Bearer stale-token"

    def test_does_not_refresh_when_well_within_expiry(self):
        provider = KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)
        provider._jwt_expires_at = time.time() + 1800
        provider._jwt_token = "fresh-token"
        headers = provider._auth_headers()
        assert headers["Authorization"] == "Bearer fresh-token"


# ═══════════════════════════════════════════════════════════════════════════════
# KlingVideoGen.generate_video
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestKlingVideoGenGenerateVideo:
    @pytest.fixture
    def provider(self):
        return KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)

    @pytest.fixture
    def mock_session(self):
        session = MagicMock()
        session.post = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        return session

    @pytest.mark.asyncio
    async def test_generate_video_returns_task_id(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {"task_id": "kling-task-001"}})
        mock_session.post.return_value = _make_post_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                task_id = await provider.generate_video("A test prompt", duration=5)

        assert task_id == "kling-task-001"

    @pytest.mark.asyncio
    async def test_generate_sends_correct_payload(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {"task_id": "t-001"}})
        mock_session.post.return_value = _make_post_cm(resp)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ),
            patch("app.services.video_gen.settings") as mock_settings,
        ):
            mock_settings.video_gen_model = "kling-v1-6"
            mock_settings.video_gen_mode = "std"
            await provider.generate_video("Prompt", duration=5)

        call_json = mock_session.post.call_args[1]["json"]
        assert call_json["model_name"] == "kling-v1-6"
        assert call_json["mode"] == "std"
        assert call_json["prompt"] == "Prompt"
        assert call_json["duration"] == "5"
        assert call_json["aspect_ratio"] == "9:16"
        assert call_json["sound"] == "off"

    @pytest.mark.asyncio
    async def test_generate_maps_duration_correctly(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {"task_id": "t-002"}})
        mock_session.post.return_value = _make_post_cm(resp)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ),
            patch("app.services.video_gen.settings") as mock_settings,
        ):
            mock_settings.video_gen_model = "kling-v1-6"
            mock_settings.video_gen_mode = "std"
            await provider.generate_video("Prompt", duration=10)

        call_json = mock_session.post.call_args[1]["json"]
        assert call_json["duration"] == "10"

    @pytest.mark.asyncio
    async def test_generate_raises_when_no_task_id(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {}})
        mock_session.post.return_value = _make_post_cm(resp)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ),
            patch("app.services.video_gen.settings") as mock_settings,
        ):
            mock_settings.video_gen_model = "kling-v1-6"
            mock_settings.video_gen_mode = "std"
            with pytest.raises(ValueError, match="did not return a task_id"):
                await provider.generate_video("Prompt", duration=5)

    @pytest.mark.asyncio
    async def test_generate_raises_on_http_error(self, provider, mock_session):
        resp = _make_aiohttp_resp(401, {"error": "unauthorized"})
        mock_session.post.return_value = _make_post_cm(resp)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ),
            patch("app.services.video_gen.settings") as mock_settings,
        ):
            mock_settings.video_gen_model = "kling-v1-6"
            mock_settings.video_gen_mode = "std"
            with pytest.raises(Exception):
                await provider.generate_video("Prompt", duration=5)

    @pytest.mark.asyncio
    async def test_generate_posts_to_correct_url(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {"task_id": "t-003"}})
        mock_session.post.return_value = _make_post_cm(resp)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ),
            patch("app.services.video_gen.settings") as mock_settings,
        ):
            mock_settings.video_gen_model = "kling-v1-6"
            mock_settings.video_gen_mode = "std"
            await provider.generate_video("Prompt", duration=5)

        call_url = mock_session.post.call_args[0][0]
        assert call_url == f"{TEST_BASE_URL}/videos/text2video"

    @pytest.mark.asyncio
    async def test_generate_uses_custom_base_url(self):
        custom_provider = KlingVideoGen(
            TEST_ACCESS_KEY, TEST_SECRET_KEY, "https://custom.api.com/v1"
        )

        session = MagicMock()
        session.post = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        resp = _make_aiohttp_resp(200, {"data": {"task_id": "t-004"}})
        session.post.return_value = _make_post_cm(resp)

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch.object(
                custom_provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ),
            patch("app.services.video_gen.settings") as mock_settings,
        ):
            mock_settings.video_gen_model = "kling-v1-6"
            mock_settings.video_gen_mode = "std"
            await custom_provider.generate_video("Prompt", duration=5)

        call_url = session.post.call_args[0][0]
        assert call_url == "https://custom.api.com/v1/videos/text2video"


# ═══════════════════════════════════════════════════════════════════════════════
# KlingVideoGen.poll_video
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestKlingVideoGenPollVideo:
    @pytest.fixture
    def provider(self):
        return KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)

    @pytest.fixture
    def mock_session(self):
        session = MagicMock()
        session.get = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        return session

    @pytest.mark.asyncio
    async def test_poll_succeed_returns_completed_with_url(
        self, provider, mock_session
    ):
        resp = _make_aiohttp_resp(
            200,
            {
                "data": {
                    "status": "succeed",
                    "task_result": {"videos": [{"url": "https://kling.com/video.mp4"}]},
                }
            },
        )
        mock_session.get.return_value = _make_get_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                result = await provider.poll_video("task-001")

        assert isinstance(result, VideoGenResult)
        assert result.status == "completed"
        assert result.download_url == "https://kling.com/video.mp4"
        assert result.failure_reason is None

    @pytest.mark.asyncio
    async def test_poll_failed_returns_failure_reason(self, provider, mock_session):
        resp = _make_aiohttp_resp(
            200,
            {
                "data": {
                    "status": "failed",
                    "task_status_reason": "Credit exhausted",
                }
            },
        )
        mock_session.get.return_value = _make_get_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                result = await provider.poll_video("task-002")

        assert result.status == "failed"
        assert result.download_url is None
        assert result.failure_reason == "Credit exhausted"

    @pytest.mark.asyncio
    async def test_poll_submitted_passes_through(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {"status": "submitted"}})
        mock_session.get.return_value = _make_get_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                result = await provider.poll_video("task-003")

        assert result.status == "submitted"
        assert result.download_url is None

    @pytest.mark.asyncio
    async def test_poll_processing_passes_through(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {"status": "processing"}})
        mock_session.get.return_value = _make_get_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                result = await provider.poll_video("task-004")

        assert result.status == "processing"

    @pytest.mark.asyncio
    async def test_poll_succeed_no_videos_returns_no_url(self, provider, mock_session):
        resp = _make_aiohttp_resp(
            200,
            {
                "data": {
                    "status": "succeed",
                    "task_result": {"videos": []},
                }
            },
        )
        mock_session.get.return_value = _make_get_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                result = await provider.poll_video("task-005")

        assert result.status == "completed"
        assert result.download_url is None

    @pytest.mark.asyncio
    async def test_poll_failed_without_reason_falls_back(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {"status": "failed"}})
        mock_session.get.return_value = _make_get_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                result = await provider.poll_video("task-006")

        assert result.status == "failed"
        assert result.failure_reason == "Unknown failure"

    @pytest.mark.asyncio
    async def test_poll_raises_on_http_error(self, provider, mock_session):
        resp = _make_aiohttp_resp(500, {})
        mock_session.get.return_value = _make_get_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                with pytest.raises(Exception):
                    await provider.poll_video("task-007")

    @pytest.mark.asyncio
    async def test_poll_hits_correct_url(self, provider, mock_session):
        resp = _make_aiohttp_resp(200, {"data": {"status": "submitted"}})
        mock_session.get.return_value = _make_get_cm(resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                provider,
                "_auth_headers",
                return_value={
                    "Authorization": "Bearer x",
                    "Content-Type": "application/json",
                },
            ):
                await provider.poll_video("the-job-id")

        call_url = mock_session.get.call_args[0][0]
        assert call_url == f"{TEST_BASE_URL}/videos/text2video/the-job-id"


# ═══════════════════════════════════════════════════════════════════════════════
# KlingVideoGen constructor
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestKlingVideoGenInit:
    def test_stores_credentials(self):
        provider = KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)
        assert provider.access_key == TEST_ACCESS_KEY
        assert provider.secret_key == TEST_SECRET_KEY
        assert provider.base_url == TEST_BASE_URL
        assert provider._jwt_token == ""
        assert provider._jwt_expires_at == 0.0

    def test_default_base_url(self):
        provider = KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY)
        assert provider.base_url == "https://api-singapore.klingai.com/v1"

    def test_custom_base_url(self):
        provider = KlingVideoGen(
            TEST_ACCESS_KEY, TEST_SECRET_KEY, "https://custom.url/v1"
        )
        assert provider.base_url == "https://custom.url/v1"

    def test_is_video_gen_provider(self):
        provider = KlingVideoGen(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_BASE_URL)
        assert isinstance(provider, VideoGenProvider)


# ═══════════════════════════════════════════════════════════════════════════════
# VideoGenProviderRegistry
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestVideoGenProviderRegistry:
    def test_registry_has_together_and_kling(self):
        assert "together" in VIDEO_GEN_PROVIDERS
        assert "kling" in VIDEO_GEN_PROVIDERS

    def test_together_config(self):
        config = VIDEO_GEN_PROVIDERS["together"]
        assert config["class"] == TogetherVideoGen
        assert config["api_key_attr"] == "together_api_key"

    def test_kling_config(self):
        config = VIDEO_GEN_PROVIDERS["kling"]
        assert config["class"] == KlingVideoGen
        assert config["api_key_attr"] is None

    def test_get_kling_provider_returns_instance(self):
        with (
            patch("app.services.video_gen.settings") as mock_settings,
            patch("app.services.video_gen._video_gen_provider_cache", {}),
        ):
            mock_settings.kling_access_key = TEST_ACCESS_KEY
            mock_settings.kling_secret_key = TEST_SECRET_KEY
            mock_settings.kling_base_url = TEST_BASE_URL
            provider = get_video_gen_provider("kling")
            assert isinstance(provider, KlingVideoGen)
            assert provider.access_key == TEST_ACCESS_KEY
            assert provider.secret_key == TEST_SECRET_KEY

    def test_get_together_provider_returns_instance(self):
        with (
            patch("app.services.video_gen.settings") as mock_settings,
            patch("app.services.video_gen._video_gen_provider_cache", {}),
        ):
            mock_settings.together_api_key = "together-key"
            provider = get_video_gen_provider("together")
            assert isinstance(provider, TogetherVideoGen)

    def test_get_provider_caches_instance(self):
        with (
            patch("app.services.video_gen.settings") as mock_settings,
            patch("app.services.video_gen._video_gen_provider_cache", {}),
        ):
            mock_settings.kling_access_key = TEST_ACCESS_KEY
            mock_settings.kling_secret_key = TEST_SECRET_KEY
            mock_settings.kling_base_url = TEST_BASE_URL
            p1 = get_video_gen_provider("kling")
            p2 = get_video_gen_provider("kling")
            assert p1 is p2

    def test_default_provider_returns_kling(self):
        with (
            patch("app.services.video_gen.settings") as mock_settings,
            patch("app.services.video_gen._video_gen_provider_cache", {}),
        ):
            mock_settings.kling_access_key = TEST_ACCESS_KEY
            mock_settings.kling_secret_key = TEST_SECRET_KEY
            mock_settings.kling_base_url = TEST_BASE_URL
            provider = get_video_gen_provider()
            assert isinstance(provider, KlingVideoGen)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown video provider"):
            get_video_gen_provider("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════════
# make_generate_video_tool / make_poll_video_tool
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestVideoGenToolFactories:
    def test_make_generate_video_tool_returns_tool(self):
        with patch("app.services.video_gen.get_video_gen_provider"):
            tool = make_generate_video_tool()
            assert tool.name == "generate_video"
            assert "video generation" in tool.description.lower()
            assert "*" in tool.permissions

    def test_make_poll_video_tool_returns_tool(self):
        with patch("app.services.video_gen.get_video_gen_provider"):
            tool = make_poll_video_tool()
            assert tool.name == "poll_video"
            assert "poll" in tool.description.lower()
            assert "*" in tool.permissions
