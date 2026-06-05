from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import aiohttp
import jwt
from pydantic import BaseModel

from app.core.config import settings
from app.services.tools import Tool

MINIMAX_6S_MODELS = frozenset(
    {
        "minimax/video-01-director",
    }
)

logger = logging.getLogger("factory.video_gen")


class VideoGenResult(BaseModel):
    status: str
    download_url: Optional[str] = None
    failure_reason: Optional[str] = None


class VideoGenProvider(ABC):
    @abstractmethod
    async def generate_video(self, prompt: str, model: str, **kwargs) -> str: ...

    @abstractmethod
    async def poll_video(self, job_id: str) -> VideoGenResult: ...


class TogetherVideoGen(VideoGenProvider):
    def __init__(self, api_key: str):
        from together import AsyncTogether

        self.client = AsyncTogether(api_key=api_key)

    async def generate_video(
        self,
        prompt: str,
        model: str = "",
        duration: int = 30,
        **kwargs,
    ) -> str:
        if not model:
            raise ValueError(
                "model is required for video generation. "
                "Set VIDEO_GEN_MODEL in .env or pass a model explicitly."
            )
        model_key = model.lower().strip()
        if model_key in MINIMAX_6S_MODELS and duration != 6:
            logger.info(
                "Model %s requires exactly 6 seconds; clamping duration from %d to 6",
                model,
                duration,
            )
            duration = 6
        response = await self.client.videos.create(
            model=model,
            prompt=prompt,
            seconds=str(duration),
        )
        return response.id

    async def poll_video(self, job_id: str) -> VideoGenResult:
        video = await self.client.videos.retrieve(id=job_id)
        download_url = video.outputs.video_url if video.outputs else None
        return VideoGenResult(
            status=video.status,
            download_url=download_url,
            failure_reason=video.error.message if video.error else None,
        )


def _generate_kling_jwt(access_key: str, secret_key: str) -> str:
    headers = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {
        "iss": access_key,
        "exp": now + 1800,
        "nbf": now - 5,
    }
    return jwt.encode(payload, secret_key, headers=headers)


def _map_duration_to_kling(target_duration: float) -> str:
    """Map 3-15s scene duration to Kling-supported 5s or 10s."""
    if target_duration <= 7.0:
        return "5"
    return "10"


class KlingVideoGen(VideoGenProvider):
    """
    Kling AI text-to-video provider.
    Uses JWT authentication with a 30-minute token expiry.
    """

    def __init__(self, access_key: str, secret_key: str, base_url: str = ""):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url or "https://api-singapore.klingai.com/v1"
        self._jwt_token: str = ""
        self._jwt_expires_at: float = 0.0

    def _auth_headers(self) -> dict:
        if time.time() >= self._jwt_expires_at - 300:  # refresh 5 min before expiry
            self._jwt_token = _generate_kling_jwt(self.access_key, self.secret_key)
            self._jwt_expires_at = time.time() + 1800
        return {
            "Authorization": f"Bearer {self._jwt_token}",
            "Content-Type": "application/json",
        }

    async def generate_video(
        self,
        prompt: str,
        model: str = "",
        duration: int = 5,
        aspect_ratio: str = "9:16",
        sound: str = "off",
        **kwargs,
    ) -> str:
        mapped_duration = _map_duration_to_kling(duration)
        payload = {
            "model_name": model or settings.video_gen_model,
            "mode": settings.video_gen_mode,
            "prompt": prompt,
            "duration": mapped_duration,
            "aspect_ratio": aspect_ratio,
            "sound": sound,
        }
        headers = self._auth_headers()
        url = f"{self.base_url}/videos/text2video"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                task_id = data.get("data", {}).get("task_id")
                if not task_id:
                    raise ValueError(
                        f"Kling API did not return a task_id. Response: {data}"
                    )
                return task_id

    async def poll_video(self, job_id: str) -> VideoGenResult:
        headers = self._auth_headers()
        url = f"{self.base_url}/videos/text2video/{job_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                task_data = data.get("data", {})
                status = task_data.get("status", "unknown")
                download_url = None
                if status == "succeed":
                    videos = task_data.get("task_result", {}).get("videos", [])
                    if videos:
                        download_url = videos[0].get("url")
                    status = "completed"
                failure_reason = None
                if status == "failed":
                    failure_reason = (
                        task_data.get("task_status_reason")
                        or task_data.get("message")
                        or "Unknown failure"
                    )
                return VideoGenResult(
                    status=status,
                    download_url=download_url,
                    failure_reason=failure_reason,
                )


VIDEO_GEN_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "together": {
        "class": TogetherVideoGen,
        "api_key_attr": "together_api_key",
    },
    "kling": {
        "class": KlingVideoGen,
        "api_key_attr": None,  # handled via kling_access_key + kling_secret_key
    },
}

_DEFAULT_VIDEO_PROVIDER = "kling"


def _resolve_video_provider(provider_name: str = "") -> tuple[str, dict]:
    if not provider_name:
        provider_name = _DEFAULT_VIDEO_PROVIDER
    config = VIDEO_GEN_PROVIDERS.get(provider_name)
    if config is None:
        raise ValueError(
            f"Unknown video provider '{provider_name}'. "
            f"Available: {', '.join(VIDEO_GEN_PROVIDERS)}."
        )
    return provider_name, config


_video_gen_provider_cache: Dict[str, VideoGenProvider] = {}


def get_video_gen_provider(provider_name: str = "") -> VideoGenProvider:
    key = provider_name or _DEFAULT_VIDEO_PROVIDER
    if key not in _video_gen_provider_cache:
        _, config = _resolve_video_provider(key)
        api_key_attr = config.get("api_key_attr")
        if api_key_attr:
            api_key = getattr(settings, api_key_attr, None)
            _video_gen_provider_cache[key] = config["class"](api_key=api_key)
        else:
            _video_gen_provider_cache[key] = config["class"](
                access_key=settings.kling_access_key,
                secret_key=settings.kling_secret_key,
                base_url=settings.kling_base_url,
            )
    return _video_gen_provider_cache[key]


def make_generate_video_tool() -> Tool:
    async def _generate(
        prompt: str,
        model: str = "",
        duration: int = 30,
    ) -> dict:
        provider = get_video_gen_provider(settings.video_gen_provider)
        resolved_model = model or settings.video_gen_model
        if not resolved_model:
            raise ValueError(
                "video_gen_model is not configured. Set VIDEO_GEN_MODEL "
                "in .env or provide a model explicitly."
            )
        job_id = await provider.generate_video(
            prompt=prompt, model=resolved_model, duration=duration
        )
        return {"job_id": job_id}

    return Tool(
        name="generate_video",
        description="Submit a video generation job via the configured video provider.",
        callable=_generate,
        permissions={"VideoGeneratorAgent", "ShortVisualAssetAgent", "*"},
    )


def make_poll_video_tool() -> Tool:
    async def _poll(job_id: str) -> dict:
        provider = get_video_gen_provider(settings.video_gen_provider)
        result = await provider.poll_video(job_id)
        return result.model_dump()

    return Tool(
        name="poll_video",
        description="Poll a video generation job for completion status.",
        callable=_poll,
        permissions={"VideoGeneratorAgent", "ShortVisualAssetAgent", "*"},
    )
