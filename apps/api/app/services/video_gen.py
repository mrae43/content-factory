from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

from app.core.config import settings
from app.services.tools import Tool

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
        response = await self.client.videos.create(
            model=model,
            prompt=prompt,
            n=1,
            duration=duration,
        )
        return response.id

    async def poll_video(self, job_id: str) -> VideoGenResult:
        video = await self.client.videos.retrieve(id=job_id)
        download_url = getattr(video, "output_url", None)
        if download_url is None:
            output = getattr(video, "output", None)
            if output is not None:
                download_url = getattr(output, "url", None)
        return VideoGenResult(
            status=video.status,
            download_url=download_url,
            failure_reason=getattr(video, "error", None),
        )


VIDEO_GEN_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "together": {
        "class": TogetherVideoGen,
        "api_key_attr": "together_api_key",
    },
}

_DEFAULT_VIDEO_PROVIDER = "together"


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
        api_key = getattr(settings, config["api_key_attr"], None)
        _video_gen_provider_cache[key] = config["class"](api_key=api_key)
    return _video_gen_provider_cache[key]


def make_generate_video_tool() -> Tool:
    async def _generate(
        prompt: str,
        model: str = "",
        duration: int = 30,
    ) -> dict:
        provider = get_video_gen_provider(settings.video_gen_provider)
        resolved_model = model or settings.video_gen_model
        job_id = await provider.generate_video(
            prompt=prompt, model=resolved_model, duration=duration
        )
        return {"job_id": job_id}

    return Tool(
        name="generate_video",
        description="Submit a video generation job via the configured video provider.",
        callable=_generate,
        permissions={"VideoGeneratorAgent", "*"},
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
        permissions={"VideoGeneratorAgent", "*"},
    )
