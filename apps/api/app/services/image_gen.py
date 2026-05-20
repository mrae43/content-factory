import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp

from app.core.config import settings

logger = logging.getLogger("factory.image_gen")

_STYLE_ENRICHMENT = (
    "Editorial infographic style, flat vector illustration, "
    "warm copper and stone tones, clean background, no text, high quality."
)

_TOGETHER_IMAGES_URL = "https://api.together.xyz/v1/images/generations"

PLATFORM_DIMENSIONS: dict[str, tuple[int, int]] = {
    "instagram": (1080, 1350),
    "linkedin": (1080, 1350),
    "twitter": (1080, 1620),
    "tiktok": (1080, 1920),
    "youtube": (1920, 1080),
}

DEFAULT_DIMENSIONS = (1080, 1350)


@dataclass
class ImageGenResult:
    success: bool
    image_bytes: Optional[bytes] = None
    width: Optional[int] = None
    height: Optional[int] = None
    failure_reason: Optional[str] = None
    prompt_used: Optional[str] = None


def _enrich_prompt(visual_description: str) -> str:
    return f"{visual_description}. {_STYLE_ENRICHMENT}"


def _get_dimensions(platform: str) -> tuple[int, int]:
    return PLATFORM_DIMENSIONS.get(platform, DEFAULT_DIMENSIONS)


class ImageGenerationService:
    def __init__(
        self,
        model: str = "",
        max_retries: int = 0,
        timeout_seconds: int = 0,
    ):
        self.model = model or settings.image_model
        self.max_retries = max_retries or settings.image_gen_max_retries
        self.timeout = timeout_seconds or settings.image_gen_timeout_seconds

    async def generate(
        self,
        visual_description: str,
        platform: str = "",
    ) -> ImageGenResult:
        prompt = _enrich_prompt(visual_description)
        width, height = _get_dimensions(platform)

        last_exception: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._call_api(prompt, width, height)
            except asyncio.TimeoutError:
                logger.warning(
                    "Image gen timeout (attempt %d/%d) for: %s",
                    attempt, self.max_retries, prompt[:60],
                )
                last_exception = asyncio.TimeoutError("HTTP timeout")
            except aiohttp.ClientResponseError as e:
                logger.warning(
                    "Image gen HTTP %d (attempt %d/%d) for: %s",
                    e.status, attempt, self.max_retries, prompt[:60],
                )
                last_exception = e
            except aiohttp.ClientError as e:
                logger.warning(
                    "Image gen connection error (attempt %d/%d): %s",
                    attempt, self.max_retries, e,
                )
                last_exception = e

            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)

        return ImageGenResult(
            success=False,
            failure_reason=f"All {self.max_retries} attempts failed: {last_exception}",
            prompt_used=prompt,
        )

    async def _call_api(
        self,
        prompt: str,
        width: int,
        height: int,
    ) -> ImageGenResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": 4,
            "n": 1,
            "response_format": "b64_json",
        }

        headers = {
            "Authorization": f"Bearer {settings.together_api_key}",
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                _TOGETHER_IMAGES_URL,
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        result = self._validate_response(data, width, height, prompt)
        return result

    def _validate_response(
        self,
        data: dict,
        expected_width: int,
        expected_height: int,
        prompt: str,
    ) -> ImageGenResult:
        result_data = data.get("data")
        if not result_data or not isinstance(result_data, list) or len(result_data) == 0:
            return ImageGenResult(
                success=False,
                failure_reason="Empty or missing data array in API response",
                prompt_used=prompt,
            )

        first = result_data[0]
        b64_json = first.get("b64_json")
        if not b64_json:
            return ImageGenResult(
                success=False,
                failure_reason="No b64_json field in API response",
                prompt_used=prompt,
            )

        import base64
        image_bytes = base64.b64decode(b64_json)

        if len(image_bytes) < 1024:
            return ImageGenResult(
                success=False,
                failure_reason=f"Image too small: {len(image_bytes)} bytes (< 1 KB)",
                prompt_used=prompt,
            )

        returned_width = first.get("width", 0)
        returned_height = first.get("height", 0)

        # Validate if the API returns dimensions; if not (some models don't),
        # we assume correctness based on the request params
        if returned_width and returned_height:
            if returned_width != expected_width or returned_height != expected_height:
                return ImageGenResult(
                    success=False,
                    failure_reason=(
                        f"Dimension mismatch: got {returned_width}x{returned_height}, "
                        f"expected {expected_width}x{expected_height}"
                    ),
                    prompt_used=prompt,
                )

        return ImageGenResult(
            success=True,
            image_bytes=image_bytes,
            width=expected_width,
            height=expected_height,
            prompt_used=prompt,
        )
