import asyncio
import json
import logging
import random
import time as time_module
from dataclasses import dataclass
from typing import Optional

from app.services.tools import Tool

import aiohttp

from app.core.config import settings

logger = logging.getLogger("factory.image_gen")

_STYLE_ENRICHMENT = (
    "Flat vector illustration, editorial style, "
    "warm copper and stone tones, clean background. "
    "No text, no letters, no typography, no writing, no labels, no words, no characters. "
    "Pure visual illustration only. High quality."
)

_TOGETHER_IMAGES_URL = "https://api.together.xyz/v1/images/generations"

PLATFORM_DIMENSIONS: dict[str, tuple[int, int]] = {
    "instagram": (1088, 1344),
    "linkedin": (1088, 1344),
    "twitter": (1088, 1616),
    "tiktok": (1088, 1920),
    "youtube": (1920, 1088),
}

DEFAULT_DIMENSIONS = (1088, 1344)

# Class-level rate-limit state shared across all ImageGenerationService instances.
# Together AI's FLUX endpoint uses per-minute sliding-window quotas; this
# coordinator enforces a minimum gap between every API call and applies
# exponential backoff when a 429 is encountered.
_rate_limit_lock = asyncio.Lock()
_last_api_call: float = 0.0
_min_gap: float = 3.0  # seconds; doubles on each 429, capped at 60
_cooldown_until: float = 0.0


async def _wait_for_rate_limit() -> None:
    """Wait for any global cooldown, then enforce min gap between calls."""
    global _last_api_call, _cooldown_until
    async with _rate_limit_lock:
        now = time_module.monotonic()
        if now < _cooldown_until:
            await asyncio.sleep(_cooldown_until - now)
            now = time_module.monotonic()
        if _last_api_call > 0:
            gap = now - _last_api_call
            if gap < _min_gap:
                await asyncio.sleep(_min_gap - gap)
        _last_api_call = time_module.monotonic()


async def _on_rate_limit(headers_retry_after: Optional[float]) -> None:
    """Escalate cooldown after a 429.

    Doubles the minimum gap (exponential backoff shared across all
    slides/retries) and sets an absolute cooldown deadline.
    """
    global _min_gap, _cooldown_until
    async with _rate_limit_lock:
        if headers_retry_after is not None and headers_retry_after > 0:
            _cooldown_until = time_module.monotonic() + headers_retry_after
        else:
            _min_gap = min(_min_gap * 2, 60.0)
            _cooldown_until = time_module.monotonic() + _min_gap


def _parse_429_body(body: str) -> Optional[float]:
    """Try to extract a numeric retry-after from the 429 JSON body.

    Together AI's 429 response body includes a message referencing
    ``X-RateLimit-Reset`` but the actual HTTP header may be absent or zero.
    As a last resort, parse the ``error`` object for any ``retry_after_ms``
    or similar field, and also attempt to extract the reset value from the
    message text.
    """
    try:
        data = json.loads(body)
        error = data.get("error") or {}
        # Some API providers include a dedicated field
        for field in ("retry_after_ms", "retry_after", "retryAfter"):
            val = error.get(field)
            if val is not None:
                return float(val) / 1000 if field == "retry_after_ms" else float(val)
    except (json.JSONDecodeError, TypeError):
        pass
    # Nothing parseable — caller will apply the default exponential backoff
    return None


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
        retry_after: float = 0.0
        for attempt in range(1, self.max_retries + 1):
            await _wait_for_rate_limit()

            try:
                return await self._call_api(prompt, width, height)
            except asyncio.TimeoutError:
                logger.warning(
                    "Image gen timeout (attempt %d/%d) for: %s",
                    attempt,
                    self.max_retries,
                    prompt[:60],
                )
                last_exception = asyncio.TimeoutError("HTTP timeout")
            except aiohttp.ClientResponseError as e:
                logger.warning(
                    "Image gen HTTP %d (attempt %d/%d) for: %s | %s",
                    e.status,
                    attempt,
                    self.max_retries,
                    prompt[:60],
                    e.message[:500],
                )
                last_exception = e
                if e.status == 429:
                    ra = getattr(e, "retry_after", None)
                    if ra is not None:
                        retry_after = ra
                    await _on_rate_limit(ra)
            except aiohttp.ClientError as e:
                logger.warning(
                    "Image gen connection error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                last_exception = e

            if attempt < self.max_retries:
                delay = retry_after if retry_after > 0 else 2**attempt
                jitter = random.uniform(0, delay * 0.25)
                await asyncio.sleep(delay + jitter)

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
                if resp.status >= 400:
                    body = await resp.text()
                    err = aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=(),
                        status=resp.status,
                        message=body[:500],
                        headers=resp.headers,
                    )
                    if resp.status == 429:
                        # Try HTTP headers first (standard Retry-After /
                        # Together AI X-RateLimit-Reset)
                        ra_str = resp.headers.get("Retry-After") or resp.headers.get(
                            "X-RateLimit-Reset"
                        )
                        if ra_str:
                            try:
                                err.retry_after = float(ra_str)
                            except (ValueError, TypeError):
                                err.retry_after = 0.0
                        else:
                            # Fall back to parsing the JSON body for
                            # structured rate-limit info.
                            parsed = _parse_429_body(body)
                            if parsed is not None:
                                err.retry_after = parsed
                    raise err
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
        if (
            not result_data
            or not isinstance(result_data, list)
            or len(result_data) == 0
        ):
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


def make_generate_image_tool(
    image_service: ImageGenerationService | None = None,
) -> Tool:
    svc = image_service or ImageGenerationService()

    async def _generate(
        visual_description: str,
        platform: str = "",
    ) -> dict:
        result = await svc.generate(visual_description, platform)
        return {
            "success": result.success,
            "image_bytes": result.image_bytes,
            "width": result.width,
            "height": result.height,
            "failure_reason": result.failure_reason,
            "prompt_used": result.prompt_used,
        }

    return Tool(
        name="generate_image",
        description="Generate a carousel slide image via FLUX on Together AI.",
        callable=_generate,
        permissions={"CarouselImageAgent", "ShortVisualAssetAgent", "*"},
    )
