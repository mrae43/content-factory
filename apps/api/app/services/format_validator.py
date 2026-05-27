from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, ValidationError

from app.services.tools import Tool

from app.schemas.formats import (
    BlogFormatPayload,
    CarouselFormatPayload,
    VideoFormatPayload,
)


class FormatValidationResult(BaseModel):
    valid: bool
    error_message: Optional[str] = None
    validated_payload: Optional[dict] = None


class FormatValidator(ABC):
    @abstractmethod
    def validate(self, payload: dict) -> FormatValidationResult: ...


class BlogValidator(FormatValidator):
    def validate(self, payload: dict) -> FormatValidationResult:
        try:
            validated = BlogFormatPayload.model_validate(payload)
        except ValidationError as e:
            return FormatValidationResult(
                valid=False,
                error_message=str(e),
            )

        if len(validated.sections) < 1:
            return FormatValidationResult(
                valid=False,
                error_message="Blog must have at least 1 section.",
            )

        return FormatValidationResult(
            valid=True,
            validated_payload=validated.model_dump(by_alias=True, mode="json"),
        )


class CarouselValidator(FormatValidator):
    def __init__(self, platform: str = "default"):
        self.char_limit_map = {
            "twitter": 280,
            "linkedin": 700,
            "instagram": 2200,
            "tiktok": 2200,
        }
        self.platform = platform
        self.char_limit = self.char_limit_map.get(platform, 500)

    def validate(self, payload: dict) -> FormatValidationResult:
        try:
            validated = CarouselFormatPayload.model_validate(payload)
        except ValidationError as e:
            return FormatValidationResult(
                valid=False,
                error_message=str(e),
            )

        violations = [
            f"Slide {s.slide_number}: {len(s.text)} chars (limit: {self.char_limit})"
            for s in validated.slides
            if len(s.text) > self.char_limit
        ]

        if violations:
            violation_detail = "; ".join(violations)
            return FormatValidationResult(
                valid=False,
                error_message=(
                    f"Char limit violations for platform '{self.platform}' "
                    f"(limit: {self.char_limit}): {violation_detail}. "
                    "Shorten the text on the flagged slides."
                ),
            )

        dump = validated.model_dump(by_alias=True, mode="json")
        dump["char_limit_violations"] = []
        return FormatValidationResult(
            valid=True,
            validated_payload=dump,
        )


def make_validate_format_tool(
    validator: FormatValidator,
) -> Tool:
    """Create a Tool wrapping ``FormatValidator.validate``.

    The returned ``Tool`` is DI-only and wraps the given validator instance
    so that AgentHarness can invoke format validation via the tool registry.
    """

    async def _validate(payload: dict) -> dict:
        result = validator.validate(payload)
        return result.model_dump()

    return Tool(
        name="validate_format",
        description="Validate a formatted payload against format-specific rules.",
        callable=_validate,
        permissions={"AgentHarness", "*"},
    )


class VideoValidator(FormatValidator):
    def validate(self, payload: dict) -> FormatValidationResult:
        try:
            validated = VideoFormatPayload.model_validate(payload)
        except ValidationError as e:
            return FormatValidationResult(
                valid=False,
                error_message=str(e),
            )

        if len(validated.scenes) < 3:
            return FormatValidationResult(
                valid=False,
                error_message=f"Video must have at least 3 scenes, got {len(validated.scenes)}.",
            )

        empty_scenes = [
            s.scene_number
            for s in validated.scenes
            if not s.visual_prompt.strip() or not s.narration_text.strip()
        ]
        if empty_scenes:
            return FormatValidationResult(
                valid=False,
                error_message=(
                    f"Scenes {empty_scenes} have empty visual_prompt or narration_text. "
                    "Every scene must have both fields populated."
                ),
            )

        return FormatValidationResult(
            valid=True,
            validated_payload=validated.model_dump(by_alias=True, mode="json"),
        )
