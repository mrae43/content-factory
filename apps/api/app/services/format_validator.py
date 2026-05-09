from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, ValidationError

from app.schemas.formats import BlogFormatPayload, CarouselFormatPayload


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
            return FormatValidationResult(
                valid=True,
                validated_payload=validated.model_dump(by_alias=True),
            )
        except ValidationError as e:
            return FormatValidationResult(
                valid=False,
                error_message=str(e),
            )


class CarouselValidator(FormatValidator):
    def __init__(self, platform: str = "default"):
        self.char_limit_map = {
            "twitter": 280,
            "linkedin": 700,
            "instagram": 2200,
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

        dump = validated.model_dump(by_alias=True)
        dump["char_limit_violations"] = []
        return FormatValidationResult(
            valid=True,
            validated_payload=dump,
        )
