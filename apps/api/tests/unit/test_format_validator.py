import pytest
from uuid import uuid4

from app.services.format_validator import (
    BlogValidator,
    CarouselValidator,
    FormatValidationResult,
)


def _valid_blog_payload(**overrides):
    base = {
        "_format": "blog",
        "_version": 1,
        "title": "Test Blog Title",
        "subtitle": "A test subtitle",
        "sections": [
            {
                "heading": "Intro",
                "body": "This is the intro section body.",
                "key_takeaway": "Key takeaway here",
                "sources_used": [str(uuid4())],
                "word_count": 6,
            }
        ],
        "seo_meta": {
            "meta_title": "Test Meta Title",
            "meta_description": "A test meta description for SEO.",
            "keywords": ["test", "blog"],
        },
        "tags": ["test"],
        "call_to_action": "Read more!",
    }
    base.update(overrides)
    return base


def _valid_carousel_payload(**overrides):
    base = {
        "_format": "carousel",
        "_version": 1,
        "slides": [
            {
                "slide_number": 1,
                "text": "Short text",
                "visual_prompt": "A visual",
                "hook_type": "question",
                "sources_used": [str(uuid4())],
            },
            {
                "slide_number": 2,
                "text": "Another short text",
                "visual_prompt": "Another visual",
                "hook_type": "statistic",
                "sources_used": [],
            },
        ],
        "thread_title": "Test Thread",
        "hashtags": ["test"],
        "cta_slide": "Follow for more!",
        "char_limit_violations": [],
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestBlogValidator:
    def test_should_validate_correct_payload(self):
        validator = BlogValidator()
        payload = _valid_blog_payload()

        result = validator.validate(payload)

        assert result.valid is True
        assert result.validated_payload is not None
        assert result.validated_payload["_format"] == "blog"
        assert result.validated_payload["_version"] == 1
        assert result.error_message is None

    def test_should_reject_missing_title(self):
        validator = BlogValidator()
        payload = _valid_blog_payload()
        del payload["title"]

        result = validator.validate(payload)

        assert result.valid is False
        assert result.error_message is not None
        assert "title" in result.error_message.lower()
        assert result.validated_payload is None

    def test_should_reject_missing_sections(self):
        validator = BlogValidator()
        payload = _valid_blog_payload()
        del payload["sections"]

        result = validator.validate(payload)

        assert result.valid is False
        assert result.error_message is not None
        assert "sections" in result.error_message.lower()

    def test_should_reject_empty_sections_list(self):
        validator = BlogValidator()
        payload = _valid_blog_payload(sections=[])

        result = validator.validate(payload)

        assert result.valid is True
        assert result.validated_payload is not None
        assert result.validated_payload["sections"] == []

    def test_should_reject_section_missing_body(self):
        validator = BlogValidator()
        section = {
            "heading": "Intro",
            "key_takeaway": "Takeaway",
            "sources_used": [],
            "word_count": 0,
        }
        payload = _valid_blog_payload(sections=[section])

        result = validator.validate(payload)

        assert result.valid is False
        assert "body" in result.error_message.lower()

    def test_should_reject_missing_seo_meta(self):
        validator = BlogValidator()
        payload = _valid_blog_payload()
        del payload["seo_meta"]

        result = validator.validate(payload)

        assert result.valid is False
        assert "seo_meta" in result.error_message.lower()

    def test_should_reject_seo_meta_missing_meta_title(self):
        validator = BlogValidator()
        seo = {"meta_description": "desc", "keywords": ["k"]}
        payload = _valid_blog_payload(seo_meta=seo)

        result = validator.validate(payload)

        assert result.valid is False
        assert "meta_title" in result.error_message.lower()

    def test_should_validate_with_optional_sources_used(self):
        validator = BlogValidator()
        section = {
            "heading": "No Sources",
            "body": "Body text",
            "key_takeaway": "Takeaway",
            "word_count": 2,
        }
        payload = _valid_blog_payload(sections=[section])

        result = validator.validate(payload)

        assert result.valid is True
        assert result.validated_payload["sections"][0]["sources_used"] == []

    def test_should_reject_invalid_sources_used_uuid(self):
        validator = BlogValidator()
        section = {
            "heading": "Bad UUID",
            "body": "Body",
            "key_takeaway": "Take",
            "sources_used": ["not-a-uuid"],
            "word_count": 1,
        }
        payload = _valid_blog_payload(sections=[section])

        result = validator.validate(payload)

        assert result.valid is False

    def test_should_reject_wrong_format_discriminator(self):
        validator = BlogValidator()
        payload = _valid_blog_payload(**{"_format": "carousel"})

        result = validator.validate(payload)

        assert result.valid is False

    def test_should_handle_empty_dict(self):
        validator = BlogValidator()

        result = validator.validate({})

        assert result.valid is False
        assert result.error_message is not None

    def test_should_use_alias_names_in_output(self):
        validator = BlogValidator()
        payload = _valid_blog_payload()

        result = validator.validate(payload)

        assert result.valid is True
        assert "_format" in result.validated_payload
        assert "_version" in result.validated_payload


@pytest.mark.unit
class TestCarouselValidator:
    def test_should_validate_correct_payload(self):
        validator = CarouselValidator()
        payload = _valid_carousel_payload()

        result = validator.validate(payload)

        assert result.valid is True
        assert result.validated_payload is not None
        assert result.validated_payload["_format"] == "carousel"
        assert result.error_message is None

    def test_should_reject_missing_slides(self):
        validator = CarouselValidator()
        payload = _valid_carousel_payload()
        del payload["slides"]

        result = validator.validate(payload)

        assert result.valid is False
        assert "slides" in result.error_message.lower()

    def test_should_reject_slide_text_exceeding_default_char_limit(self):
        validator = CarouselValidator(platform="default")
        payload = _valid_carousel_payload(
            slides=[
                {
                    "slide_number": 1,
                    "text": "x" * 501,
                    "visual_prompt": "Visual",
                    "hook_type": "question",
                    "sources_used": [],
                }
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "char limit" in result.error_message.lower()
        assert "500" in result.error_message

    def test_should_pass_twitter_char_limit_with_short_text(self):
        validator = CarouselValidator(platform="twitter")
        payload = _valid_carousel_payload(
            slides=[
                {
                    "slide_number": 1,
                    "text": "x" * 280,
                    "visual_prompt": "Visual",
                    "hook_type": "question",
                    "sources_used": [],
                }
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True

    def test_should_reject_twitter_char_limit_exceeded(self):
        validator = CarouselValidator(platform="twitter")
        payload = _valid_carousel_payload(
            slides=[
                {
                    "slide_number": 1,
                    "text": "x" * 281,
                    "visual_prompt": "Visual",
                    "hook_type": "question",
                    "sources_used": [],
                }
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "280" in result.error_message
        assert "Slide 1" in result.error_message

    def test_should_pass_linkedin_char_limit(self):
        validator = CarouselValidator(platform="linkedin")
        payload = _valid_carousel_payload(
            slides=[
                {
                    "slide_number": 1,
                    "text": "x" * 700,
                    "visual_prompt": "Visual",
                    "hook_type": "statistic",
                    "sources_used": [],
                }
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True

    def test_should_reject_linkedin_char_limit_exceeded(self):
        validator = CarouselValidator(platform="linkedin")
        payload = _valid_carousel_payload(
            slides=[
                {
                    "slide_number": 1,
                    "text": "x" * 701,
                    "visual_prompt": "Visual",
                    "hook_type": "statistic",
                    "sources_used": [],
                }
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "700" in result.error_message

    def test_should_pass_instagram_char_limit(self):
        validator = CarouselValidator(platform="instagram")
        payload = _valid_carousel_payload(
            slides=[
                {
                    "slide_number": 1,
                    "text": "x" * 2200,
                    "visual_prompt": "Visual",
                    "hook_type": "visual",
                    "sources_used": [],
                }
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True

    def test_should_reject_instagram_char_limit_exceeded(self):
        validator = CarouselValidator(platform="instagram")
        payload = _valid_carousel_payload(
            slides=[
                {
                    "slide_number": 1,
                    "text": "x" * 2201,
                    "visual_prompt": "Visual",
                    "hook_type": "visual",
                    "sources_used": [],
                }
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "2200" in result.error_message

    def test_should_report_multiple_violations(self):
        validator = CarouselValidator(platform="twitter")
        payload = _valid_carousel_payload(
            slides=[
                {
                    "slide_number": 1,
                    "text": "x" * 300,
                    "visual_prompt": "Visual",
                    "hook_type": "question",
                    "sources_used": [],
                },
                {
                    "slide_number": 2,
                    "text": "y" * 290,
                    "visual_prompt": "Visual",
                    "hook_type": "statistic",
                    "sources_used": [],
                },
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "Slide 1" in result.error_message
        assert "Slide 2" in result.error_message

    def test_should_clear_char_limit_violations_on_valid_payload(self):
        validator = CarouselValidator()
        payload = _valid_carousel_payload(
            char_limit_violations=["old violation"]
        )

        result = validator.validate(payload)

        assert result.valid is True
        assert result.validated_payload["char_limit_violations"] == []

    def test_should_reject_missing_thread_title(self):
        validator = CarouselValidator()
        payload = _valid_carousel_payload()
        del payload["thread_title"]

        result = validator.validate(payload)

        assert result.valid is False
        assert "thread_title" in result.error_message.lower()

    def test_should_reject_missing_cta_slide(self):
        validator = CarouselValidator()
        payload = _valid_carousel_payload()
        del payload["cta_slide"]

        result = validator.validate(payload)

        assert result.valid is False
        assert "cta_slide" in result.error_message.lower()

    def test_should_use_default_platform_when_unknown(self):
        validator = CarouselValidator(platform="unknown_platform")
        assert validator.char_limit == 500

    def test_should_reject_wrong_format_discriminator(self):
        validator = CarouselValidator()
        payload = _valid_carousel_payload(**{"_format": "blog"})

        result = validator.validate(payload)

        assert result.valid is False

    def test_should_handle_empty_dict(self):
        validator = CarouselValidator()

        result = validator.validate({})

        assert result.valid is False
        assert result.error_message is not None


@pytest.mark.unit
class TestFormatValidationResult:
    def test_should_create_valid_result(self):
        result = FormatValidationResult(
            valid=True, validated_payload={"_format": "blog"}
        )

        assert result.valid is True
        assert result.error_message is None
        assert result.validated_payload == {"_format": "blog"}

    def test_should_create_invalid_result(self):
        result = FormatValidationResult(valid=False, error_message="Missing field")

        assert result.valid is False
        assert result.error_message == "Missing field"
        assert result.validated_payload is None
