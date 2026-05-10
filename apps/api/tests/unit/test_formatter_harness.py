import pytest
from unittest.mock import AsyncMock, MagicMock

from app.workers.agents import AgentActionStatus, AgentResult
from app.workers.harness import FormatterHarness, HarnessResult
from app.services.format_validator import FormatValidationResult


def _success_result(payload: dict) -> AgentResult:
    return AgentResult(
        status=AgentActionStatus.SUCCESS,
        payload=payload,
        reasoning="OK",
        confidence_score=0.9,
    )


def _error_result(reasoning: str = "Agent error") -> AgentResult:
    return AgentResult(
        status=AgentActionStatus.ERROR,
        payload={},
        reasoning=reasoning,
        confidence_score=0.0,
    )


def _always_valid_validator():
    validator = MagicMock()
    validator.validate.return_value = FormatValidationResult(
        valid=True,
        validated_payload={"_format": "blog", "title": "Test"},
    )
    return validator


def _always_invalid_validator(msg="Schema validation failed"):
    validator = MagicMock()
    validator.validate.return_value = FormatValidationResult(
        valid=False,
        error_message=msg,
    )
    return validator


@pytest.mark.unit
class TestFormatterHarnessSuccess:
    async def test_should_succeed_on_first_attempt(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            return_value=_success_result({"title": "Blog Post"})
        )
        validator = _always_valid_validator()
        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        result = await harness.run_with_harness(
            {"format_type": "blog", "script_content": "script"}
        )

        assert result.success is True
        assert result.attempts == 1
        assert result.payload == {"_format": "blog", "title": "Test"}
        assert result.error_log == []
        mock_agent.run.assert_awaited_once()

    async def test_should_set_format_type_from_context(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            return_value=_success_result({"title": "Blog"})
        )
        validator = _always_valid_validator()
        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        result = await harness.run_with_harness({"format_type": "carousel"})

        assert result.format_type == "carousel"

    async def test_should_default_format_type_to_unknown(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            return_value=_success_result({"title": "Blog"})
        )
        validator = _always_valid_validator()
        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        result = await harness.run_with_harness({})

        assert result.format_type == "unknown"


@pytest.mark.unit
class TestFormatterHarnessRetry:
    async def test_should_retry_on_validation_failure_and_succeed(self):
        mock_agent = AsyncMock()
        first_payload = {"title": "Bad"}
        second_payload = {"title": "Good"}
        mock_agent.run = AsyncMock(
            side_effect=[
                _success_result(first_payload),
                _success_result(second_payload),
            ]
        )

        validator = MagicMock()
        validator.validate.side_effect = [
            FormatValidationResult(valid=False, error_message="Missing sections"),
            FormatValidationResult(
                valid=True, validated_payload={"_format": "blog", "title": "Good"}
            ),
        ]

        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )
        context = {"format_type": "blog"}
        result = await harness.run_with_harness(context)

        assert result.success is True
        assert result.attempts == 2
        assert len(result.error_log) == 1
        assert "Missing sections" in result.error_log[0]
        assert context["correction_hint"] == "Missing sections"

    async def test_should_retry_on_agent_error_and_succeed(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=[
                _error_result("API timeout"),
                _success_result({"title": "Blog"}),
            ]
        )
        validator = _always_valid_validator()
        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        context = {"format_type": "blog"}
        result = await harness.run_with_harness(context)

        assert result.success is True
        assert result.attempts == 2
        assert "API timeout" in result.error_log[0]
        assert context["correction_hint"] == "Agent failed: API timeout"

    async def test_should_fail_after_exhausting_all_retries(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=[
                _success_result({"title": "Bad", "attempt": 1}),
                _success_result({"title": "Bad", "attempt": 2}),
                _success_result({"title": "Bad", "attempt": 3}),
            ]
        )
        validator = _always_invalid_validator("Always fails")
        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        result = await harness.run_with_harness({"format_type": "blog"})

        assert result.success is False
        assert result.attempts == 3
        assert len(result.error_log) == 3

    async def test_should_fail_when_agent_always_errors(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=_error_result("Boom"))
        validator = _always_valid_validator()
        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        result = await harness.run_with_harness({"format_type": "blog"})

        assert result.success is False
        assert result.attempts == 3
        assert len(result.error_log) == 3
        for entry in result.error_log:
            assert "Agent failed" in entry


@pytest.mark.unit
class TestFormatterHarnessDoomLoop:
    async def test_should_break_on_identical_payload_hash(self):
        identical_payload = {"title": "Same", "sections": []}
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=[
                _success_result(identical_payload),
                _success_result(identical_payload),
            ]
        )

        validator = MagicMock()
        validator.validate.side_effect = [
            FormatValidationResult(valid=False, error_message="Validation fail 1"),
            FormatValidationResult(valid=False, error_message="Validation fail 2"),
        ]

        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        result = await harness.run_with_harness({"format_type": "blog"})

        assert result.success is False
        assert result.attempts == 2
        assert any("Doom loop" in e for e in result.error_log)

    async def test_should_not_break_when_payloads_differ(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=[
                _success_result({"title": "First", "sections": []}),
                _success_result({"title": "Second", "sections": []}),
            ]
        )

        validator = MagicMock()
        validator.validate.side_effect = [
            FormatValidationResult(valid=False, error_message="Fail 1"),
            FormatValidationResult(
                valid=True, validated_payload={"_format": "blog", "title": "Second"}
            ),
        ]

        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        result = await harness.run_with_harness({"format_type": "blog"})

        assert result.success is True
        assert result.attempts == 2


@pytest.mark.unit
class TestFormatterHarnessCorrectiveContext:
    async def test_should_inject_correction_hint_on_agent_error(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=[
                _error_result("Model returned empty"),
                _success_result({"title": "Fixed"}),
            ]
        )
        validator = _always_valid_validator()
        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        context = {"format_type": "blog"}
        await harness.run_with_harness(context)

        assert "correction_hint" in context
        assert "Agent failed" in context["correction_hint"]

    async def test_should_inject_correction_hint_on_validation_failure(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=[
                _success_result({"title": "Bad"}),
                _success_result({"title": "Good"}),
            ]
        )

        validator = MagicMock()
        validator.validate.side_effect = [
            FormatValidationResult(
                valid=False, error_message="Missing required field: seo_meta"
            ),
            FormatValidationResult(
                valid=True, validated_payload={"_format": "blog"}
            ),
        ]

        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        context = {"format_type": "blog"}
        await harness.run_with_harness(context)

        assert context["correction_hint"] == "Missing required field: seo_meta"

    async def test_should_update_correction_hint_per_attempt(self):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=[
                _success_result({"title": "A"}),
                _success_result({"title": "B"}),
                _success_result({"title": "C"}),
            ]
        )

        validator = MagicMock()
        validator.validate.side_effect = [
            FormatValidationResult(valid=False, error_message="Error A"),
            FormatValidationResult(valid=False, error_message="Error B"),
            FormatValidationResult(valid=True, validated_payload={"_format": "blog"}),
        ]

        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=2
        )

        context = {"format_type": "blog"}
        result = await harness.run_with_harness(context)

        assert result.success is True
        assert context["correction_hint"] == "Error B"


@pytest.mark.unit
class TestFormatterHarnessMaxRetries:
    @pytest.mark.parametrize("max_retries,expected_attempts", [(0, 1), (1, 2), (2, 3)])
    async def test_should_respect_max_retries_setting(
        self, max_retries, expected_attempts
    ):
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=_error_result("Fail"))
        validator = _always_valid_validator()
        harness = FormatterHarness(
            formatter=mock_agent, validator=validator, max_retries=max_retries
        )

        result = await harness.run_with_harness({"format_type": "blog"})

        assert result.success is False
        assert result.attempts == expected_attempts


@pytest.mark.unit
class TestHarnessResultModel:
    def test_should_create_failure_result_with_defaults(self):
        result = HarnessResult(success=False, format_type="blog")

        assert result.success is False
        assert result.format_type == "blog"
        assert result.payload is None
        assert result.error_log == []
        assert result.attempts == 0

    def test_should_create_success_result(self):
        result = HarnessResult(
            success=True,
            format_type="carousel",
            payload={"title": "Test"},
            attempts=2,
        )

        assert result.success is True
        assert result.payload == {"title": "Test"}
        assert result.attempts == 2


@pytest.mark.unit
class TestFormatterHarnessPayloadHash:
    def test_should_produce_deterministic_hash(self):
        payload = {"title": "Test", "items": [1, 2, 3]}
        h1 = FormatterHarness._hash_payload(payload)
        h2 = FormatterHarness._hash_payload(payload)

        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64

    def test_should_hash_differently_for_different_payloads(self):
        h1 = FormatterHarness._hash_payload({"title": "A"})
        h2 = FormatterHarness._hash_payload({"title": "B"})

        assert h1 != h2
