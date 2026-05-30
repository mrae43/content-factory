import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from app.core.config import Settings


class _IsolatedSettings(Settings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


_ENV_KEYS = [
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
    "TOGETHER_API_KEY",
    "DATABASE_URL",
    "POSTGRES_URI",
    "SYNTHID_WATERMARK_ENABLED",
    "MAX_RED_TEAM_REVISIONS",
    "SIMILARITY_THRESHOLD",
    "WORKER_POLL_INTERVAL_SECONDS",
    "WORKER_LOCK_TIMEOUT_MINUTES",
    "COPYWRITER_MODEL",
    "COPYWRITER_TEMPERATURE",
    "EVALUATOR_MODEL",
    "EVALUATOR_TEMPERATURE",
    "OPTIMIZER_MODEL",
    "OPTIMIZER_TEMPERATURE",
    "ASSET_MODEL",
    "ASSET_TEMPERATURE",
    "FORMATTER_MODEL",
    "FORMATTER_TEMPERATURE",
    "PROMOTION_MODEL",
    "PROMOTION_TEMPERATURE",
    "EVAL_COPYWRITER_MODEL",
    "EVAL_COPYWRITER_TEMPERATURE",
    "EVAL_RED_TEAM_MODEL",
    "EVAL_RED_TEAM_TEMPERATURE",
    "EVAL_OPTIMIZER_MODEL",
    "EVAL_OPTIMIZER_TEMPERATURE",
    "EVAL_JUDGE_MODEL",
    "EVAL_JUDGE_TEMPERATURE",
    "IMAGE_STORAGE_PATH",
    "STORAGE_BACKEND",
]


def _del_all_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.unit
class TestSettings:
    def test_raises_validation_error_when_gemini_api_key_missing(self, monkeypatch):
        _del_all_env(monkeypatch)
        with pytest.raises(ValidationError):
            _IsolatedSettings(tavily_api_key="test-key")

    def test_raises_validation_error_when_tavily_api_key_missing(self, monkeypatch):
        _del_all_env(monkeypatch)
        with pytest.raises(ValidationError):
            _IsolatedSettings(gemini_api_key="test-key")

    def test_raises_validation_error_when_no_api_keys_provided(self, monkeypatch):
        _del_all_env(monkeypatch)
        with pytest.raises(ValidationError):
            _IsolatedSettings()

    def test_instantiates_with_required_keys(self):
        s = Settings(gemini_api_key="g", tavily_api_key="t")
        assert s.gemini_api_key == "g"
        assert s.tavily_api_key == "t"

    def test_default_values(self, monkeypatch):
        _del_all_env(monkeypatch)
        with pytest.raises(ValidationError):
            _IsolatedSettings(gemini_api_key="g", tavily_api_key="t")

        s = _IsolatedSettings(
            gemini_api_key="g", tavily_api_key="t", database_url="postgresql://test"
        )
        assert s.synthid_watermark_enabled is True
        assert s.max_red_team_revisions == 3
        assert s.similarity_threshold == 0.75
        assert s.worker_poll_interval_seconds == 5
        assert s.worker_lock_timeout_minutes == 15
        assert s.image_storage_path == "static/carousel_images"
        assert s.storage_backend == "s3"

    def test_custom_values_override_defaults(self):
        s = Settings(
            gemini_api_key="g",
            tavily_api_key="t",
            database_url="postgresql://custom",
            synthid_watermark_enabled=False,
            max_red_team_revisions=5,
            similarity_threshold=0.9,
            worker_poll_interval_seconds=10,
            worker_lock_timeout_minutes=30,
        )
        assert s.database_url == "postgresql://custom"
        assert s.synthid_watermark_enabled is False
        assert s.max_red_team_revisions == 5
        assert s.similarity_threshold == 0.9
        assert s.worker_poll_interval_seconds == 10
        assert s.worker_lock_timeout_minutes == 30
        assert s.image_storage_path == "static/carousel_images"
        assert s.storage_backend == "s3"

    def test_custom_image_storage_path(self):
        s = Settings(
            gemini_api_key="g",
            tavily_api_key="t",
            database_url="postgresql://custom",
            image_storage_path="/custom/path",
        )
        assert s.image_storage_path == "/custom/path"

    def test_custom_storage_backend(self):
        s = Settings(
            gemini_api_key="g",
            tavily_api_key="t",
            database_url="postgresql://custom",
            storage_backend="s3",
        )
        assert s.storage_backend == "s3"

    def test_promotion_defaults(self):
        s = Settings(
            gemini_api_key="g",
            tavily_api_key="t",
            database_url="postgresql://custom",
        )
        assert s.promotion_model == "openai/gpt-oss-20b"
        assert s.promotion_temperature == 0.3


@pytest.mark.unit
class TestGetGuardrailConfig:
    def test_high_profile_defaults(self):
        from app.core.guardrails import get_guardrail_config, GuardrailStrictness

        cfg = get_guardrail_config(GuardrailStrictness.High)
        assert cfg.uncertain_is_soft_fail is True
        assert cfg.requires_human_review is True

    def test_high_profile_with_uncertain_pass_through(self):
        from app.core.guardrails import get_guardrail_config, GuardrailStrictness

        cfg = get_guardrail_config(
            GuardrailStrictness.High, uncertain_pass_through=True
        )
        assert cfg.uncertain_is_soft_fail is False
        assert cfg.requires_human_review is True

    def test_low_profile_unchanged(self):
        from app.core.guardrails import get_guardrail_config, GuardrailStrictness

        cfg = get_guardrail_config(GuardrailStrictness.Low)
        assert cfg.uncertain_is_soft_fail is False
        assert cfg.requires_human_review is False
        assert cfg.similarity_threshold == 0.65

    def test_medium_profile_unchanged(self):
        from app.core.guardrails import get_guardrail_config, GuardrailStrictness

        cfg = get_guardrail_config(GuardrailStrictness.Medium)
        assert cfg.uncertain_is_soft_fail is False
        assert cfg.requires_human_review is False
        assert cfg.similarity_threshold == 0.72

    def test_low_with_uncertain_pass_through_warns(self):
        from app.core.guardrails import get_guardrail_config, GuardrailStrictness

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = get_guardrail_config(
                GuardrailStrictness.Low, uncertain_pass_through=True
            )
            assert len(w) == 1
            assert "uncertain_pass_through=True has no effect" in str(w[0].message)
        assert cfg.requires_human_review is False

    def test_invalid_strictness_raises_value_error(self):
        from app.core.guardrails import get_guardrail_config

        import pytest

        with pytest.raises(ValueError, match="Unknown guardrail strictness"):
            get_guardrail_config("Invalid")  # type: ignore
