import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from app.core.config import Settings


class _IsolatedSettings(Settings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


_ENV_KEYS = [
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
    "POSTGRES_URI",
    "SYNTHID_WATERMARK_ENABLED",
    "MAX_RED_TEAM_REVISIONS",
    "SIMILARITY_THRESHOLD",
    "WORKER_POLL_INTERVAL_SECONDS",
    "WORKER_LOCK_TIMEOUT_MINUTES",
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
        s = _IsolatedSettings(gemini_api_key="g", tavily_api_key="t")
        assert s.postgres_uri is None
        assert s.synthid_watermark_enabled is True
        assert s.max_red_team_revisions == 3
        assert s.similarity_threshold == 0.75
        assert s.worker_poll_interval_seconds == 5
        assert s.worker_lock_timeout_minutes == 15

    def test_custom_values_override_defaults(self):
        s = Settings(
            gemini_api_key="g",
            tavily_api_key="t",
            postgres_uri="postgresql://custom",
            synthid_watermark_enabled=False,
            max_red_team_revisions=5,
            similarity_threshold=0.9,
            worker_poll_interval_seconds=10,
            worker_lock_timeout_minutes=30,
        )
        assert s.postgres_uri == "postgresql://custom"
        assert s.synthid_watermark_enabled is False
        assert s.max_red_team_revisions == 5
        assert s.similarity_threshold == 0.9
        assert s.worker_poll_interval_seconds == 10
        assert s.worker_lock_timeout_minutes == 30
