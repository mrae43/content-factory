from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Core Configuration for the 2026 Content Factory.
    Enforces strict typing and fast-fails if API keys are missing.
    """

    gemini_api_key: str
    tavily_api_key: str
    postgres_uri: Optional[str] = None

    # 2026 Governance & Compliance Standards
    synthid_watermark_enabled: bool = True
    max_red_team_revisions: int = 3
    similarity_threshold: float = 0.75

    # Evaluator-Optimizer Model Configuration
    evaluator_model: str = "gemini-1.5-pro"
    evaluator_temperature: float = 0.0
    optimizer_model: str = "gemini-2.5-flash"
    optimizer_temperature: float = 0.3

    # Queue Worker
    worker_poll_interval_seconds: int = 5
    worker_lock_timeout_minutes: int = 15

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
