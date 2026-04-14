from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Core Configuration for the 2026 Content Factory.
    Enforces strict typing and fast-fails if API keys are missing.
    """

    gemini_api_key: str
    postgres_uri: Optional[str] = None  # For pgvector integration later

    # 2026 Governance & Compliance Standards
    synthid_watermark_enabled: bool = True
    max_red_team_revisions: int = 3
    similarity_threshold: float = 0.75

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
