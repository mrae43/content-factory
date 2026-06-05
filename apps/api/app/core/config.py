from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / ".env").exists() or (p / "nx.json").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root()


class Settings(BaseSettings):
    """
    Core Configuration for the 2026 Content Factory.
    Enforces strict typing and fast-fails if API keys are missing.
    """

    gemini_api_key: str
    together_api_key: str = ""
    tavily_api_key: str
    database_url: str

    # 2026 Governance & Compliance Standards
    synthid_watermark_enabled: bool = True
    max_red_team_revisions: int = 3
    retrieval_retry_max: int = 3
    similarity_threshold: float = 0.75

    # Agent Model Configuration (Together AI)
    # Premium-tier models: meta-llama/Llama-3.3-70B-Instruct-Turbo
    # Standard-tier models: openai/gpt-oss-20b
    copywriter_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    copywriter_temperature: float = 0.7
    evaluator_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    evaluator_temperature: float = 0.0
    optimizer_model: str = "openai/gpt-oss-20b"
    optimizer_temperature: float = 0.3
    asset_model: str = "openai/gpt-oss-20b"
    asset_temperature: float = 0.5
    formatter_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    formatter_temperature: float = 0.3
    promotion_model: str = "openai/gpt-oss-20b"
    promotion_temperature: float = 0.3

    # Image Generation
    image_model: str = "black-forest-labs/FLUX.1-schnell"
    image_gen_max_retries: int = 3
    image_gen_timeout_seconds: int = 30
    image_gen_min_gap: float = 3.0
    image_storage_path: str = "static/carousel_images"
    image_gen_slide_delay: float = 1.5

    # Video Generation
    video_gen_provider: str = "kling"
    video_gen_model: str = "kling-v1-6"
    video_gen_mode: str = "std"
    video_gen_poll_interval_seconds: int = 5
    video_gen_max_poll_retries: int = 60

    # Kling AI
    kling_access_key: str = ""
    kling_secret_key: str = ""
    kling_base_url: str = "https://api-singapore.klingai.com/v1"

    # TTS (Text-to-Speech)
    tts_provider: str = "elevenlabs"
    tts_model_id: str = "eleven_flash_v2_5"
    tts_api_key: str = ""
    default_voice_map: dict[str, str] = {}

    storage_backend: str = "s3"

    # S3 / SeaweedFS
    s3_endpoint_url: str = "http://seaweedfs:8333"
    s3_access_key_id: str = "factory"
    s3_secret_access_key: str = "factory-secret"
    s3_bucket_images: str = "media-images"
    s3_bucket_videos: str = "media-videos"
    s3_bucket_music: str = "media-music"
    # Public URL for browser-accessible image URLs.
    # Docker dev: localhost reaches the host where SeaweedFS exposes port 8333.
    # Remote deploys: set to the external domain/port of your S3-compatible gateway.
    s3_public_url: str = "http://localhost:8333"

    # Embedding Model
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimension: int = 768

    # Eval Model Configuration (Together AI)
    eval_copywriter_model: str = "MiniMaxAI/MiniMax-M2.7"
    eval_copywriter_temperature: float = 0.7
    eval_red_team_model: str = "openai/gpt-oss-120b"
    eval_red_team_temperature: float = 0.0
    eval_optimizer_model: str = "openai/gpt-oss-20b"
    eval_optimizer_temperature: float = 0.3
    eval_judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
    eval_judge_temperature: float = 0.0

    # Context Builder
    context_builder_top_k: int = 10

    # Queue Worker
    worker_poll_interval_seconds: int = 5
    worker_lock_timeout_minutes: int = 15

    # Discord Bot
    discord_token: str = ""
    discord_guild_id: int = 0
    discord_channel_id: int | None = None

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
