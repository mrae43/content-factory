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
    research_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    research_temperature: float = 0.2
    copywriter_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    copywriter_temperature: float = 0.7
    evaluator_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    evaluator_temperature: float = 0.0
    optimizer_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    optimizer_temperature: float = 0.3
    asset_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    asset_temperature: float = 0.5
    formatter_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    formatter_temperature: float = 0.3

    # Image Generation
    image_model: str = "black-forest-labs/FLUX.1-schnell"
    image_gen_max_retries: int = 3
    image_gen_timeout_seconds: int = 30
    image_storage_path: str = "static/carousel_images"
    storage_backend: str = "local"

    # Eval Model Configuration (Together AI)
    eval_research_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    eval_research_temperature: float = 0.2
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

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
