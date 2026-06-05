import math
from typing import List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

from app.core.config import settings

# ── Provider Registry ──────────────────────────────────────────────────────────

PROVIDERS = {
    "gemini": {
        "class": ChatGoogleGenerativeAI,
        "api_key_attr": "gemini_api_key",
    },
    "together": {
        "class": ChatOpenAI,
        "base_url": "https://api.together.xyz/v1",
        "api_key_attr": "together_api_key",
    },
}

_DEFAULT_PROVIDER = "together"


def _resolve_provider(model_name: str) -> Tuple[str, str, dict]:
    """Parse ``provider:model`` prefix or fall back to backward-compatible rules."""
    if ":" in model_name:
        provider_key, model = model_name.split(":", 1)
    elif model_name.startswith("gemini"):
        provider_key, model = "gemini", model_name
    else:
        provider_key, model = _DEFAULT_PROVIDER, model_name

    config = PROVIDERS.get(provider_key)
    if config is None:
        raise ValueError(
            f"Unknown provider '{provider_key}'. "
            f"Available: {', '.join(PROVIDERS)}. "
            "Use the `provider:model` convention (e.g. `together:meta-llama/...`)."
        )
    return provider_key, model, config


def get_llm(
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> BaseChatModel:
    provider_key, model, config = _resolve_provider(model_name)
    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_retries": 3,
    }
    api_key = getattr(settings, config["api_key_attr"], None)
    if api_key:
        if provider_key == "gemini":
            kwargs["google_api_key"] = api_key
        else:
            kwargs["api_key"] = api_key

    base_url = config.get("base_url")
    if base_url:
        kwargs["base_url"] = base_url

    if provider_key != "gemini":
        kwargs["max_tokens"] = max_tokens or 4096

    return config["class"](**kwargs)


# ── Embedding Normalization ────────────────────────────────────────────────────


def _l2_normalize(vector: List[float]) -> List[float]:
    """L2-normalize a single vector in-place."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class _NormalizedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """Wrapper that L2-normalises every output vector.

    ``gemini-embedding-001`` at ``output_dimensionality < 3072`` does **not**
    normalise by default.  Without manual normalisation ``cosine_distance``
    returns distorted rankings.
    """

    async def aembed_documents(self, texts):
        vectors = await super().aembed_documents(texts)
        return [_l2_normalize(v) for v in vectors]

    async def aembed_query(self, text):
        vector = await super().aembed_query(text)
        return _l2_normalize(vector)

    def embed_documents(self, texts):
        vectors = super().embed_documents(texts)
        return [_l2_normalize(v) for v in vectors]

    def embed_query(self, text):
        vector = super().embed_query(text)
        return _l2_normalize(vector)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return _NormalizedGoogleEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_document",
        output_dimensionality=settings.embedding_dimension,
    )


def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    return _NormalizedGoogleEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_query",
        output_dimensionality=settings.embedding_dimension,
    )
