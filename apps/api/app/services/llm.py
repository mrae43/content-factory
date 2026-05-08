from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from app.core.config import settings

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def get_llm(
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature: float = 0.2,
) -> BaseChatModel:
    if model_name.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
            max_retries=3,
        )
    return ChatOpenAI(
        model=model_name,
        api_key=settings.together_api_key,
        base_url=_TOGETHER_BASE_URL,
        temperature=temperature,
        max_tokens=4096,
        max_retries=3,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_document",
        output_dimensionality=768,
    )


def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_query",
        output_dimensionality=768,
    )
