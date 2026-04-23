from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings


def get_llm(
    model_name: str = "gemini-2.5-flash", temperature: float = 0.2
) -> ChatGoogleGenerativeAI:
    """
    Instantiates a Gemini model.
    Defaults to 2.5 Flash for high-volume tasks.
    Switch to 'gemini-1.5-pro' for Red Team and Complex Strategy.
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
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
