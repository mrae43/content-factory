from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings

def get_llm(model_name: str = "gemini-3.1-flash", temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Instantiates the Gemini 3.1 model.
    Defaults to Flash for AEO-optimized high-volume tasks.
    Switch to 'gemini-3.1-pro' for Red Team and Complex Strategy.
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        max_retries=3, 
    )

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    2026 standard high-dimensional embeddings for pgvector RAG extraction.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-004", # Advanced embedding model
        google_api_key=settings.gemini_api_key
    )