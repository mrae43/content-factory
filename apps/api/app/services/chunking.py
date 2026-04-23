import logging
from typing import List
from langchain_text_splitters import MarkdownTextSplitter

logger = logging.getLogger("factory.chunking")


def split_pre_context(
    text: str, chunk_size: int = 2000, chunk_overlap: int = 400
) -> List[str]:
    """
    MVP text chunking strategy using a context-engineering mindset.
    Ensures cohesive thoughts and paragraphs are retained.
    """
    if not text or not text.strip():
        return []

    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )

    return splitter.split_text(text)


async def process_extraction_job(job_id: str, raw_text: str) -> List[str]:
    """
    Takes raw pre-context or fetched content and splits it into logical
    token bounds for the RAG vector store.
    """
    logger.info(f"Extracting and chunking data for job {job_id}")
    chunks = split_pre_context(raw_text)
    logger.info(f"Generated {len(chunks)} contextual chunks for Job {job_id}")
    return chunks
