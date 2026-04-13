import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import ResearchChunk
from app.services.llm import get_embeddings

logger = logging.getLogger(__name__)


class ContentFactoryVectorStore:
    """
    2026 Standard: Unified pgvector abstraction layer.
    Handles semantic routing and chunk isolation via metadata.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.embedder = get_embeddings()

    async def ingest_chunks(
        self, job_id: UUID, chunks: List[str], scope: str = "LOCAL"
    ) -> int:
        """
        Embeds text chunks via Gemini 3.1 and inserts them into the pgvector table.
        Scope defaults to LOCAL (ephemeral for this specific render job).
        """
        if not chunks:
            return 0

        logger.info(f"Embedding {len(chunks)} chunks for job {job_id} [Scope {scope}]")
        # Batch generate embeddings via Gemini Native integration
        embeddings = await self.embedder.aembed_documents(chunks)

        db_chunks = []
        for content, embedding in zip(chunks, embeddings):
            db_chunks.append(
                ResearchChunk(
                    job_id=job_id,
                    content=content,
                    embedding=embedding,
                    meta={"scope": scope, "version": "1.0"},
                )
            )

        self.db.add_all(db_chunks)
        await self.db.commit()
        logger.info(f"Successfully ingested {len(db_chunks)} chunks.")
        return len(db_chunks)

    async def semantic_search(
        self,
        query: str,
        job_id: Optional[UUID] = None,
        scope: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Actionable RAG: Search the HNSW-indexed pgvector table.
        Filters by job_id (LOCAL) or scope (GLOBAL).
        """
        # Embed the search query
        query_embedding = await self.embedder.aembed_query(query)

        # Base query using vector cosine distance
        stmt = (
            select(ResearchChunk)
            .order_by(ResearchChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        # Apply Governance-as-Code Metadata Filtering
        if job_id:
            stmt = stmt.where(ResearchChunk.job_id == job_id)
        if scope:
            # Filtering inside JSONB column
            stmt = stmt.where(ResearchChunk.meta["scope"].astext == scope)

        result = await self.db.execute(stmt)
        chunks = result.scalars().all()

        return [
            {
                "id": str(c.id),
                "content": c.content,
                "meta": c.meta,
                "job_id": str(c.job_id),
            }
            for c in chunks
        ]
