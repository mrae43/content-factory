import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import ResearchChunk
from app.services.llm import get_embeddings, get_query_embeddings
from app.services.tools import Tool
from app.core.config import settings

logger = logging.getLogger(__name__)


class ContentFactoryVectorStore:
    """
    2026 Standard: Unified pgvector abstraction layer.
    Handles semantic routing and chunk isolation via metadata.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.embedder = get_embeddings()
        self.query_embedder = get_query_embeddings()

    async def ingest_chunks(
        self,
        job_id: UUID,
        chunks: List[str],
        scope: str = "LOCAL",
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Embeds text chunks via Gemini and inserts them into the pgvector table.
        Scope defaults to LOCAL (ephemeral for this specific render job).
        Optional meta dict is merged into the default metadata.
        """
        if not chunks:
            return 0

        logger.info(f"Embedding {len(chunks)} chunks for job {job_id} [Scope {scope}]")
        embeddings = await self.embedder.aembed_documents(chunks)

        db_chunks = []
        for content, embedding in zip(chunks, embeddings):
            chunk_meta = {"scope": scope, "version": "1.0"}
            if meta:
                chunk_meta.update(meta)
            db_chunks.append(
                ResearchChunk(
                    job_id=job_id,
                    content=content,
                    embedding=embedding,
                    meta=chunk_meta,
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
        scopes: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Actionable RAG: Search the HNSW-indexed pgvector table.

        Args:
            query: The text to search semantically.
            job_id: Filter to chunks belonging to a specific job.
            scope: Single scope filter (e.g. "RAW-CONTEXT").
            scopes: Multi-scope filter (e.g. ["RAW-CONTEXT", "LOCAL"]).
                    Takes precedence over `scope` if both provided.
            top_k: Maximum number of results to return.
            similarity_threshold: Minimum cosine similarity (0.0-1.0).
                                  Defaults to settings.similarity_threshold.
                                  Set to 0.0 to disable filtering.
        """
        if similarity_threshold is None:
            similarity_threshold = settings.similarity_threshold

        query_embedding = await self.query_embedder.aembed_query(query)

        distance_expr = ResearchChunk.embedding.cosine_distance(query_embedding)
        similarity_expr = 1.0 - distance_expr

        stmt = (
            select(
                ResearchChunk,
                similarity_expr.label("similarity_score"),
            )
            .order_by(distance_expr)
            .limit(top_k)
        )

        if job_id:
            stmt = stmt.where(ResearchChunk.job_id == job_id)

        if scopes:
            scope_filter = ResearchChunk.meta["scope"].astext
            stmt = stmt.where(scope_filter.in_(scopes))
        elif scope:
            stmt = stmt.where(ResearchChunk.meta["scope"].astext == scope)

        result = await self.db.execute(stmt)
        rows = result.all()

        filtered = []
        for row in rows:
            chunk = row[0]
            score = float(row[1])
            if score >= similarity_threshold:
                filtered.append(
                    {
                        "id": str(chunk.id),
                        "content": chunk.content,
                        "meta": chunk.meta,
                        "job_id": str(chunk.job_id),
                        "similarity_score": score,
                    }
                )

        if len(filtered) < len(rows):
            logger.warning(
                f"Threshold {similarity_threshold} filtered {len(rows) - len(filtered)}/{len(rows)} results"
            )

        return filtered


def make_semantic_search_tool(
    vector_store: ContentFactoryVectorStore,
) -> Tool:
    """Create a Tool wrapping ``ContentFactoryVectorStore.semantic_search``.

    The returned ``Tool`` can be injected as a DI tool into agents that
    need to perform semantic retrieval (e.g. ``RedTeamAgent``).
    """

    async def _search(
        query: str,
        job_id: Optional[UUID] = None,
        scope: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return await vector_store.semantic_search(
            query=query,
            job_id=job_id,
            scope=scope,
            scopes=scopes,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    return Tool(
        name="semantic_search",
        description="Search the pgvector HNSW index for semantically similar chunks.",
        callable=_search,
        permissions={"RedTeamAgent", "*"},
        llm_schema={
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": (
                    "Search the vector store for semantically similar chunks "
                    "to the given query text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The text to search semantically",
                        },
                        "job_id": {
                            "type": "string",
                            "description": "Filter to chunks from a specific job",
                        },
                        "scope": {
                            "type": "string",
                            "description": "Single scope filter (e.g. RAW-CONTEXT, LOCAL)",
                        },
                        "scopes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multi-scope filter, takes precedence over scope",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    )
