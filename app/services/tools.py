from langchain_core.tools import tool
from uuid import UUID

from app.services.vector_store import ContentFactoryVectorStore


def make_search_database_tool(vector_store: ContentFactoryVectorStore, job_id: UUID):
    @tool
    async def search_database(
        query: str,
        scope: str = "RAW-CONTEXT,LOCAL",
        top_k: int = 5,
    ) -> str:
        """Search the research database for evidence relevant to a specific claim.
        Use this to find supporting or contradicting evidence for factual claims.
        Always search for specific claims, not the entire script."""
        scopes = [s.strip() for s in scope.split(",")]
        results = await vector_store.semantic_search(
            query=query,
            job_id=job_id,
            scopes=scopes,
            top_k=top_k,
        )
        if not results:
            return "No results found for this query."
        return "\n\n".join(
            f"[Source {i + 1}] (similarity: {r['similarity_score']:.2f})\n{r['content']}"
            for i, r in enumerate(results)
        )

    return search_database
