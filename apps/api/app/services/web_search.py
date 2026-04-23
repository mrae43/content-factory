import logging

from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)


class TavilySearchService:
    def __init__(self):
        self.client = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="basic",
        )

    async def search(self, query: str) -> list[dict]:
        try:
            response = await self.client.ainvoke({"query": query})
            if isinstance(response, dict):
                results = response.get("results", [])
                logger.info(
                    f"Tavily returned {len(results)} results for query: {query}"
                )
                return results
            logger.warning(f"Unexpected Tavily response type for query: {query}")
            return []
        except Exception:
            logger.warning(f"Tavily search failed for query: {query}", exc_info=True)
            return []
