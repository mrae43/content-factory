import logging
import os

import httpx
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


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

    async def extract(self, urls: list[str]) -> list[dict]:
        """Extract content from specific URLs via Tavily extract API.

        Returns list of {url, content} dicts for successfully extracted pages.
        """
        if not urls:
            return []
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set, skipping URL extraction")
            return []
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    TAVILY_EXTRACT_URL,
                    json={"urls": urls, "api_key": api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                logger.info(
                    f"Tavily extract returned {len(results)} results for {len(urls)} URLs"
                )
                return results
        except Exception:
            logger.warning(f"Tavily extract failed for {len(urls)} URLs", exc_info=True)
            return []
