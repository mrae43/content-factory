import logging
import os
from typing import Any, Dict, List

import httpx
from langchain_tavily import TavilySearch

from app.services.tools import Tool

logger = logging.getLogger(__name__)

TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


_tavily_service: "TavilySearchService | None" = None


def get_tavily_service() -> "TavilySearchService":
    global _tavily_service
    if _tavily_service is None:
        _tavily_service = TavilySearchService()
    return _tavily_service


class TavilySearchService:
    def __init__(self):
        self.client = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="basic",
        )

    async def search(self, query: str, search_depth: str = "basic") -> list[dict]:
        try:
            invoke_kwargs = {"query": query, "search_depth": search_depth}
            response = await self.client.ainvoke(invoke_kwargs)
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


def make_execute_web_search_tool() -> Tool:
    """Create a Tool wrapping TavilySearchService.search.

    The returned ``Tool`` exposes ``llm_schema`` for ``bind_tools`` so
    agents can let the LLM decide to call web search at runtime.
    """
    svc = get_tavily_service()

    async def _search(query: str) -> List[Dict[str, Any]]:
        return await svc.search(query)

    return Tool(
        name="execute_web_search",
        description=(
            "Search the web for current information on a given query. "
            "Returns a list of result dicts with url, content, title, and score."
        ),
        callable=_search,
        permissions={"RedTeamAgent", "*"},
        llm_schema={
            "type": "function",
            "function": {
                "name": "execute_web_search",
                "description": (
                    "Search the web for current information. "
                    "Returns results with url, content, title, and score."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    )
