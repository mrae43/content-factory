import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.services.web_search import TavilySearchService, TAVILY_EXTRACT_URL


@pytest.mark.unit
class TestTavilyExtract:
    @pytest.fixture
    def service(self):
        return TavilySearchService()

    async def test_extract_returns_results_on_success(self, service):
        mock_response = {
            "results": [
                {"url": "https://example.com/article1", "content": "Article 1 content"},
                {"url": "https://example.com/article2", "content": "Article 2 content"},
            ]
        }

        with patch("app.services.web_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response_obj
            mock_client_cls.return_value = mock_client

            result = await service.extract(
                ["https://example.com/article1", "https://example.com/article2"]
            )

            assert len(result) == 2
            assert result[0]["url"] == "https://example.com/article1"
            assert result[0]["content"] == "Article 1 content"
            mock_client.post.assert_awaited_once_with(
                TAVILY_EXTRACT_URL,
                json={
                    "urls": [
                        "https://example.com/article1",
                        "https://example.com/article2",
                    ],
                    "api_key": "",
                },
            )

    async def test_extract_returns_empty_when_urls_empty(self, service):
        result = await service.extract([])
        assert result == []

    async def test_extract_returns_empty_when_no_api_key(self, service):
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
            result = await service.extract(["https://example.com"])
            assert result == []

    async def test_extract_returns_empty_on_exception(self, service):
        with patch("app.services.web_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.side_effect = httpx.ConnectError("Connection failed")
            mock_client_cls.return_value = mock_client

            result = await service.extract(["https://example.com"])

            assert result == []
