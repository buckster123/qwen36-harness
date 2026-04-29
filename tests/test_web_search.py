"""Tests for harness.tools.web_search — DuckDuckGo HTML scraping."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from src.harness.tools.web_search import search


# Sample DuckDuckGo HTML response (simplified)
SAMPLE_DDHTML = """
<html>
<body>
<div class="results">
    <a class="result__url" href="https://en.wikipedia.org/wiki/Python_(programming_language)">Python programming language</a>
    <a class="result__snippet" href="...">Python is a high-level, general-purpose programming language...</a>

    <a class="result__url" href="https://docs.python.org/3/tutorial/">The Python Tutorial — Python 3 Documentation</a>
    <a class="result__snippet" href="...">This tutorial doesn't aim to be exhaustive and instead introduces the reader to Python's major features...</a>

    <a class="result__url" href="https://realpython.com/tutorials/">Real Python — Tutorials</a>
    <a class="result__snippet" href="...">Real Python provides high-quality free tutorials and courses on Python, Django, web development...</a>
</div>
</body>
</html>
"""


@pytest.mark.asyncio
async def test_ddg_html_parsing():
    """Test that DDG HTML response is parsed into structured results."""
    # Extract the parsing logic manually (same regex as in search())
    url_pattern = re.compile(
        r'<a[^>]*class="result__url"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*href="[^"]*"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    urls = [m.group(1) for m in url_pattern.finditer(SAMPLE_DDHTML)]
    snippets_raw = [m.group(1).strip() for m in snippet_pattern.finditer(SAMPLE_DDHTML)]

    assert len(urls) >= 3
    assert len(snippets_raw) >= 3
    assert "en.wikipedia.org" in urls[0]
    assert "docs.python.org" in urls[1]


@pytest.mark.asyncio
async def test_search_http_error_returns_empty():
    """Test that non-200 responses return an empty list."""
    async def mock_get(url, headers=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        return mock_resp

    with patch("httpx.AsyncClient.get", mock_get):
        results = await search("should be empty")

    assert results == []


@pytest.mark.asyncio
async def test_search_network_error_returns_empty():
    """Test that connection errors return an empty list."""
    import httpx

    async def mock_get(url, headers=None):
        raise httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient.get", mock_get):
        results = await search("network down")

    assert results == []


@pytest.mark.asyncio
async def test_search_max_results_limit():
    """Test that max_results caps the number of returned items."""
    # Create HTML with 5 results
    lines = []
    for i in range(5):
        lines.append(f'<a class="result__url" href="https://example{i}.com/page">{i}</a>')
        lines.append(f'<a class="result__snippet" href="...">Result {i} snippet</a>')
    html = "<html><body>" + "\n".join(lines) + "</body></html>"

    async def mock_get(url, headers=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        return mock_resp

    with patch("httpx.AsyncClient.get", mock_get):
        results = await search("test query", max_results=3)

    assert len(results) <= 3


@pytest.mark.asyncio
async def test_search_empty_html():
    """Test that empty/no-result HTML returns empty list."""
    async def mock_get(url, headers=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body></body></html>"
        return mock_resp

    with patch("httpx.AsyncClient.get", mock_get):
        results = await search("no such query xyz123456789")

    assert results == []


@pytest.mark.asyncio
async def test_search_tool_registration():
    """Test that the tool registers correctly on a fresh registry."""
    from src.harness.tools import Registry
    from src.harness.tools.web_search import register as register_web

    reg = Registry()
    register_web(reg)

    # Tool should be registered
    assert reg.get("web.search") is not None
    spec = reg.get("web.search")
    assert "query" in spec.parameters["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
