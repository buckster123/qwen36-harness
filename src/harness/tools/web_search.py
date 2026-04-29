"""Web search via DuckDuckGo HTML (no API key needed).

Scrapes the client-side-rendered HTML page from html.duckduckgo.com and
parses out result URLs, titles, and snippets using simple regex extraction.
Zero external deps beyond what httpx already provides.

Usage from the model:
    web.search("what is quantum entanglement")
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from . import Registry, default_registry


# ---------------------------------------------------------------------------
# Core search logic
# ---------------------------------------------------------------------------


async def search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo and return structured results.

    Returns a list of {url, title, snippet} dicts, at most max_results.
    """
    url = f"https://html.duckduckgo.com/html/?q={query}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "harness-agent/1.0"})
            if resp.status_code != 200:
                return []
            body = resp.text
        except Exception:
            return []

    # DuckDuckGo HTML structure (client-rendered):
    # <a class="result__url" href="...">...</a>   — result URL + encoded title
    # <a class="result__snippet" href="...">...</a> — snippet text
    url_pattern = re.compile(
        r'<a[^>]*class="result__url"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*href="[^"]*"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    urls: list[str] = [m.group(1) for m in url_pattern.finditer(body)]
    snippets_raw: list[str] = [m.group(1).strip() for m in snippet_pattern.finditer(body)]

    results = []
    for i in range(min(max_results, len(urls))):
        raw_title = re.sub(r'<[^>]+>', '', urls[i].split("?", 1)[0]).strip()
        title = re.sub(r'^https?://', '', urls[i]).replace("/html", "").strip()
        results.append({
            "url": urls[i],
            "title": title or raw_title,
            "snippet": snippets_raw[i] if i < len(snippets_raw) else "",
        })

    return results


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register(registry: Registry = default_registry) -> None:
    """Register web.search on the registry."""

    @registry.tool(
        name="web.search",
        description=(
            "Search the web using DuckDuckGo. Returns a list of search results "
            "with URL, title, and snippet for each result.\n\n"
            "Parameters:\n"
            "- query: The search query string (required)\n"
            "- max_results: Max number of results to return (default 5)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 5, max 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )
    def do_search(
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        """Search the web for information."""
        import asyncio

        limit = min(max(1, max_results), 10)
        return asyncio.run(search(query, max_results=limit))
