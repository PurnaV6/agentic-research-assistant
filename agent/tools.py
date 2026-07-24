"""Tools the agent can call: web search (no API key needed) and page fetching.

Both functions return plain dicts/strings (never raise) so a bad URL or an
empty search result becomes an observation the LLM can react to, instead of
crashing the agent loop.
"""
from __future__ import annotations

from duckduckgo_search import DDGS
import trafilatura

from agent.config import MAX_PAGE_CHARS, MAX_SEARCH_RESULTS


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """Search the web and return a list of {title, url, snippet} results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # network hiccups, rate limiting, etc.
        return [{"error": f"web_search failed: {exc}"}]

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "snippet": r.get("body", r.get("snippet", "")),
        }
        for r in results
    ]


def fetch_page(url: str, max_chars: int = MAX_PAGE_CHARS) -> dict:
    """Fetch a URL and extract its main readable text (strips nav/ads/boilerplate)."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"url": url, "error": "could not download page"}
        text = trafilatura.extract(downloaded)
        if not text:
            return {"url": url, "error": "could not extract readable content"}
        return {"url": url, "text": text[:max_chars]}
    except Exception as exc:
        return {"url": url, "error": f"fetch_page failed: {exc}"}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query and return titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch a specific URL (e.g. one returned by web_search) and return its "
                "main readable text content, for closer reading before citing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "web_search": web_search,
    "fetch_page": fetch_page,
}
