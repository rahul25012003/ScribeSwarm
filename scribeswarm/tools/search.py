"""Web search adapters.

Resolution mirrors the LLM adapter: TAVILY_API_KEY set -> Tavily (better
snippets, stable API); otherwise DuckDuckGo via `ddgs` — keyless, fine for
dev, politely rate-limited so results are capped and failures degrade to
an empty list (the researcher just gets fewer sources, never crashes).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


class DdgsSearch:
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        from ddgs import DDGS

        try:
            with DDGS() as ddgs:
                rows = ddgs.text(query, max_results=max_results)
        except Exception:
            return []          # degraded, not dead — researcher continues
        return [
            SearchResult(
                url=r.get("href") or r.get("url") or "",
                title=r.get("title", ""),
                snippet=r.get("body", ""),
            )
            for r in rows or [] if (r.get("href") or r.get("url"))
        ]


class TavilySearch:
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": max_results,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return [
            SearchResult(url=r["url"], title=r.get("title", ""),
                         snippet=r.get("content", ""))
            for r in resp.json().get("results", [])
        ]


def resolve_search():
    if os.environ.get("TAVILY_API_KEY"):
        return TavilySearch()
    return DdgsSearch()
