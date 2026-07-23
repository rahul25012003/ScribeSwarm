"""Toolset used by researcher agents.

`get_toolset()` is the injection seam (same pattern as LexRAG's
pipeline.get_retriever): production resolves real search + fetch from
settings/env; tests monkeypatch it with fakes so the graph runs offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from scribeswarm.tools.fetch import Fetcher
from scribeswarm.tools.search import resolve_search


@dataclass
class Toolset:
    search: object   # .search(query, max_results) -> list[SearchResult]
    fetcher: object  # .fetch(url) -> FetchedPage | None

    def describe(self) -> list[dict]:
        """Registry listing (surfaces in logs now, in the UI in Phase 5)."""
        return [
            {"name": "web_search", "impl": type(self.search).__name__,
             "desc": "search the web, returns url/title/snippet"},
            {"name": "fetch_url", "impl": type(self.fetcher).__name__,
             "desc": "fetch a url, extract main text (HTML via trafilatura, PDF via PyMuPDF)"},
        ]


_toolset: Toolset | None = None


def get_toolset() -> Toolset:
    global _toolset
    if _toolset is None:
        _toolset = Toolset(search=resolve_search(), fetcher=Fetcher())
    return _toolset
