"""URL fetching + main-text extraction.

HTML goes through trafilatura (boilerplate removal); PDFs through PyMuPDF.
Hard size/time caps; any failure returns None — a researcher losing one
source is normal operation, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

MAX_BYTES = 2_500_000
MAX_TEXT_CHARS = 40_000


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    content_type: str    # html | pdf


class Fetcher:
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def fetch(self, url: str) -> FetchedPage | None:
        try:
            resp = httpx.get(
                url, timeout=self.timeout, follow_redirects=True,
                headers={"User-Agent": "ScribeSwarm/0.1 (research agent)"},
            )
            resp.raise_for_status()
        except Exception:
            return None
        body = resp.content[:MAX_BYTES]
        ctype = resp.headers.get("content-type", "")
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            return self._pdf(url, body)
        return self._html(url, body)

    def _html(self, url: str, body: bytes) -> FetchedPage | None:
        import trafilatura

        html = body.decode("utf-8", errors="replace")
        text = trafilatura.extract(html, include_comments=False) or ""
        if not text.strip():
            return None
        meta = trafilatura.extract_metadata(html)
        title = (meta.title if meta and meta.title else url)[:200]
        return FetchedPage(url=url, title=title,
                           text=text[:MAX_TEXT_CHARS], content_type="html")

    def _pdf(self, url: str, body: bytes) -> FetchedPage | None:
        import fitz  # PyMuPDF

        try:
            with fitz.open(stream=body, filetype="pdf") as doc:
                text = "\n".join(page.get_text("text") for page in doc)
                title = (doc.metadata or {}).get("title") or url
        except Exception:
            return None
        if not text.strip():
            return None
        return FetchedPage(url=url, title=title[:200],
                           text=text[:MAX_TEXT_CHARS], content_type="pdf")
