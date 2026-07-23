"""Phase 1 tests — tool-driven research with quote grounding, all offline."""

import pytest

from scribeswarm.config import settings
from scribeswarm.graph import nodes
from scribeswarm.graph.build import run
from scribeswarm.schemas import Claim
from scribeswarm.tools import Toolset
from scribeswarm.tools.fetch import FetchedPage
from scribeswarm.tools.search import SearchResult

QUESTION = "Compare the EV battery supply chains of India, China, and the EU"

PAGE_TEXT = (
    "India plans forty gigawatt hours of cell manufacturing capacity by 2030.\n"
    "China currently controls most global battery refining capacity.\n"
    "The EU battery passport becomes mandatory in 2027."
)


class FakeSearch:
    def __init__(self):
        self.queries = []

    def search(self, query, max_results=5):
        self.queries.append(query)
        n = len(self.queries)
        return [
            SearchResult(url=f"https://example.org/a{n}", title="A", snippet="s"),
            SearchResult(url="https://example.org/shared", title="B", snippet="s"),
        ]


class FakeFetcher:
    def __init__(self):
        self.fetched = []

    def fetch(self, url):
        self.fetched.append(url)
        return FetchedPage(url=url, title="Fake page", text=PAGE_TEXT,
                           content_type="html")


@pytest.fixture(autouse=True)
def wired(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    search, fetcher = FakeSearch(), FakeFetcher()
    monkeypatch.setattr(
        nodes, "get_toolset", lambda: Toolset(search=search, fetcher=fetcher))
    return search, fetcher


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "ckpt.sqlite"


def test_sourced_run_produces_grounded_web_claims(db, wired):
    state = run(QUESTION, "t1", db_path=db)
    claims = [Claim.model_validate(c) for c in state["claims"]]
    assert claims
    for c in claims:
        assert c.source_type == "web"
        assert c.source_url.startswith("https://example.org/")
        assert c.status == "unverified"           # quote matched the page
        assert nodes.quote_in_text(c.quote, PAGE_TEXT)
    assert state["tool_calls"] > 0


def test_page_cap_respected(db, wired):
    _, fetcher = wired
    state = run(QUESTION, "t2", db_path=db)
    sections = len(state["plan"]["sections"])
    assert len(fetcher.fetched) <= sections * settings.pages_per_section


def test_report_has_sources_and_no_scare_note(db):
    state = run(QUESTION, "t3", db_path=db)
    report = state["report"]
    assert "## Sources" in report
    assert "https://example.org/" in report
    # fully sourced run -> no model-knowledge provenance warning
    assert "model knowledge only" not in report


def test_ungrounded_quote_is_flagged_and_never_cited(db, monkeypatch):
    from scribeswarm.llm import client

    def bad_extract(user):
        import json
        return json.dumps([{"claim": "Fabricated fact.",
                            "quote": "this sentence is not in the page",
                            "confidence": 0.9}])

    monkeypatch.setattr(client, "_mock_extract", bad_extract)
    state = run(QUESTION, "t4", db_path=db)
    claims = [Claim.model_validate(c) for c in state["claims"]]
    assert claims and all(c.status == "quote_not_found" for c in claims)
    for d in state["drafts"]:
        assert d["cited_claim_ids"] == []          # writer refused them all
    assert "quote_not_found" in state["report"]    # but the ledger shows them


def test_quote_matching_normalizes_whitespace_and_case():
    text = "India plans   Forty gigawatt\nhours of capacity."
    assert nodes.quote_in_text("forty GIGAWATT hours", text)
    assert not nodes.quote_in_text("fifty gigawatt hours", text)
    assert not nodes.quote_in_text("", text)


def test_search_failure_falls_back_to_model_prior(db, monkeypatch):
    class DeadSearch:
        def search(self, query, max_results=5):
            return []

    monkeypatch.setattr(
        nodes, "get_toolset",
        lambda: Toolset(search=DeadSearch(), fetcher=FakeFetcher()))
    state = run(QUESTION, "t5", db_path=db)
    claims = [Claim.model_validate(c) for c in state["claims"]]
    assert claims and all(c.source_type == "model_prior" for c in claims)
    assert "model knowledge only" in state["report"]
