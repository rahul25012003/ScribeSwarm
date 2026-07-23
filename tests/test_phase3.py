"""Phase 3 tests — adversarial fact-checking, all offline."""

import json

import pytest

from scribeswarm.config import settings
from scribeswarm.graph import nodes
from scribeswarm.graph.build import run
from scribeswarm.schemas import Claim
from scribeswarm.tools import Toolset
from scribeswarm.tools.fetch import FetchedPage
from scribeswarm.tools.search import SearchResult

QUESTION = "Compare the EV battery supply chains of India, China, and the EU"
PAGE_TEXT = "India plans forty gigawatt hours of cell capacity by 2030."


class FakeSearch:
    """Distinct URL per query -> verification always finds an independent one."""

    def __init__(self):
        self.n = 0

    def search(self, query, max_results=5):
        self.n += 1
        return [SearchResult(url=f"https://example.org/p{self.n}",
                             title="T", snippet="s")]


class SingleUrlSearch:
    """Every query returns ONLY the original source URL -> nothing independent."""

    def search(self, query, max_results=5):
        return [SearchResult(url="https://example.org/only",
                             title="T", snippet="s")]


class FakeFetcher:
    def fetch(self, url):
        return FetchedPage(url=url, title="T", text=PAGE_TEXT,
                           content_type="html")


@pytest.fixture(autouse=True)
def mocked(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "ckpt.sqlite"


def _wire(monkeypatch, search):
    monkeypatch.setattr(
        nodes, "get_toolset",
        lambda: Toolset(search=search, fetcher=FakeFetcher()))


def test_corroborated_flow(db, monkeypatch):
    _wire(monkeypatch, FakeSearch())
    state = run(QUESTION, "t1", db_path=db)
    verdicts = state["verdicts"]
    assert verdicts and all(v["status"] == "corroborated" for v in verdicts)
    assert "corroborated" in state["report"]
    assert "## Honesty appendix" not in state["report"]


def test_contested_claim_lands_in_honesty_appendix_with_counter(db, monkeypatch):
    _wire(monkeypatch, FakeSearch())
    from scribeswarm.llm import client

    def extract_disputed(user):
        return json.dumps([{"claim": "This disputed figure is exactly 42.",
                            "quote": PAGE_TEXT[:40], "confidence": 0.9}])

    monkeypatch.setattr(client, "_mock_extract", extract_disputed)
    state = run(QUESTION, "t2", db_path=db)
    assert all(v["status"] == "contested" for v in state["verdicts"])
    v = state["verdicts"][0]
    assert v["counter_url"] and v["counter_quote"]
    report = state["report"]
    assert "## Honesty appendix" in report
    assert "disputed by" in report
    # contested stays citable, rendered as disputed — writer saw the marker
    assert any(d["cited_claim_ids"] for d in state["drafts"])


def test_no_independent_source_means_unverifiable_and_uncited(db, monkeypatch):
    _wire(monkeypatch, SingleUrlSearch())
    state = run(QUESTION, "t3", db_path=db)
    assert state["verdicts"]
    assert all(v["status"] == "unverifiable" for v in state["verdicts"])
    assert all(v["reason"] == "no independent source found"
               for v in state["verdicts"])
    for d in state["drafts"]:
        assert d["cited_claim_ids"] == []
    assert "excluded from prose" in state["report"]


def test_no_verify_flag_skips_fact_checking(db, monkeypatch):
    _wire(monkeypatch, FakeSearch())
    monkeypatch.setattr(settings, "verify_claims", False)
    state = run(QUESTION, "t4", db_path=db)
    assert state.get("verdicts", []) == []
    claims = [Claim.model_validate(c) for c in state["claims"]]
    assert all(c.status == "unverified" for c in claims)


def test_verify_cap_per_section(db, monkeypatch):
    _wire(monkeypatch, FakeSearch())
    from scribeswarm.llm import client

    def extract_three(user):
        return json.dumps([
            {"claim": f"Claim number {i}.", "quote": PAGE_TEXT[:40],
             "confidence": 0.5 + i / 10}
            for i in range(3)
        ])

    monkeypatch.setattr(client, "_mock_extract", extract_three)
    monkeypatch.setattr(settings, "max_verify_per_section", 1)
    state = run(QUESTION, "t5", db_path=db)
    sections = len(state["plan"]["sections"])
    assert len(state["verdicts"]) == sections          # exactly 1 per section
    # the highest-confidence claim of each section was picked
    picked = {v["claim_id"] for v in state["verdicts"]}
    for sid in {c["section_id"] for c in state["claims"]}:
        best = max((c for c in state["claims"] if c["section_id"] == sid),
                   key=lambda c: c["confidence"])
        assert best["id"] in picked
