"""Phase 4 tests — analyst charts, critic/revision loop, HTML export."""

import json

import pytest

from scribeswarm.charts import render_chart, render_html, validate_spec
from scribeswarm.config import settings
from scribeswarm.graph import nodes
from scribeswarm.graph.build import run
from scribeswarm.tools import Toolset
from scribeswarm.tools.fetch import FetchedPage
from scribeswarm.tools.search import SearchResult

QUESTION = "Compare the EV battery supply chains of India, China, and the EU"
PAGE_TEXT = ("India plans 40 gigawatt hours of cell capacity by 2030. "
             "China holds 70 percent of refining. The EU targets 90 percent "
             "local sourcing.")


class FakeSearch:
    def __init__(self):
        self.n = 0

    def search(self, query, max_results=5):
        self.n += 1
        return [SearchResult(url=f"https://example.org/p{self.n}",
                             title="T", snippet="s")]


class FakeFetcher:
    def fetch(self, url):
        return FetchedPage(url=url, title="T", text=PAGE_TEXT,
                           content_type="html")


@pytest.fixture(autouse=True)
def wired(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "verify_claims", False)  # focus on Ph4
    monkeypatch.setattr(
        nodes, "get_toolset",
        lambda: Toolset(search=FakeSearch(), fetcher=FakeFetcher()))


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "ckpt.sqlite"


def test_analyst_produces_validated_chart_specs(db):
    state = run(QUESTION, "t1", db_path=db)
    charts = state["charts"]
    assert charts, "numeric claims present but no chart"
    valid_ids = {c["id"] for c in state["claims"]}
    for ch in charts:
        assert validate_spec(ch, valid_ids) is None
        assert ch["id"].startswith("ch")
    assert f"charts/{charts[0]['id']}.png" in state["report"]
    assert "data from claims:" in state["report"]


def test_invalid_chart_specs_are_dropped_with_reason(db, monkeypatch):
    from scribeswarm.llm import client

    def bad_specs(user):
        return json.dumps([
            {"section_id": "s1", "title": "Fabricated", "kind": "bar",
             "x": ["a"], "y": [1.0], "claim_ids": ["not_a_claim"]},
            {"section_id": "s1", "title": "Mismatch", "kind": "line",
             "x": ["a", "b"], "y": [1.0], "claim_ids": []},
        ])

    monkeypatch.setattr(client, "_mock_analyst", bad_specs)
    state = run(QUESTION, "t2", db_path=db)
    assert state["charts"] == []
    assert any("chart spec dropped" in e for e in state["budget_events"])
    assert ".png" not in state["report"]


def test_critic_accepts_cited_drafts_no_revision(db):
    state = run(QUESTION, "t3", db_path=db)
    assert state["critiques"]
    assert all(c["verdict"] == "accept" for c in state["critiques"])
    assert all(not d["revised"] for d in state["drafts"])


def test_uncited_draft_gets_exactly_one_revision(db, monkeypatch):
    from scribeswarm.llm import client

    calls = {"n": 0}
    real_draft = client._mock_draft

    def uncited_first(user):
        calls["n"] += 1
        if "reviewer rejected" in user:
            return real_draft(user)          # revision writes citations
        return "Prose with no citations at all."

    monkeypatch.setattr(client, "_mock_draft", uncited_first)
    state = run(QUESTION, "t4", db_path=db)
    assert all(c["verdict"] == "revise" for c in state["critiques"])
    assert all(d["revised"] for d in state["drafts"])
    assert all(d["cited_claim_ids"] for d in state["drafts"])
    assert any("revised once after critic review" in e
               for e in state["budget_events"])
    # one initial + one revision per section — never a loop
    n = len(state["plan"]["sections"])
    assert calls["n"] == 2 * n


def test_chart_renders_png_and_html_inlines_it(tmp_path):
    spec = {"id": "ch1", "section_id": "s1", "title": "T", "kind": "bar",
            "x": ["a", "b"], "y": [1.0, 2.0], "claim_ids": ["s1c1"]}
    png = render_chart(spec, tmp_path / "charts" / "ch1.png")
    assert png.exists() and png.stat().st_size > 1000
    md = "# R\n\n![T](charts/ch1.png)\n"
    out = render_html(md, tmp_path / "charts", tmp_path / "report.html", "R")
    html = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "charts/ch1.png" not in html


def test_critic_disabled_skips_review(db, monkeypatch):
    monkeypatch.setattr(settings, "critic_enabled", False)
    state = run(QUESTION, "t5", db_path=db)
    assert state["critiques"] == []
    assert state["report"]
