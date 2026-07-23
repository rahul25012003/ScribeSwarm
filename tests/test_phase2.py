"""Phase 2 tests — parallel fan-out merging + budget governor, all offline."""

import time

import pytest

from scribeswarm.config import settings
from scribeswarm.governor import allocate
from scribeswarm.graph import nodes
from scribeswarm.graph.build import run
from scribeswarm.schemas import Claim
from scribeswarm.tools import Toolset
from scribeswarm.tools.fetch import FetchedPage
from scribeswarm.tools.search import SearchResult

QUESTION = "Compare the EV battery supply chains of India, China, and the EU"
PAGE_TEXT = "India plans forty gigawatt hours of cell capacity by 2030."


class FakeSearch:
    def search(self, query, max_results=5):
        return [SearchResult(url=f"https://example.org/{abs(hash(query)) % 997}",
                             title="T", snippet="s")]


class FakeFetcher:
    def fetch(self, url):
        return FetchedPage(url=url, title="T", text=PAGE_TEXT,
                           content_type="html")


@pytest.fixture(autouse=True)
def wired(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "verify_claims", False)  # Ph3 tested separately
    monkeypatch.setattr(
        nodes, "get_toolset",
        lambda: Toolset(search=FakeSearch(), fetcher=FakeFetcher()))


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "ckpt.sqlite"


def _budget(**over):
    base = {"max_llm_calls": 60, "max_tool_calls": 80,
            "max_wall_seconds": 0, "started_at": time.time()}
    base.update(over)
    return base


def test_parallel_branches_merge_all_sections(db):
    state = run(QUESTION, "t1", db_path=db, budget=_budget())
    sections = {s["id"] for s in state["plan"]["sections"]}
    claims = [Claim.model_validate(c) for c in state["claims"]]
    assert {c.section_id for c in claims} == sections
    # section-scoped ids are unique across branches
    ids = [c.id for c in claims]
    assert len(ids) == len(set(ids))
    assert all(c.id.startswith(c.section_id) for c in claims)


def test_counters_sum_across_branches(db):
    state = run(QUESTION, "t2", db_path=db, budget=_budget())
    n = len(state["plan"]["sections"])
    # planner 1 + per section: extraction + writer + critic, + analyst 1
    assert state["llm_calls"] == 1 + 3 * n + 1
    # searches (1 sub-question per section in mock) + fetches (1/section)
    assert state["tool_calls"] == n + n


def test_deterministic_report_across_runs(db, tmp_path):
    a = run(QUESTION, "t3", db_path=db, budget=_budget())
    b = run(QUESTION, "t4", db_path=tmp_path / "other.sqlite",
            budget=_budget())
    assert a["report"] == b["report"]


def test_zero_tool_budget_degrades_to_model_prior(db):
    state = run(QUESTION, "t5", db_path=db,
                budget=_budget(max_tool_calls=0))
    claims = [Claim.model_validate(c) for c in state["claims"]]
    assert claims and all(c.source_type == "model_prior" for c in claims)
    assert state["tool_calls"] == 0
    assert state["budget_events"]
    assert "Run notes (budget governor)" in state["report"]


def test_expired_wall_clock_yields_report_not_crash(db):
    state = run(QUESTION, "t6", db_path=db,
                budget=_budget(max_wall_seconds=0.001,
                               started_at=time.time() - 60))
    assert state["report"].startswith("# ")       # report always assembles
    assert state["claims"] == []                  # no research happened
    assert any("budget exhausted" in e or "wall-time" in e
               for e in state["budget_events"])


def test_allocation_math():
    sb = allocate({"max_llm_calls": 21, "max_tool_calls": 40,
                   "max_wall_seconds": 0, "started_at": 0},
                  n_sections=4, llm_spent=1, tool_spent=0)
    # (21 - 1 spent - 4 writer reserve) // 4 = 4 ; 40 // 4 = 10
    assert sb.llm == 4 and sb.tool == 10 and sb.deadline == 0
    assert not sb.expired()
