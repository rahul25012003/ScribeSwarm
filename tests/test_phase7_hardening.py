"""Phase 7 hardening tests — LLM-outage degradation, all offline.

The first live-Gemini shakedown (2026-07-23) crashed the graph: a 429 that
survived the backoff tiers escaped research_section as an exception. These
tests pin the fix — a hard LLM failure degrades the branch, never the run.
"""

import time

import pytest

from scribeswarm.config import settings
from scribeswarm.graph import nodes
from scribeswarm.graph.build import run
from scribeswarm.llm import client
from scribeswarm.schemas import Claim, PlanSection
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
    monkeypatch.setattr(settings, "verify_claims", False)
    monkeypatch.setattr(
        nodes, "get_toolset",
        lambda: Toolset(search=FakeSearch(), fetcher=FakeFetcher()))


def _budget(**over):
    base = {"max_llm_calls": 60, "max_tool_calls": 80,
            "max_wall_seconds": 0, "started_at": time.time()}
    base.update(over)
    return base


def _dying(after: int):
    """A generate() that answers `after` calls, then hard-fails forever."""
    calls = []

    def fake(system, user):
        calls.append(1)
        if len(calls) > after:
            raise RuntimeError("simulated 429 past all backoff tiers")
        return client._mock(system, user)

    return fake


def test_llm_death_mid_run_still_produces_report(tmp_path, monkeypatch):
    # planner succeeds, every later call dies — the run must still finish
    monkeypatch.setattr(nodes, "generate", _dying(after=1))
    state = run(QUESTION, "hard1", db_path=tmp_path / "c.sqlite",
                budget=_budget())
    assert state["report"].startswith("# ")
    assert state["claims"] == []          # extraction + fallback both died
    joined = "\n".join(state["budget_events"])
    assert "extractor LLM failed" in joined
    assert "writer LLM failed" in joined
    assert "Run notes (budget governor)" in state["report"]


def test_total_llm_outage_degrades_to_flat_plan_report(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "generate", _dying(after=0))
    state = run(QUESTION, "hard2", db_path=tmp_path / "c.sqlite",
                budget=_budget())
    assert state["report"].startswith("# ")
    assert len(state["plan"]["sections"]) == 1        # fallback flat plan
    assert state["plan"]["sections"][0]["heading"] == "Findings"
    assert any("planner degraded" in e for e in state["budget_events"])
    assert "No citable claims" in state["report"]


def test_writer_fallback_lists_citable_claims(monkeypatch):
    monkeypatch.setattr(nodes, "generate", _dying(after=0))
    section = PlanSection(id="s1", heading="H", sub_questions=["q"])
    citable = [
        Claim(id="s1c1", section_id="s1", text="Alpha fact.",
              source_type="web", source_url="https://a", quote="Alpha",
              status="corroborated"),
        Claim(id="s1c2", section_id="s1", text="Beta fact.",
              source_type="web", source_url="https://b", quote="Beta",
              status="contested"),
    ]
    draft, failed = nodes._write_section(section, citable)
    assert failed
    assert "[s1c1]" in draft.markdown and "[s1c2]" in draft.markdown
    assert "disputed" in draft.markdown          # contested stays flagged
    assert set(draft.cited_claim_ids) == {"s1c1", "s1c2"}


def test_verifier_outage_leaves_claim_citable(monkeypatch):
    monkeypatch.setattr(nodes, "generate", _dying(after=0))
    claim = Claim(id="s1c1", section_id="s1", text="India plans forty",
                  source_type="web", source_url="https://original.example/x",
                  quote="India plans", status="unverified")
    out = nodes.fact_check_section({
        "section_id": "s1", "claims": [claim.model_dump()],
        "budget": _budget(), "n_sections": 1,
        "llm_spent": 0, "tool_spent": 0,
    })
    assert out["verdicts"] == []                 # no verdict: stays citable
    assert any("left unverified" in e for e in out["budget_events"])


def test_reviser_failure_keeps_original_draft(monkeypatch):
    monkeypatch.setattr(nodes, "generate", _dying(after=0))
    state = {
        "plan": {"title": "T", "sections": [
            {"id": "s1", "heading": "H", "sub_questions": ["q"]}]},
        "claims": [Claim(id="s1c1", section_id="s1", text="Alpha fact.",
                         source_type="web", source_url="https://a",
                         quote="Alpha", status="unverified").model_dump()],
        "verdicts": [],
        "drafts": [{"section_id": "s1", "heading": "H",
                    "markdown": "Original prose [s1c1].",
                    "cited_claim_ids": ["s1c1"], "revised": False}],
        "critiques": [{"section_id": "s1", "verdict": "revise",
                       "notes": "tighten", "scores": {}}],
    }
    out = nodes.reviser(state)
    assert out["drafts"][0]["markdown"] == "Original prose [s1c1]."
    assert any("original draft kept" in e for e in out["budget_events"])


def test_pace_enforces_min_interval(monkeypatch):
    monkeypatch.setattr(settings, "llm_min_interval", 0.2)
    monkeypatch.setattr(client, "_last_real_call", 0.0)
    start = time.monotonic()
    client._pace()
    client._pace()
    # generous floor: Windows sleep granularity is ~15ms
    assert time.monotonic() - start >= 0.15


def test_mock_provider_is_never_paced(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_min_interval", 30.0)
    start = time.monotonic()
    client.generate("You are the PLANNER of a research-report pipeline.", "q")
    assert time.monotonic() - start < 1.0
