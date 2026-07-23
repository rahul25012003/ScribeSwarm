"""Phase 0 tests — full graph on the mock provider, zero network."""

import pytest

from scribeswarm.config import settings
from scribeswarm.graph.build import get_state, run
from scribeswarm.llm.client import extract_json
from scribeswarm.schemas import Claim, ResearchPlan

QUESTION = "Compare the EV battery supply chains of India, China, and the EU"


class EmptySearch:
    """No results — forces the model-prior fallback path (Phase 0 behavior)."""

    def search(self, query, max_results=5):
        return []


class NoFetch:
    def fetch(self, url):
        return None


@pytest.fixture(autouse=True)
def mock_provider(monkeypatch):
    from scribeswarm.graph import nodes
    from scribeswarm.tools import Toolset

    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(
        nodes, "get_toolset",
        lambda: Toolset(search=EmptySearch(), fetcher=NoFetch()))


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "ckpt.sqlite"


def test_full_linear_run(db):
    state = run(QUESTION, "t1", db_path=db)

    plan = ResearchPlan.model_validate(state["plan"])
    assert 1 <= len(plan.sections) <= settings.max_sections

    claims = [Claim.model_validate(c) for c in state["claims"]]
    assert claims and all(c.source_type == "model_prior" for c in claims)
    covered = {c.section_id for c in claims}
    assert covered == {s.id for s in plan.sections}

    assert state["provider"] == "mock"
    # planner 1 + per section: research fallback + writer + critic
    # (analyst is silent — model_prior claims are never charted)
    assert state["llm_calls"] == 1 + 3 * len(plan.sections)

    report = state["report"]
    assert report.startswith(f"# {plan.title}")
    assert "Provenance note" in report          # honesty header
    assert "## Claim ledger" in report
    for s in plan.sections:
        assert f"## {s.heading}" in report


def test_writer_cites_only_ledger_claims(db):
    state = run(QUESTION, "t2", db_path=db)
    valid = {c["id"] for c in state["claims"]}
    for d in state["drafts"]:
        assert d["cited_claim_ids"], "draft cites nothing"
        assert set(d["cited_claim_ids"]) <= valid


def test_checkpoint_persists_across_instances(db):
    run(QUESTION, "t3", db_path=db)
    # a fresh graph over the same db file (simulates process restart)
    snap = get_state("t3", db_path=db)
    assert snap.values["report"]
    assert not snap.next  # finished — nothing left to execute


def test_finished_thread_not_rerun(db):
    first = run(QUESTION, "t4", db_path=db)
    again = run("ignored — thread is finished", "t4", db_path=db)
    assert again["report"] == first["report"]
    assert again["question"] == QUESTION


def test_mock_determinism(db):
    a = run(QUESTION, "t5", db_path=db)
    b = run(QUESTION, "t6", db_path=db)
    assert a["report"] == b["report"]


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: [1, 2]') == [1, 2]
    with pytest.raises(ValueError):
        extract_json("no json at all")
