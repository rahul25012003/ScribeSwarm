"""Phase 5 tests — API server: HITL approval, events, history, report."""

import time

import pytest
from fastapi.testclient import TestClient

from scribeswarm import api
from scribeswarm.config import settings
from scribeswarm.graph import nodes
from scribeswarm.tools import Toolset
from scribeswarm.tools.fetch import FetchedPage
from scribeswarm.tools.search import SearchResult

QUESTION = "Compare the EV battery supply chains of India, China, and the EU"
PAGE_TEXT = "India plans 40 gigawatt hours of cell capacity by 2030."


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


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "ckpt.sqlite"))
    monkeypatch.setattr(settings, "runs_dir", str(tmp_path / "runs"))
    monkeypatch.setattr(
        nodes, "get_toolset",
        lambda: Toolset(search=FakeSearch(), fetcher=FakeFetcher()))
    api.RUNS.clear()
    return TestClient(api.app)


def _wait(client, tid, want, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/runs/{tid}/state")
        if st.status_code == 200 and st.json()["status"] == want:
            return st.json()
        time.sleep(0.05)
    raise AssertionError(f"thread {tid} never reached {want}")


def test_hitl_flow_edit_plan_then_finish(client):
    res = client.post("/runs", json={"question": QUESTION, "hitl": True,
                                     "verify": False})
    tid = res.json()["thread_id"]

    st = _wait(client, tid, "awaiting_approval")
    assert st["plan"] and st["plan"]["sections"]
    assert st["drafts"] == []            # nothing executed past the gate

    plan = st["plan"]
    plan["title"] = "EDITED TITLE FOR TEST"
    plan["sections"][0]["heading"] = "Edited Heading"
    res = client.post(f"/runs/{tid}/approve", json={"plan": plan})
    assert res.status_code == 200

    st = _wait(client, tid, "finished")
    assert st["has_report"]

    md = client.get(f"/runs/{tid}/report.md")
    assert md.status_code == 200
    assert "EDITED TITLE FOR TEST" in md.text
    assert "## Edited Heading" in md.text

    html = client.get(f"/runs/{tid}/report")
    assert html.status_code == 200
    assert "EDITED TITLE FOR TEST" in html.text

    log_nodes = [e["node"] for e in client.get(f"/runs/{tid}/eventlog").json()]
    assert "planner" in log_nodes and "__gate__" in log_nodes
    assert "__approved__" in log_nodes and "__done__" in log_nodes

    hist = client.get(f"/runs/{tid}/history").json()
    assert hist and any(h["has_report"] for h in hist)


def test_no_hitl_runs_straight_through(client):
    res = client.post("/runs", json={"question": QUESTION, "hitl": False,
                                     "verify": False})
    tid = res.json()["thread_id"]
    st = _wait(client, tid, "finished")
    assert st["has_report"] and st["claims"]
    assert tid in [r["thread_id"] for r in client.get("/runs").json()]


def test_approve_rejects_wrong_states(client):
    assert client.post("/runs/nope/approve", json={}).status_code == 404
    res = client.post("/runs", json={"question": QUESTION, "hitl": False,
                                     "verify": False})
    tid = res.json()["thread_id"]
    _wait(client, tid, "finished")
    assert client.post(f"/runs/{tid}/approve", json={}).status_code == 409
