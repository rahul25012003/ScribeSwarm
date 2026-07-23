"""Phase 6 tests — benchmark harness mechanics (mock provider, fake tools)."""

import json

import pytest

from evals import run_benchmark
from scribeswarm.config import settings
from scribeswarm.graph import nodes
from scribeswarm.tools import Toolset
from scribeswarm.tools.fetch import FetchedPage
from scribeswarm.tools.search import SearchResult

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


@pytest.fixture(autouse=True)
def wired(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "ckpt.sqlite"))
    monkeypatch.setattr(settings, "runs_dir", str(tmp_path / "runs"))
    monkeypatch.setattr(run_benchmark, "RESULTS_DIR", tmp_path / "results")
    toolset = Toolset(search=FakeSearch(), fetcher=FakeFetcher())
    monkeypatch.setattr(nodes, "get_toolset", lambda: toolset)


def test_benchmark_runs_and_summarizes(tmp_path):
    res = run_benchmark.run(["multi_verify", "single_section"], limit=2)
    rows = res["rows"]
    assert len(rows) == 4 and all(r["ok"] for r in rows)

    # structural metrics present and sane
    for r in rows:
        assert r["citation_validity"] in (None, 1.0)   # mock cites only ledger ids
        assert r["sourced_ratio"] == 1.0
        assert r["judge"]["coverage"] == 3

    # single_section config really ran with one section
    single = [r for r in rows if r["config"] == "single_section"]
    assert all(r["llm_calls"] < 10 for r in single)

    summary = res["summary"]
    assert set(summary) == {"multi_verify", "single_section"}
    assert summary["multi_verify"]["corroborated"] > 0
    assert summary["single_section"]["corroborated"] == 0   # verify off

    # artifacts written
    out = res["output"]
    lines = [json.loads(l) for l in
             open(out, encoding="utf-8").read().splitlines()]
    assert len(lines) == 4


def test_benchmark_resume_skips_done(tmp_path):
    first = run_benchmark.run(["single_section"], limit=1)
    prev = run_benchmark._load_previous(
        __import__("pathlib").Path(first["output"]))
    assert len(prev) == 1

    second = run_benchmark.run(["single_section"], limit=1,
                               resume=__import__("pathlib").Path(first["output"]))
    assert second["rows"][0]["thread_id"] == first["rows"][0]["thread_id"]
