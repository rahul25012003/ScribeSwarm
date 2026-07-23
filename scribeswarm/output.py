"""Run artifacts on disk — shared by the CLI and the API server.

data/runs/<thread_id>/
    report.md      assembled markdown
    report.html    self-contained HTML (charts inlined base64)
    charts/*.png   rendered chart specs
"""

from __future__ import annotations

from pathlib import Path

from scribeswarm.config import settings


def run_dir(thread_id: str) -> Path:
    d = Path(settings.runs_dir) / thread_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_outputs(state: dict, thread_id: str) -> Path:
    from scribeswarm.charts import render_chart, render_html

    d = run_dir(thread_id)
    report_md = state.get("report", "")
    (d / "report.md").write_text(report_md, encoding="utf-8")
    for spec in state.get("charts", []):
        render_chart(spec, d / "charts" / f"{spec['id']}.png")
    title = (state.get("plan") or {}).get("title", "ScribeSwarm report")
    render_html(report_md, d / "charts", d / "report.html", title)
    return d
