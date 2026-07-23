"""ScribeSwarm CLI.

  python cli.py run "Compare X and Y" [--thread t1]   # new run (or resume t1)
  python cli.py state t1                              # checkpoint summary
  python cli.py report t1                             # print saved report
"""

from __future__ import annotations

import time
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _runs_dir() -> Path:
    from scribeswarm.config import settings

    d = Path(settings.runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.command()
def run(question: str,
        thread: str = typer.Option(None, help="Thread id (reuse to resume)"),
        max_llm: int = typer.Option(None, help="Cap on LLM calls this run"),
        max_tools: int = typer.Option(None, help="Cap on tool calls this run"),
        max_seconds: float = typer.Option(None, help="Wall-time cap (0 = off)")):
    from scribeswarm.graph.build import default_budget, run as run_graph

    budget = default_budget()
    if max_llm is not None:
        budget["max_llm_calls"] = max_llm
    if max_tools is not None:
        budget["max_tool_calls"] = max_tools
    if max_seconds is not None:
        budget["max_wall_seconds"] = max_seconds

    thread_id = thread or time.strftime("run_%Y%m%d_%H%M%S")
    typer.echo(f"thread: {thread_id}")
    t0 = time.perf_counter()
    state = run_graph(question, thread_id, budget=budget)
    elapsed = time.perf_counter() - t0

    out = _runs_dir() / f"{thread_id}.md"
    out.write_text(state.get("report", ""), encoding="utf-8")
    typer.echo(
        f"done in {elapsed:.1f}s — provider={state.get('provider')} "
        f"llm_calls={state.get('llm_calls')} tool_calls={state.get('tool_calls', 0)} "
        f"claims={len(state.get('claims', []))}"
    )
    for event in state.get("budget_events", []):
        typer.echo(f"  budget: {event}")
    typer.echo(f"report: {out}")


@app.command()
def state(thread: str):
    from scribeswarm.graph.build import get_state

    snap = get_state(thread)
    if not snap or not snap.values:
        typer.echo("no checkpoints for this thread")
        raise typer.Exit(1)
    values = snap.values
    typer.echo(f"next node: {snap.next or '(finished)'}")
    for key in ("question", "provider", "llm_calls"):
        if key in values:
            typer.echo(f"{key}: {values[key]}")
    typer.echo(f"plan: {'yes' if values.get('plan') else 'no'} · "
               f"claims: {len(values.get('claims', []))} · "
               f"report: {'yes' if values.get('report') else 'no'}")


@app.command()
def report(thread: str):
    path = _runs_dir() / f"{thread}.md"
    if not path.exists():
        typer.echo("no saved report for this thread")
        raise typer.Exit(1)
    typer.echo(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
