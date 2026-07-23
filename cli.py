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
        thread: str = typer.Option(None, help="Thread id (reuse to resume)")):
    from scribeswarm.graph.build import run as run_graph

    thread_id = thread or time.strftime("run_%Y%m%d_%H%M%S")
    typer.echo(f"thread: {thread_id}")
    t0 = time.perf_counter()
    state = run_graph(question, thread_id)
    elapsed = time.perf_counter() - t0

    out = _runs_dir() / f"{thread_id}.md"
    out.write_text(state.get("report", ""), encoding="utf-8")
    typer.echo(
        f"done in {elapsed:.1f}s — provider={state.get('provider')} "
        f"llm_calls={state.get('llm_calls')} claims={len(state.get('claims', []))}"
    )
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
