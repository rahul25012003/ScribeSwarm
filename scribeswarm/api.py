"""FastAPI server: run management, HITL plan approval, live SSE timeline,
report viewing, checkpoint replay.

Runs execute in daemon threads; every node-completion event is appended to
an in-memory event log per thread (survives UI reconnects within the server
process) and fanned out to SSE subscribers via per-subscriber queues.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from scribeswarm.config import settings
from scribeswarm.graph.build import (
    default_budget, get_history, get_state, list_threads, stream_resume,
    stream_run,
)
from scribeswarm.output import run_dir, write_run_outputs

app = FastAPI(title="ScribeSwarm", version="0.5")

STATIC_DIR = Path(__file__).parent / "static"


class _Run:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.subscribers: list[queue.Queue] = []
        self.status = "starting"
        self.lock = threading.Lock()

    def emit(self, event: dict) -> None:
        event["ts"] = round(time.time(), 3)
        with self.lock:
            self.events.append(event)
            for q in self.subscribers:
                q.put(event)

    def subscribe(self) -> tuple[list[dict], queue.Queue]:
        q: queue.Queue = queue.Queue()
        with self.lock:
            return list(self.events), self.subscribers.append(q) or q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


RUNS: dict[str, _Run] = {}


def _summarize(node: str, update: dict | None) -> dict:
    update = update or {}
    out = {"node": node}
    if "plan" in update:
        out["sections"] = len((update["plan"] or {}).get("sections", []))
    if "claims" in update:
        out["claims"] = len(update["claims"] or [])
    if "verdicts" in update:
        out["verdicts"] = [v["status"] for v in update["verdicts"] or []]
    if "drafts" in update:
        out["drafts"] = len(update["drafts"] or [])
        out["revised"] = sum(1 for d in update["drafts"] or [] if d.get("revised"))
    if "charts" in update:
        out["charts"] = len(update["charts"] or [])
    if "critiques" in update:
        out["revise_requests"] = sum(
            1 for c in update["critiques"] or [] if c["verdict"] == "revise")
    for k in ("llm_calls", "tool_calls"):
        if update.get(k):
            out[k] = update[k]
    if update.get("budget_events"):
        out["notes"] = update["budget_events"]
    return out


def _drive(tid: str, generator) -> None:
    run = RUNS[tid]
    try:
        for chunk in generator:
            if "__interrupt__" in chunk:
                continue
            for node, update in chunk.items():
                run.emit(_summarize(node, update))
        snap = get_state(tid)
        if snap.next:                     # paused at the plan-approval gate
            run.status = "awaiting_approval"
            run.emit({"node": "__gate__",
                      "message": "plan ready — edit or approve to continue"})
        else:
            run.status = "finished"
            write_run_outputs(snap.values, tid)
            run.emit({"node": "__done__",
                      "llm_calls": snap.values.get("llm_calls", 0),
                      "tool_calls": snap.values.get("tool_calls", 0),
                      "claims": len(snap.values.get("claims", []))})
    except Exception as exc:              # surfaced to the timeline, not lost
        run.status = "error"
        run.emit({"node": "__error__", "message": str(exc)[:500]})


class NewRun(BaseModel):
    question: str
    hitl: bool = True
    verify: bool = True
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_wall_seconds: float | None = None


class PlanApproval(BaseModel):
    plan: dict | None = None    # edited plan, or None to approve as-is


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "runs_in_memory": len(RUNS)}


@app.get("/runs")
def runs() -> list[dict]:
    known = {tid: {"thread_id": tid, "status": run.status}
             for tid, run in RUNS.items()}
    for tid in list_threads():
        if tid not in known:
            snap = get_state(tid)
            known[tid] = {
                "thread_id": tid,
                "status": ("finished" if snap.values.get("report")
                           else "awaiting_approval" if snap.next
                           else "incomplete"),
            }
    return sorted(known.values(), key=lambda r: r["thread_id"], reverse=True)


@app.post("/runs")
def create_run(body: NewRun) -> dict:
    settings.verify_claims = body.verify
    tid = time.strftime("run_%Y%m%d_%H%M%S")
    if tid in RUNS:
        tid = f"{tid}_{len(RUNS)}"
    budget = default_budget()
    for key in ("max_llm_calls", "max_tool_calls", "max_wall_seconds"):
        if getattr(body, key) is not None:
            budget[key] = getattr(body, key)
    RUNS[tid] = _Run()
    RUNS[tid].status = "running"
    gen = stream_run(body.question, tid, budget=budget, hitl=body.hitl)
    threading.Thread(target=_drive, args=(tid, gen), daemon=True).start()
    return {"thread_id": tid, "hitl": body.hitl}


@app.post("/runs/{tid}/approve")
def approve(tid: str, body: PlanApproval) -> dict:
    run = RUNS.get(tid)
    snap = get_state(tid)
    if not snap.values:
        raise HTTPException(404, "unknown thread")
    if not snap.next:
        raise HTTPException(409, "run is not awaiting approval")
    if run is None:
        run = RUNS[tid] = _Run()
    run.status = "running"
    run.emit({"node": "__approved__",
              "edited": body.plan is not None})
    gen = stream_resume(tid, plan=body.plan)
    threading.Thread(target=_drive, args=(tid, gen), daemon=True).start()
    return {"thread_id": tid, "resumed": True}


@app.get("/runs/{tid}/state")
def state(tid: str) -> dict:
    snap = get_state(tid)
    if not snap.values:
        raise HTTPException(404, "unknown thread")
    v = snap.values
    run = RUNS.get(tid)
    return {
        "thread_id": tid,
        "status": (run.status if run else
                   "finished" if v.get("report")
                   else "awaiting_approval" if snap.next else "incomplete"),
        "next": list(snap.next or ()),
        "question": v.get("question", ""),
        "provider": v.get("provider", ""),
        "plan": v.get("plan"),
        "claims": v.get("claims", []),
        "verdicts": v.get("verdicts", []),
        "drafts": v.get("drafts", []),
        "charts": v.get("charts", []),
        "critiques": v.get("critiques", []),
        "budget_events": v.get("budget_events", []),
        "llm_calls": v.get("llm_calls", 0),
        "tool_calls": v.get("tool_calls", 0),
        "has_report": bool(v.get("report")),
    }


@app.get("/runs/{tid}/eventlog")
def eventlog(tid: str) -> list[dict]:
    run = RUNS.get(tid)
    return run.events if run else []


@app.get("/runs/{tid}/events")
def events(tid: str) -> StreamingResponse:
    run = RUNS.get(tid)
    if run is None:
        raise HTTPException(404, "no live event stream for this thread")

    def gen():
        backlog, q = run.subscribe()
        try:
            for e in backlog:
                yield f"data: {json.dumps(e)}\n\n"
            while run.status in ("starting", "running"):
                try:
                    e = q.get(timeout=2.0)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(e)}\n\n"
            yield f"data: {json.dumps({'node': '__stream_end__', 'status': run.status})}\n\n"
        finally:
            run.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/runs/{tid}/history")
def history(tid: str) -> list[dict]:
    return get_history(tid)


@app.get("/runs/{tid}/report")
def report(tid: str):
    path = run_dir(tid) / "report.html"
    if not path.exists():
        snap = get_state(tid)
        if snap.values.get("report"):
            write_run_outputs(snap.values, tid)   # render on demand
        else:
            raise HTTPException(404, "no report yet")
    return FileResponse(path, media_type="text/html")


@app.get("/runs/{tid}/report.md")
def report_md(tid: str):
    path = run_dir(tid) / "report.md"
    if not path.exists():
        raise HTTPException(404, "no report yet")
    return FileResponse(path, media_type="text/markdown; charset=utf-8")
