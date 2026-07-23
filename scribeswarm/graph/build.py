"""Graph wiring + checkpointed execution.

Phase 2 shape: planner -> Send fan-out (one research_section branch per
plan section, run in parallel) -> writer joins at the superstep barrier.
SQLite checkpointer from day 0 (ADR-001): every superstep is persisted, so
a killed process resumes with the same thread_id.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from scribeswarm.config import settings
from scribeswarm.graph import nodes
from scribeswarm.schemas import ResearchPlan, RunState


def _fan_out(state: RunState):
    plan = ResearchPlan.model_validate(state["plan"])
    return [
        Send("research_section", {
            "section": section.model_dump(),
            "plan_title": plan.title,
            "n_sections": len(plan.sections),
            "budget": state.get("budget", {}),
            "llm_spent": state.get("llm_calls", 0),
            "tool_spent": state.get("tool_calls", 0),
        })
        for section in plan.sections
    ]


def _fan_out_verify(state: RunState):
    """After research merges: fan out one fact-check branch per section that
    produced sourced claims — or skip straight to the writer."""
    if not settings.verify_claims:
        return "writer"
    by_section: dict[str, list[dict]] = {}
    for c in state.get("claims", []):
        if c["source_type"] != "model_prior" and c["status"] == "unverified":
            by_section.setdefault(c["section_id"], []).append(c)
    if not by_section:
        return "writer"
    return [
        Send("fact_check_section", {
            "section_id": sid,
            "claims": section_claims,
            "n_sections": len(by_section),
            "budget": state.get("budget", {}),
            "llm_spent": state.get("llm_calls", 0),
            "tool_spent": state.get("tool_calls", 0),
        })
        for sid, section_claims in by_section.items()
    ]


def build_graph(checkpointer: SqliteSaver, interrupt_after_plan: bool = False):
    builder = StateGraph(RunState)
    builder.add_node("planner", nodes.planner)
    builder.add_node("research_section", nodes.research_section)
    builder.add_node("verify_gate", nodes.verify_gate)
    builder.add_node("fact_check_section", nodes.fact_check_section)
    builder.add_node("writer", nodes.writer)
    builder.add_node("analyst", nodes.analyst)
    builder.add_node("critic", nodes.critic)
    builder.add_node("reviser", nodes.reviser)
    builder.add_node("assemble", nodes.assemble)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", _fan_out, ["research_section"])
    builder.add_edge("research_section", "verify_gate")
    builder.add_conditional_edges(
        "verify_gate", _fan_out_verify, ["fact_check_section", "writer"])
    builder.add_edge("fact_check_section", "writer")
    builder.add_edge("writer", "analyst")
    builder.add_edge("analyst", "critic")
    builder.add_edge("critic", "reviser")
    builder.add_edge("reviser", "assemble")
    builder.add_edge("assemble", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["planner"] if interrupt_after_plan else None,
    )


def open_checkpointer(db_path: str | Path | None = None) -> SqliteSaver:
    path = Path(db_path or settings.checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False is safe here: SqliteSaver serializes access
    # through its own internal lock.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def default_budget() -> dict:
    return {
        "max_llm_calls": settings.max_llm_calls,
        "max_tool_calls": settings.max_tool_calls,
        "max_wall_seconds": settings.max_wall_seconds,
        "started_at": time.time(),
    }


def run(question: str, thread_id: str, db_path: str | Path | None = None,
        budget: dict | None = None) -> RunState:
    """Execute (or resume) a run. If the thread already has checkpoints, the
    graph continues from the last completed superstep instead of restarting."""
    saver = open_checkpointer(db_path)
    graph = build_graph(saver)
    config = {"configurable": {"thread_id": thread_id}}
    existing = graph.get_state(config)
    if existing.values and not existing.next:
        return existing.values          # finished thread — immutable result
    payload: RunState | None = (
        None if existing.values
        else {"question": question, "budget": budget or default_budget()}
    )
    return graph.invoke(payload, config=config)


def get_state(thread_id: str, db_path: str | Path | None = None):
    graph = build_graph(open_checkpointer(db_path))
    return graph.get_state({"configurable": {"thread_id": thread_id}})


# --- Phase 5: streaming execution, HITL plan approval, history/replay ------

def stream_run(question: str, thread_id: str,
               db_path: str | Path | None = None, budget: dict | None = None,
               hitl: bool = False):
    """Start a run, yielding one {node: update} dict per completed node.
    With hitl=True execution pauses after the planner (plan approval gate)."""
    graph = build_graph(open_checkpointer(db_path), interrupt_after_plan=hitl)
    config = {"configurable": {"thread_id": thread_id}}
    payload: RunState = {"question": question,
                         "budget": budget or default_budget()}
    yield from graph.stream(payload, config=config, stream_mode="updates")


def stream_resume(thread_id: str, db_path: str | Path | None = None,
                  plan: dict | None = None):
    """Resume an interrupted (or killed) run; optionally with an edited plan
    applied first — the human-in-the-loop steering hook."""
    graph = build_graph(open_checkpointer(db_path))
    config = {"configurable": {"thread_id": thread_id}}
    if plan is not None:
        ResearchPlan.model_validate(plan)          # reject malformed edits
        graph.update_state(config, {"plan": plan})
    yield from graph.stream(None, config=config, stream_mode="updates")


def get_history(thread_id: str, db_path: str | Path | None = None) -> list[dict]:
    """Checkpoint history for the replay view — newest first."""
    graph = build_graph(open_checkpointer(db_path))
    config = {"configurable": {"thread_id": thread_id}}
    steps = []
    for snap in graph.get_state_history(config):
        meta = snap.metadata or {}
        writes = meta.get("writes") or {}
        steps.append({
            "checkpoint_id": (snap.config.get("configurable") or {}).get(
                "checkpoint_id", ""),
            "step": meta.get("step", -1),
            "nodes": sorted(writes.keys()),
            "next": list(snap.next or ()),
            "llm_calls": snap.values.get("llm_calls", 0),
            "tool_calls": snap.values.get("tool_calls", 0),
            "n_claims": len(snap.values.get("claims", [])),
            "has_report": bool(snap.values.get("report")),
            "created_at": snap.created_at or "",
        })
    return steps


def list_threads(db_path: str | Path | None = None) -> list[str]:
    path = Path(db_path or settings.checkpoint_db)
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(str(path))
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except sqlite3.Error:
        return []
