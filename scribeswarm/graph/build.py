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


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(RunState)
    builder.add_node("planner", nodes.planner)
    builder.add_node("research_section", nodes.research_section)
    builder.add_node("writer", nodes.writer)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", _fan_out, ["research_section"])
    builder.add_edge("research_section", "writer")
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer)


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
