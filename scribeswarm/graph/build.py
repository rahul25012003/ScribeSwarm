"""Graph wiring + checkpointed execution.

SQLite checkpointer from day 0 (ADR-001): every node's output is persisted,
so a killed process resumes from the last completed node with the same
thread_id — the Phase 0 seed of pause/steer/replay.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from scribeswarm.config import settings
from scribeswarm.graph import nodes
from scribeswarm.schemas import RunState


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(RunState)
    builder.add_node("planner", nodes.planner)
    builder.add_node("researcher", nodes.researcher)
    builder.add_node("writer", nodes.writer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer)


def open_checkpointer(db_path: str | Path | None = None) -> SqliteSaver:
    path = Path(db_path or settings.checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False is safe here: SqliteSaver serializes access
    # through its own internal lock.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def run(question: str, thread_id: str, db_path: str | Path | None = None) -> RunState:
    """Execute (or resume) a run. If the thread already has checkpoints, the
    graph continues from the last completed node instead of restarting."""
    saver = open_checkpointer(db_path)
    graph = build_graph(saver)
    config = {"configurable": {"thread_id": thread_id}}
    existing = graph.get_state(config)
    if existing.values and not existing.next:
        return existing.values          # finished thread — immutable result
    payload: RunState | None = (
        None if existing.values else {"question": question}
    )
    return graph.invoke(payload, config=config)


def get_state(thread_id: str, db_path: str | Path | None = None):
    graph = build_graph(open_checkpointer(db_path))
    return graph.get_state({"configurable": {"thread_id": thread_id}})
