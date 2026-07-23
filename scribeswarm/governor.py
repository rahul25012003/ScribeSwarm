"""Budget governor (Phase 2).

Hard caps per run — max LLM calls, max tool calls, max wall seconds — split
into per-section allocations at fan-out time. Agents DEGRADE when a leg runs
out (fewer pages, fewer extractions, finally no research at all) and log a
budget event; they never die mid-run. Parallel branches cannot observe each
other's live spend (LangGraph merges state at superstep boundaries), so
allocation is up-front per branch — documented, deterministic, and honest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SectionBudget:
    llm: int             # LLM calls this section may spend on research
    tool: int            # search+fetch calls this section may spend
    deadline: float      # absolute epoch seconds; 0 = no wall cap

    def expired(self) -> bool:
        return bool(self.deadline) and time.time() >= self.deadline


def allocate(budget: dict, n_sections: int, llm_spent: int, tool_spent: int) -> SectionBudget:
    """Split what's left of the run budget evenly across sections, reserving
    one LLM call per section for the writer."""
    n = max(1, n_sections)
    writer_reserve = n
    llm_left = int(budget.get("max_llm_calls", 60)) - llm_spent - writer_reserve
    tool_left = int(budget.get("max_tool_calls", 80)) - tool_spent
    wall = float(budget.get("max_wall_seconds", 0) or 0)
    started = float(budget.get("started_at", 0) or 0)
    return SectionBudget(
        llm=max(0, llm_left // n),
        tool=max(0, tool_left // n),
        deadline=(started + wall) if (wall and started) else 0.0,
    )
