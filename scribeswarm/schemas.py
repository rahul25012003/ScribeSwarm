"""Pydantic models for structured node I/O + the LangGraph run state.

The claim ledger is defined on day 0 (Phase 0) even though researchers have
no tools yet: every downstream phase — fact-checker, writer constraints,
honesty appendix — operates on Claim rows, so the schema is the contract.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class PlanSection(BaseModel):
    id: str
    heading: str
    sub_questions: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    title: str
    sections: list[PlanSection]


class Claim(BaseModel):
    id: str
    section_id: str
    text: str
    # Phase 0 researchers have no tools — every claim is a model prior and is
    # labeled as such. Phase 1 introduces web/pdf sources; Phase 3 the
    # fact-checker flips status to corroborated / contested / unverifiable.
    source_type: str = "model_prior"   # model_prior | web | pdf | dataset
    source_url: str = ""
    quote: str = ""
    confidence: float = 0.5
    status: str = "unverified"


class SectionDraft(BaseModel):
    section_id: str
    heading: str
    markdown: str
    cited_claim_ids: list[str] = Field(default_factory=list)


class RunState(TypedDict, total=False):
    """The LangGraph state. Values stay JSON-serializable (plain dicts) so the
    SQLite checkpointer can persist every step without custom serializers.

    Reducer-annotated keys merge across the parallel research branches
    (Phase 2 Send fan-out): each branch returns DELTAS (its own claims, its
    own call counts) and LangGraph folds them together deterministically."""

    question: str
    provider: str          # resolved LLM provider recorded for the run
    plan: dict             # ResearchPlan.model_dump()
    budget: dict           # {max_llm_calls, max_tool_calls, max_wall_seconds, started_at}
    claims: Annotated[list[dict], operator.add]   # Claim ledger
    drafts: list[dict]     # list of SectionDraft.model_dump() (writer, single write)
    report: str            # final assembled markdown
    llm_calls: Annotated[int, operator.add]
    tool_calls: Annotated[int, operator.add]
    budget_events: Annotated[list[str], operator.add]  # degradation log
    # Fact-checker output (Phase 3) — append-only verdict events; the writer
    # folds them over the ledger (claims channel itself is append-only):
    # {claim_id, status, reason, counter_url, counter_quote}
    verdicts: Annotated[list[dict], operator.add]
