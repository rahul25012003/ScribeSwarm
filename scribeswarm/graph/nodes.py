"""Phase 0 nodes: PLANNER -> RESEARCHER -> WRITER (linear).

Every node is a pure function RunState -> partial RunState update, validated
through pydantic at the boundary and stored as plain dicts so the SQLite
checkpointer can persist each step.

Honesty invariant (day 0): researchers have no tools yet, so every claim is
source_type=model_prior and the assembled report says so up front — the
report never pretends its claims are sourced before Phase 1 lands.
"""

from __future__ import annotations

from scribeswarm.config import settings
from scribeswarm.llm.client import extract_json, generate, resolve_provider
from scribeswarm.schemas import Claim, ResearchPlan, RunState, SectionDraft

PLANNER_SYSTEM = (
    "You are the PLANNER of a research-report pipeline. Decompose the user's "
    "research question into a report plan. Reply with ONLY valid JSON: "
    '{"title": str, "sections": [{"id": "s1", "heading": str, '
    '"sub_questions": [str, ...]}, ...]}. '
    f"At most {settings.max_sections} sections, "
    f"{settings.max_sub_questions} sub-questions each. No prose, no fences."
)

RESEARCHER_SYSTEM = (
    "You are the RESEARCHER for one section of a report. You currently have "
    "NO search tools, so answer from general knowledge and be conservative: "
    "state only widely known facts, assign honest confidence 0.0-1.0, and "
    "prefer fewer, safer claims. Reply with ONLY a valid JSON array: "
    '[{"text": str, "confidence": float}, ...] — 2 to 5 claims. '
    "No prose, no fences."
)

WRITER_SYSTEM = (
    "You are the WRITER for one section of a report. Write 1-2 tight "
    "markdown paragraphs for the given heading, using ONLY the numbered "
    "claims provided — cite each claim you use inline as [cN]. Do not "
    "introduce any fact that is not in a claim. No heading in your reply, "
    "no fences, just the paragraphs."
)


def _bump(state: RunState, calls: int) -> int:
    return int(state.get("llm_calls", 0)) + calls


def planner(state: RunState) -> RunState:
    raw = generate(PLANNER_SYSTEM, f"Research question:\n{state['question']}")
    plan = ResearchPlan.model_validate(extract_json(raw))
    plan.sections = plan.sections[: settings.max_sections]
    return {
        "plan": plan.model_dump(),
        "provider": resolve_provider(),
        "llm_calls": _bump(state, 1),
    }


def researcher(state: RunState) -> RunState:
    plan = ResearchPlan.model_validate(state["plan"])
    ledger: list[dict] = []
    calls = 0
    for section in plan.sections:
        subq = "\n".join(f"- {q}" for q in section.sub_questions)
        raw = generate(
            RESEARCHER_SYSTEM,
            f"Report: {plan.title}\nsection id: {section.id}\n"
            f"Section heading: {section.heading}\nSub-questions:\n{subq}",
        )
        calls += 1
        for item in extract_json(raw):
            ledger.append(Claim(
                id=f"c{len(ledger) + 1}",
                section_id=section.id,
                text=str(item.get("text", "")).strip(),
                confidence=float(item.get("confidence", 0.5)),
            ).model_dump())
    return {"claims": ledger, "llm_calls": _bump(state, calls)}


def writer(state: RunState) -> RunState:
    plan = ResearchPlan.model_validate(state["plan"])
    claims = [Claim.model_validate(c) for c in state["claims"]]
    drafts: list[dict] = []
    calls = 0
    for section in plan.sections:
        own = [c for c in claims if c.section_id == section.id]
        listing = "\n".join(f"[{c.id}] {c.text}" for c in own)
        raw = generate(
            WRITER_SYSTEM,
            f"## {section.heading}\nClaims you may cite:\n{listing}",
        )
        calls += 1
        drafts.append(SectionDraft(
            section_id=section.id,
            heading=section.heading,
            markdown=raw.strip(),
            cited_claim_ids=[c.id for c in own if f"[{c.id}]" in raw],
        ).model_dump())
    return {
        "drafts": drafts,
        "report": _assemble(plan, drafts, claims),
        "llm_calls": _bump(state, calls),
    }


def _assemble(plan: ResearchPlan, drafts: list[dict], claims: list[Claim]) -> str:
    """Deterministic assembly in code — the LLM never controls the frame."""
    parts = [f"# {plan.title}", ""]
    if all(c.source_type == "model_prior" for c in claims):
        parts += [
            "> **Provenance note:** this Phase 0 report is generated from "
            "model knowledge only — no sources were searched or cited. "
            "Sourced, fact-checked claims arrive in later phases.",
            "",
        ]
    for d in drafts:
        parts += [f"## {d['heading']}", "", d["markdown"], ""]
    parts += ["---", "", "## Claim ledger", ""]
    for c in claims:
        parts.append(
            f"- **[{c.id}]** ({c.section_id}, {c.source_type}, "
            f"confidence {c.confidence:.1f}, {c.status}) {c.text}"
        )
    return "\n".join(parts)
