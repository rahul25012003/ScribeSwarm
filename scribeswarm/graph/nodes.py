"""Nodes: PLANNER -> parallel RESEARCH_SECTION fan-out -> WRITER.

Each node returns DELTAS: reducer-annotated state keys (claims, llm_calls,
tool_calls, budget_events) are merged by LangGraph across the parallel
research branches, so a node never reports totals — only what it spent.

Phase 1 honesty invariant: a web/pdf claim enters the ledger only with an
exact quote that VERBATIM-matches the fetched source text (whitespace/case
normalized); failures stay in the ledger as quote_not_found but are never
citable. Phase 2 adds the budget governor: sections degrade (fewer pages,
then no research) and log a budget event instead of dying mid-run.
"""

from __future__ import annotations

import re

from scribeswarm.config import settings
from scribeswarm.governor import allocate
from scribeswarm.llm.client import extract_json, generate, resolve_provider
from scribeswarm.schemas import Claim, PlanSection, ResearchPlan, RunState, SectionDraft
from scribeswarm.tools import get_toolset

PLANNER_SYSTEM = (
    "You are the PLANNER of a research-report pipeline. Decompose the user's "
    "research question into a report plan. Reply with ONLY valid JSON: "
    '{"title": str, "sections": [{"id": "s1", "heading": str, '
    '"sub_questions": [str, ...]}, ...]}. '
    f"At most {settings.max_sections} sections, "
    f"{settings.max_sub_questions} sub-questions each. No prose, no fences."
)

EXTRACTOR_SYSTEM = (
    "You are the EXTRACTOR. From the web page text provided, extract factual "
    "claims relevant to the section heading and sub-questions. Each claim "
    "needs an exact supporting quote COPIED VERBATIM from the page text — "
    "do not paraphrase quotes. Reply with ONLY a valid JSON array: "
    '[{"claim": str, "quote": str, "confidence": float}, ...] — up to '
    f"{settings.max_claims_per_page} claims; return [] if the page is "
    "irrelevant. No prose, no fences."
)

RESEARCHER_SYSTEM = (
    "You are the RESEARCHER for one section of a report. You have NO sources "
    "for this section (search failed), so answer from general knowledge and "
    "be conservative: state only widely known facts, assign honest "
    "confidence 0.0-1.0, and prefer fewer, safer claims. Reply with ONLY a "
    'valid JSON array: [{"text": str, "confidence": float}, ...] — 2 to 5 '
    "claims. No prose, no fences."
)

WRITER_SYSTEM = (
    "You are the WRITER for one section of a report. Write 1-2 tight "
    "markdown paragraphs for the given heading, using ONLY the numbered "
    "claims provided — cite each claim you use inline as [id]. Do not "
    "introduce any fact that is not in a claim. No heading in your reply, "
    "no fences, just the paragraphs."
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """Whitespace/case-insensitive verbatim containment check."""
    q = _norm(quote)
    return bool(q) and q in _norm(text)


def planner(state: RunState) -> RunState:
    raw = generate(PLANNER_SYSTEM, f"Research question:\n{state['question']}")
    plan = ResearchPlan.model_validate(extract_json(raw))
    plan.sections = plan.sections[: settings.max_sections]
    return {
        "plan": plan.model_dump(),
        "provider": resolve_provider(),
        "llm_calls": 1,
    }


def research_section(payload: dict) -> RunState:
    """One parallel branch: research a single section within its budget slice.

    `payload` comes from the Send fan-out, not the global state:
    {section, plan_title, n_sections, budget, llm_spent, tool_spent}."""
    section = PlanSection.model_validate(payload["section"])
    sb = allocate(payload.get("budget", {}), payload.get("n_sections", 1),
                  payload.get("llm_spent", 0), payload.get("tool_spent", 0))
    tools = get_toolset()
    ledger: list[dict] = []
    events: list[str] = []
    llm = tool = 0

    def claim_id() -> str:
        return f"{section.id}c{len(ledger) + 1}"

    # 1) search every sub-question, dedupe URLs preserving rank order
    seen: dict[str, str] = {}
    for sub_q in section.sub_questions[: settings.max_sub_questions]:
        if sb.expired():
            events.append(f"{section.id}: wall-time cap hit before search")
            break
        if tool >= sb.tool:
            events.append(f"{section.id}: tool budget exhausted during search")
            break
        results = tools.search.search(
            sub_q, max_results=settings.search_results_per_query)
        tool += 1
        for r in results:
            seen.setdefault(r.url, r.title)

    # 2) fetch until the per-section page cap or budget runs out
    pages = []
    for url in seen:
        if len(pages) >= settings.pages_per_section:
            break
        if sb.expired() or tool >= sb.tool:
            events.append(f"{section.id}: budget stopped fetching at "
                          f"{len(pages)} page(s)")
            break
        page = tools.fetcher.fetch(url)
        tool += 1
        if page:
            pages.append(page)

    # 3) extract quote-grounded claims per page
    for page in pages:
        if sb.expired() or llm >= sb.llm:
            events.append(f"{section.id}: LLM budget stopped extraction")
            break
        raw = generate(EXTRACTOR_SYSTEM, (
            f"Section heading: {section.heading}\n"
            f"Sub-questions: {'; '.join(section.sub_questions)}\n"
            f"URL: {page.url}\nPAGE TITLE: {page.title}\n"
            f"PAGE TEXT:\n{page.text[: settings.extract_chars]}"
        ))
        llm += 1
        for item in extract_json(raw)[: settings.max_claims_per_page]:
            quote = str(item.get("quote", "")).strip()
            grounded = quote_in_text(quote, page.text)
            ledger.append(Claim(
                id=claim_id(),
                section_id=section.id,
                text=str(item.get("claim", "")).strip(),
                source_type=page.content_type if page.content_type == "pdf" else "web",
                source_url=page.url,
                quote=quote,
                confidence=float(item.get("confidence", 0.5)),
                status="unverified" if grounded else "quote_not_found",
            ).model_dump())

    # 4) nothing sourced -> honest model-prior fallback (if budget allows)
    if not ledger:
        if sb.expired() or llm >= sb.llm:
            events.append(f"{section.id}: no research done — budget exhausted")
        else:
            raw = generate(RESEARCHER_SYSTEM, (
                f"Report: {payload.get('plan_title', '')}\n"
                f"section id: {section.id}\n"
                f"Section heading: {section.heading}\nSub-questions:\n"
                + "\n".join(f"- {q}" for q in section.sub_questions)
            ))
            llm += 1
            for item in extract_json(raw):
                ledger.append(Claim(
                    id=claim_id(),
                    section_id=section.id,
                    text=str(item.get("text", "")).strip(),
                    confidence=float(item.get("confidence", 0.5)),
                ).model_dump())

    return {"claims": ledger, "llm_calls": llm, "tool_calls": tool,
            "budget_events": events}


def writer(state: RunState) -> RunState:
    plan = ResearchPlan.model_validate(state["plan"])
    order = {s.id: i for i, s in enumerate(plan.sections)}
    claims = sorted(
        (Claim.model_validate(c) for c in state["claims"]),
        key=lambda c: (order.get(c.section_id, 99),
                       int(re.sub(r"\D", "", c.id) or 0)),
    )
    drafts: list[dict] = []
    calls = 0
    for section in plan.sections:
        citable = [c for c in claims
                   if c.section_id == section.id and c.status != "quote_not_found"]
        listing = "\n".join(f"[{c.id}] {c.text}" for c in citable)
        raw = generate(
            WRITER_SYSTEM,
            f"## {section.heading}\nClaims you may cite:\n{listing}",
        )
        calls += 1
        drafts.append(SectionDraft(
            section_id=section.id,
            heading=section.heading,
            markdown=raw.strip(),
            cited_claim_ids=[c.id for c in citable if f"[{c.id}]" in raw],
        ).model_dump())
    return {
        "drafts": drafts,
        "report": _assemble(plan, drafts, claims, state.get("budget_events", [])),
        "llm_calls": calls,
    }


def _assemble(plan: ResearchPlan, drafts: list[dict], claims: list[Claim],
              events: list[str]) -> str:
    """Deterministic assembly in code — the LLM never controls the frame."""
    sourced = [c for c in claims if c.source_type != "model_prior"]
    unsourced = [c for c in claims if c.source_type == "model_prior"]

    parts = [f"# {plan.title}", ""]
    if not sourced:
        parts += [
            "> **Provenance note:** this report is generated from model "
            "knowledge only — no sources were reachable. Treat every claim "
            "as unverified.",
            "",
        ]
    elif unsourced:
        parts += [
            f"> **Provenance note:** {len(sourced)} of {len(claims)} claims "
            "are quote-grounded in fetched sources; the remaining "
            f"{len(unsourced)} are unsourced model knowledge (marked "
            "model_prior in the ledger).",
            "",
        ]
    for d in drafts:
        parts += [f"## {d['heading']}", "", d["markdown"], ""]

    parts += ["---", "", "## Claim ledger", ""]
    for c in claims:
        src = f" — {c.source_url}" if c.source_url else ""
        parts.append(
            f"- **[{c.id}]** ({c.section_id}, {c.source_type}, "
            f"confidence {c.confidence:.1f}, {c.status}) {c.text}{src}"
        )

    urls: dict[str, None] = {}
    for c in sourced:
        urls.setdefault(c.source_url)
    if urls:
        parts += ["", "## Sources", ""]
        parts += [f"- {u}" for u in urls]

    if events:
        parts += ["", "## Run notes (budget governor)", ""]
        parts += [f"- {e}" for e in events]
    return "\n".join(parts)
