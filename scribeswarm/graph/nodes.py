"""Nodes: PLANNER -> RESEARCHER (tools) -> WRITER (linear until Phase 2).

Every node is a pure function RunState -> partial RunState update, validated
through pydantic at the boundary and stored as plain dicts so the SQLite
checkpointer can persist each step.

Phase 1 honesty invariant: a web/pdf claim enters the ledger only with an
exact quote that VERBATIM-matches the fetched source text (whitespace/case
normalized). Extractor output that fails the match is kept in the ledger as
status=quote_not_found but is never citable by the writer. Sections whose
sources all fail degrade to clearly-labeled model_prior claims — the report
frame says how many claims are sourced vs not.
"""

from __future__ import annotations

import re

from scribeswarm.config import settings
from scribeswarm.llm.client import extract_json, generate, resolve_provider
from scribeswarm.schemas import Claim, ResearchPlan, RunState, SectionDraft
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
    "claims provided — cite each claim you use inline as [cN]. Do not "
    "introduce any fact that is not in a claim. No heading in your reply, "
    "no fences, just the paragraphs."
)


def _bump(state: RunState, key: str, n: int) -> int:
    return int(state.get(key, 0)) + n


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
        "llm_calls": _bump(state, "llm_calls", 1),
    }


def researcher(state: RunState) -> RunState:
    plan = ResearchPlan.model_validate(state["plan"])
    tools = get_toolset()
    ledger: list[dict] = []
    llm_calls = tool_calls = 0

    for section in plan.sections:
        # 1) search every sub-question, dedupe URLs preserving rank order
        seen: dict[str, str] = {}       # url -> snippet/title context
        for sub_q in section.sub_questions[: settings.max_sub_questions]:
            results = tools.search.search(
                sub_q, max_results=settings.search_results_per_query)
            tool_calls += 1
            for r in results:
                seen.setdefault(r.url, r.title)

        # 2) fetch until the per-section page cap is met
        pages = []
        for url in seen:
            if len(pages) >= settings.pages_per_section:
                break
            page = tools.fetcher.fetch(url)
            tool_calls += 1
            if page:
                pages.append(page)

        # 3) extract quote-grounded claims per page
        for page in pages:
            raw = generate(EXTRACTOR_SYSTEM, (
                f"Section heading: {section.heading}\n"
                f"Sub-questions: {'; '.join(section.sub_questions)}\n"
                f"URL: {page.url}\nPAGE TITLE: {page.title}\n"
                f"PAGE TEXT:\n{page.text[: settings.extract_chars]}"
            ))
            llm_calls += 1
            for item in extract_json(raw)[: settings.max_claims_per_page]:
                quote = str(item.get("quote", "")).strip()
                grounded = quote_in_text(quote, page.text)
                ledger.append(Claim(
                    id=f"c{len(ledger) + 1}",
                    section_id=section.id,
                    text=str(item.get("claim", "")).strip(),
                    source_type=page.content_type if page.content_type == "pdf" else "web",
                    source_url=page.url,
                    quote=quote,
                    confidence=float(item.get("confidence", 0.5)),
                    status="unverified" if grounded else "quote_not_found",
                ).model_dump())

        # 4) no usable sources -> honest model-prior fallback
        if not pages:
            raw = generate(RESEARCHER_SYSTEM, (
                f"Report: {plan.title}\nsection id: {section.id}\n"
                f"Section heading: {section.heading}\nSub-questions:\n"
                + "\n".join(f"- {q}" for q in section.sub_questions)
            ))
            llm_calls += 1
            for item in extract_json(raw):
                ledger.append(Claim(
                    id=f"c{len(ledger) + 1}",
                    section_id=section.id,
                    text=str(item.get("text", "")).strip(),
                    confidence=float(item.get("confidence", 0.5)),
                ).model_dump())

    return {
        "claims": ledger,
        "llm_calls": _bump(state, "llm_calls", llm_calls),
        "tool_calls": _bump(state, "tool_calls", tool_calls),
    }


def writer(state: RunState) -> RunState:
    plan = ResearchPlan.model_validate(state["plan"])
    claims = [Claim.model_validate(c) for c in state["claims"]]
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
        "report": _assemble(plan, drafts, claims),
        "llm_calls": _bump(state, "llm_calls", calls),
    }


def _assemble(plan: ResearchPlan, drafts: list[dict], claims: list[Claim]) -> str:
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
    return "\n".join(parts)
