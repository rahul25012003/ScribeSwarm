"""Nodes: PLANNER -> parallel RESEARCH_SECTION -> parallel FACT_CHECK_SECTION
-> WRITER.

Each node returns DELTAS: reducer-annotated state keys (claims, verdicts,
llm_calls, tool_calls, budget_events) are merged by LangGraph across the
parallel branches, so a node never reports totals — only what it spent.

Honesty invariants:
- Phase 1: a web/pdf claim is citable only with a VERBATIM quote from its
  source (normalized containment); failures stay ledgered as quote_not_found.
- Phase 2: budgets degrade sections gracefully and log events, never crash.
- Phase 3: material claims are adversarially re-searched WITHOUT the original
  source. corroborated/contested survive into prose (contested rendered as
  disputed, with the counter-source); unverifiable claims are pulled from
  prose and land in the honesty appendix. The claims channel is append-only,
  so verdicts are separate events folded over the ledger at write time.
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

VERIFIER_SYSTEM = (
    "You are the VERIFIER, an adversarial fact-checker. You get a claim and "
    "INDEPENDENT evidence text found without the claim's original source. "
    "Judge strictly: corroborated (evidence clearly supports it), contested "
    "(evidence clearly disagrees — include the disagreeing passage COPIED "
    "VERBATIM as counter_quote), or unverifiable (evidence is silent or "
    "off-topic). Reply with ONLY valid JSON: "
    '{"status": "corroborated"|"contested"|"unverifiable", "reason": str, '
    '"counter_quote": str}. No prose, no fences.'
)

WRITER_SYSTEM = (
    "You are the WRITER for one section of a report. Write 1-2 tight "
    "markdown paragraphs for the given heading, using ONLY the numbered "
    "claims provided — cite each claim you use inline as [id]. Claims "
    "marked CONTESTED are disputed by an independent source: present them "
    "as disputed, never as settled fact. Do not introduce any fact that is "
    "not in a claim. No heading in your reply, no fences, just the "
    "paragraphs."
)

ANALYST_SYSTEM = (
    "You are the ANALYST. From the numbered claims provided, propose charts "
    "ONLY where claims contain comparable numbers. Reply with ONLY a valid "
    'JSON array (possibly empty): [{"section_id": str, "title": str, '
    '"kind": "bar"|"line", "x": [labels], "y": [numbers], "y_label": str, '
    '"claim_ids": [ids of the claims the numbers come from]}]. Never invent '
    "numbers that are not in a claim. No prose, no fences."
)

CRITIC_SYSTEM = (
    "You are the CRITIC. Review one report section draft against this "
    "rubric, scoring 0-5 each: citation_density (facts carry [id] markers), "
    "structure (clear, no filler), hedging (uncertain things hedged, "
    "certain things not over-hedged), consistency (no internal "
    "contradictions). Verdict revise only for real defects fixable by "
    "rewriting. Reply with ONLY valid JSON: "
    '{"verdict": "accept"|"revise", "notes": str, "scores": {"citation_density": '
    'int, "structure": int, "hedging": int, "consistency": int}}. '
    "No prose, no fences."
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


def verify_gate(state: RunState) -> RunState:
    """Barrier node between research fan-out and fact-check fan-out."""
    return {}


def fact_check_section(payload: dict) -> RunState:
    """One parallel branch: adversarially re-check the highest-confidence
    sourced claims of a single section, WITHOUT using their original source.

    `payload`: {section_id, claims, budget, n_sections, llm_spent, tool_spent}."""
    claims = [Claim.model_validate(c) for c in payload["claims"]]
    sb = allocate(payload.get("budget", {}), payload.get("n_sections", 1),
                  payload.get("llm_spent", 0), payload.get("tool_spent", 0))
    tools = get_toolset()
    verdicts: list[dict] = []
    events: list[str] = []
    llm = tool = 0

    material = sorted(claims, key=lambda c: -c.confidence)
    material = material[: settings.max_verify_per_section]

    for claim in material:
        if sb.expired() or llm >= sb.llm or tool >= sb.tool:
            events.append(f"{payload['section_id']}: verification budget "
                          f"exhausted after {len(verdicts)} claim(s)")
            break

        # independent re-search: never fetch the claim's own source
        results = tools.search.search(claim.text[:120], max_results=settings.
                                      search_results_per_query)
        tool += 1
        page = None
        for r in results:
            if r.url == claim.source_url:
                continue
            if tool >= sb.tool:
                break
            page = tools.fetcher.fetch(r.url)
            tool += 1
            if page:
                break

        if page is None:
            verdicts.append({"claim_id": claim.id, "status": "unverifiable",
                             "reason": "no independent source found",
                             "counter_url": "", "counter_quote": ""})
            continue

        raw = generate(VERIFIER_SYSTEM, (
            f"CLAIM: {claim.text}\n"
            f"(original source, do NOT rely on it: {claim.source_url})\n"
            f"INDEPENDENT EVIDENCE URL: {page.url}\n"
            f"INDEPENDENT EVIDENCE TEXT:\n{page.text[: settings.extract_chars]}"
        ))
        llm += 1
        try:
            v = extract_json(raw)
        except ValueError:
            v = {"status": "unverifiable", "reason": "verifier reply unparseable"}
        status = str(v.get("status", "unverifiable"))
        if status not in ("corroborated", "contested", "unverifiable"):
            status = "unverifiable"
        counter_quote = str(v.get("counter_quote", "")).strip()
        if status == "contested" and not quote_in_text(counter_quote, page.text):
            status, counter_quote = "unverifiable", ""
            v["reason"] = "contest quote failed grounding against evidence"
        verdicts.append({
            "claim_id": claim.id,
            "status": status,
            "reason": str(v.get("reason", "")).strip(),
            "counter_url": page.url if status == "contested" else "",
            "counter_quote": counter_quote if status == "contested" else "",
        })

    return {"verdicts": verdicts, "llm_calls": llm, "tool_calls": tool,
            "budget_events": events}


CITABLE = ("corroborated", "contested", "unverified")
TRUSTED_FOR_CHARTS = ("corroborated", "unverified")


def resolved_claims(state: RunState) -> list[Claim]:
    """Ledger in deterministic order with fact-check verdicts folded in."""
    plan = ResearchPlan.model_validate(state["plan"])
    order = {s.id: i for i, s in enumerate(plan.sections)}
    vmap = {v["claim_id"]: v for v in state.get("verdicts", [])}
    claims = sorted(
        (Claim.model_validate(c) for c in state["claims"]),
        key=lambda c: (order.get(c.section_id, 99),
                       int(re.sub(r"\D", "", c.id[len(c.section_id):]) or 0)),
    )
    for c in claims:
        if c.id in vmap:
            c.status = vmap[c.id]["status"]
    return claims


def _write_section(section: PlanSection, citable: list[Claim],
                   critic_notes: str = "") -> SectionDraft:
    listing = "\n".join(
        f"[{c.id}]{' (CONTESTED)' if c.status == 'contested' else ''} {c.text}"
        for c in citable)
    user = f"## {section.heading}\nClaims you may cite:\n{listing}"
    if critic_notes:
        user += ("\n\nA reviewer rejected the previous draft. Fix exactly "
                 f"these issues:\n{critic_notes}")
    raw = generate(WRITER_SYSTEM, user)
    return SectionDraft(
        section_id=section.id,
        heading=section.heading,
        markdown=raw.strip(),
        cited_claim_ids=[c.id for c in citable if f"[{c.id}]" in raw],
        revised=bool(critic_notes),
    )


def writer(state: RunState) -> RunState:
    plan = ResearchPlan.model_validate(state["plan"])
    claims = resolved_claims(state)
    drafts: list[dict] = []
    calls = 0
    for section in plan.sections:
        citable = [c for c in claims
                   if c.section_id == section.id and c.status in CITABLE]
        drafts.append(_write_section(section, citable).model_dump())
        calls += 1
    return {"drafts": drafts, "llm_calls": calls}


def analyst(state: RunState) -> RunState:
    """Propose charts from numeric, trustworthy claims — spec-driven, never
    code (ADR-005). Invalid specs are dropped with a logged reason."""
    from scribeswarm.charts import validate_spec

    claims = [c for c in resolved_claims(state)
              if c.status in TRUSTED_FOR_CHARTS
              and c.source_type != "model_prior"     # charts imply authority:
              and re.search(r"\d", c.text)]          # sourced numbers only
    if not claims:
        return {"charts": []}
    listing = "\n".join(f"[{c.id}] ({c.section_id}) {c.text}" for c in claims)
    raw = generate(ANALYST_SYSTEM, f"Numbered claims:\n{listing}")
    try:
        specs = extract_json(raw)
    except ValueError:
        return {"charts": [], "llm_calls": 1,
                "budget_events": ["analyst reply unparseable — no charts"]}
    valid_ids = {c.id for c in claims}
    charts: list[dict] = []
    events: list[str] = []
    for spec in (specs if isinstance(specs, list) else []):
        if len(charts) >= settings.max_charts:
            events.append("chart cap reached — extra specs dropped")
            break
        reason = validate_spec(spec, valid_ids)
        if reason:
            events.append(f"chart spec dropped: {reason}")
            continue
        spec["id"] = f"ch{len(charts) + 1}"
        charts.append(spec)
    return {"charts": charts, "llm_calls": 1, "budget_events": events}


def critic(state: RunState) -> RunState:
    if not settings.critic_enabled:
        return {"critiques": []}
    critiques: list[dict] = []
    calls = 0
    for d in state.get("drafts", []):
        raw = generate(CRITIC_SYSTEM, f"## {d['heading']}\n\n{d['markdown']}")
        calls += 1
        try:
            v = extract_json(raw)
        except ValueError:
            v = {"verdict": "accept", "notes": "critic reply unparseable",
                 "scores": {}}
        critiques.append({
            "section_id": d["section_id"],
            "verdict": v.get("verdict") if v.get("verdict") in ("accept", "revise")
            else "accept",
            "notes": str(v.get("notes", "")).strip(),
            "scores": v.get("scores") or {},
        })
    return {"critiques": critiques, "llm_calls": calls}


def reviser(state: RunState) -> RunState:
    """One targeted revision pass for sections the critic rejected — a hard
    single iteration, never a loop."""
    to_fix = {c["section_id"]: c["notes"]
              for c in state.get("critiques", []) if c["verdict"] == "revise"}
    if not to_fix:
        return {}
    plan = ResearchPlan.model_validate(state["plan"])
    claims = resolved_claims(state)
    drafts = [dict(d) for d in state["drafts"]]
    events: list[str] = []
    calls = 0
    for i, d in enumerate(drafts):
        if d["section_id"] not in to_fix:
            continue
        section = next(s for s in plan.sections if s.id == d["section_id"])
        citable = [c for c in claims
                   if c.section_id == section.id and c.status in CITABLE]
        drafts[i] = _write_section(
            section, citable, critic_notes=to_fix[d["section_id"]]).model_dump()
        calls += 1
        events.append(f"{section.id}: revised once after critic review")
    return {"drafts": drafts, "llm_calls": calls, "budget_events": events}


def assemble(state: RunState) -> RunState:
    plan = ResearchPlan.model_validate(state["plan"])
    claims = resolved_claims(state)
    vmap = {v["claim_id"]: v for v in state.get("verdicts", [])}
    report = _assemble(plan, state.get("drafts", []), claims, vmap,
                       state.get("budget_events", []),
                       state.get("charts", []))
    return {"report": report}


def _assemble(plan: ResearchPlan, drafts: list[dict], claims: list[Claim],
              vmap: dict, events: list[str],
              charts: list[dict] | None = None) -> str:
    """Deterministic assembly in code — the LLM never controls the frame."""
    charts = charts or []
    sourced = [c for c in claims if c.source_type != "model_prior"]
    unsourced = [c for c in claims if c.source_type == "model_prior"]
    tally = {"corroborated": 0, "contested": 0, "unverifiable": 0}
    for c in claims:
        if c.status in tally:
            tally[c.status] += 1

    parts = [f"# {plan.title}", ""]
    if not sourced:
        parts += [
            "> **Provenance note:** this report is generated from model "
            "knowledge only — no sources were reachable. Treat every claim "
            "as unverified.",
            "",
        ]
    else:
        note = (f"> **Provenance note:** {len(sourced)} of {len(claims)} "
                "claims are quote-grounded in fetched sources")
        if unsourced:
            note += (f"; {len(unsourced)} are unsourced model knowledge "
                     "(marked model_prior in the ledger)")
        if vmap:
            note += (f". Independent fact-check: {tally['corroborated']} "
                     f"corroborated, {tally['contested']} contested, "
                     f"{tally['unverifiable']} unverifiable")
        parts += [note + ".", ""]

    for d in drafts:
        parts += [f"## {d['heading']}", "", d["markdown"], ""]
        for ch in charts:
            if ch.get("section_id") == d["section_id"]:
                backing = ", ".join(ch.get("claim_ids", []))
                parts += [f"![{ch['title']}](charts/{ch['id']}.png)",
                          f"*{ch['title']} — data from claims: {backing}*", ""]

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

    rejected = [c for c in claims if c.status in ("unverifiable", "quote_not_found")]
    contested = [c for c in claims if c.status == "contested"]
    if rejected or contested:
        parts += ["", "## Honesty appendix", ""]
        for c in contested:
            v = vmap.get(c.id, {})
            parts.append(
                f"- **[{c.id}] contested** — {c.text} · disputed by "
                f"{v.get('counter_url', '?')}: \"{v.get('counter_quote', '')}\""
            )
        for c in rejected:
            reason = (vmap.get(c.id, {}).get("reason")
                      or ("supporting quote not found in source"
                          if c.status == "quote_not_found" else ""))
            parts.append(
                f"- **[{c.id}] {c.status}** (excluded from prose) — "
                f"{c.text} · {reason}"
            )

    if events:
        parts += ["", "## Run notes (budget governor)", ""]
        parts += [f"- {e}" for e in events]
    return "\n".join(parts)
