# Architecture Decision Records — ScribeSwarm

## ADR-001: SQLite checkpointer now, Postgres when Docker lands
**Date:** 2026-07-23 · **Status:** accepted

The master plan calls for a Postgres checkpointer from day 1. This machine
has no Docker (same constraint LexRAG hit), so we start with
`langgraph-checkpoint-sqlite` — the checkpointer is constructor-injected in
`graph/build.py:open_checkpointer`, and LangGraph's `PostgresSaver` is a
drop-in at that single seam. What we do NOT compromise on: checkpointing
itself exists from the first commit — every node result is durable, any
thread resumes after a process kill. Revisit when Docker Desktop is
installed (also unblocks LexRAG's pgvector variant).

## ADR-002: Mock LLM provider as a first-class citizen
**Date:** 2026-07-23 · **Status:** accepted

Provider resolution (auto → anthropic → openai → gemini → **mock**) mirrors
LexRAG, but where LexRAG's keyless fallback was extractive answers, here it
is a deterministic mock that emits schema-valid plans/claims/prose keyed on
the agent-role marker in the system prompt. Consequences: the full graph —
plan, research, write, checkpoint, resume — runs offline; tests never touch
the network; dev doesn't burn the shared Gemini free-tier quota (~10 RPM,
hard daily cap — learned the hard way on LexRAG's eval). Live-provider
behavior differs only in text quality, never in control flow.

## ADR-003: Claim ledger schema defined in Phase 0, before any tools exist
**Date:** 2026-07-23 · **Status:** accepted

The differentiator feature (writers can only cite ledger claims → fabricated
citations are structurally impossible) only works if every stage speaks
Claim from the start. So the `Claim` model — id, section_id, text,
source_type, source_url, quote, confidence, status — is the researcher/writer
contract from day 0, even though Phase 0 researchers have no tools and every
claim is `source_type=model_prior`. The assembled report states this in a
provenance note and prints the full ledger; honesty is a day-0 invariant,
not a Phase 3 retrofit.

## ADR-004: Report frame assembled in code, not by the LLM
**Date:** 2026-07-23 · **Status:** accepted

The writer LLM produces section prose only. Title, section order, provenance
note, and the claim-ledger appendix are assembled deterministically in
`nodes._assemble`. The LLM can degrade a paragraph; it cannot drop the
honesty header, reorder the report, or invent an appendix entry. Same
principle as LexRAG's verifier sitting outside the model.
