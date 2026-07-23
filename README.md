# ScribeSwarm — Multi-Agent Research & Report Generation

Give it a research question; it plans the report, researches each section,
and writes a cited draft — with **checkpointed state** (kill the process,
resume the run) and a **claim ledger** that makes fabricated citations
structurally impossible.

> Mission 2030 · Project 5 (build order #2). Plan and tracker:
> [`../PROJECT_MASTER_PLAN.md`](../PROJECT_MASTER_PLAN.md)

## Status: Phase 0 (LangGraph skeleton) — in progress

```
question ─► PLANNER ─► RESEARCHER ─► WRITER ─► report.md
            (plan      (claim        (prose cites ONLY
             JSON)      ledger)       ledger claims [cN])

            every arrow checkpointed to SQLite (resume any thread)
```

What exists now:
- **LangGraph linear graph** with SQLite checkpointing from the first
  commit — `run --thread t1`, kill it, run again: it resumes, and a
  finished thread returns its stored result instead of re-executing
- **Claim ledger schema** (ADR-003): researchers return structured claims,
  the writer may cite only `[cN]` markers that exist in the ledger
- **Honest provenance**: Phase 0 researchers have no tools, so every claim
  is labeled `model_prior` and the report opens with a provenance note
  saying exactly that — no fake authority
- **LLM adapter** (auto → anthropic / openai / gemini / **mock**): the
  deterministic mock runs the whole pipeline offline — all tests, zero
  network, zero quota
- Report frame (title, sections, ledger appendix) assembled **in code**,
  not by the LLM (ADR-004)

Coming per the plan: tool registry + real sources (Ph 1), parallel fan-out +
budget governor (Ph 2), adversarial fact-checker (Ph 3), critic/charts/PDF
(Ph 4), live timeline UI + HITL plan editing (Ph 5), agent evals (Ph 6).

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# zero keys needed — runs on the deterministic mock provider:
.\.venv\Scripts\python cli.py run "Compare EV battery supply chains of India, China and the EU"

# with a GEMINI_API_KEY in .env (copy .env.example) you get real generation
.\.venv\Scripts\python cli.py state <thread_id>    # checkpoint summary
.\.venv\Scripts\python cli.py report <thread_id>   # print saved report
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest tests -q    # 6 tests, all offline (mock)
```

## Design decisions

See [`docs/decisions.md`](docs/decisions.md) — SQLite-vs-Postgres
checkpointer, mock-first LLM adapter, day-0 claim ledger, code-owned report
frame.
