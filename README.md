# ScribeSwarm — Multi-Agent Research & Report Generation

Give it a research question; a **checkpointed LangGraph swarm** plans the
report, pauses for your approval, researches every section in parallel with
real web tools, **adversarially fact-checks its own claims**, writes cited
prose, survives a critic's rubric review — and produces a report in which
**fabricating a citation is structurally impossible**.

> Mission 2030 · Project 5 (build order #2). Plan and tracker:
> [`../PROJECT_MASTER_PLAN.md`](../PROJECT_MASTER_PLAN.md)

## Architecture (LangGraph state machine)

```
 question ─► PLANNER ─► ⏸ human gate (edit/approve plan — LangGraph interrupt)
                │
                ▼  Send fan-out, one branch per section, parallel
        RESEARCH_SECTION ×N     search → fetch (HTML/PDF) → extract claims
                │               every claim carries a VERBATIM quote,
                │               verified against the fetched page text
                ▼  fan-out over sections with sourced claims
        FACT_CHECK_SECTION ×N   re-search WITHOUT the original source →
                │               corroborated / contested / unverifiable
                ▼
        WRITER ─► ANALYST ─► CRITIC ─► REVISER ─► ASSEMBLER
        cites ledger  chart     4-part    exactly    report frame built
        claims only   specs,    rubric    one        in code: provenance
        [id] markers  not code  review    revision   note, ledger, sources,
                                          pass       honesty appendix

 CROSS-CUTTING: SQLite checkpointer (kill it, resume the thread) ·
 budget governor (LLM/tool/wall caps, graceful degradation, event log) ·
 live SSE timeline UI · plan editing at the gate · checkpoint replay
```

## The honesty invariants (the differentiator)

1. **Claim ledger** — researchers return structured claims, not prose. The
   writer may cite only `[id]` markers that exist in the ledger.
2. **Quote grounding** — a web claim enters the ledger only with an exact
   quote that verbatim-matches the fetched source (whitespace/case
   normalized). Failures stay visible as `quote_not_found`, never citable.
3. **Adversarial verification** — top claims are re-searched *without* the
   original source. Contested claims render as disputed with the
   counter-source; unverifiable ones are pulled from prose into the
   **honesty appendix**.
4. **Code owns the frame** — provenance note, ledger, sources, appendix are
   assembled deterministically; the LLM can't drop or reorder them.
5. **Charts trace to claims** — the analyst emits validated JSON specs
   (never code); only sourced numbers get charted.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# zero keys needed — the deterministic mock provider runs the whole graph:
.\.venv\Scripts\python cli.py run "Compare EV battery supply chains of India, China and the EU"

# real generation: copy .env.example to .env, set GEMINI_API_KEY (or
# ANTHROPIC_API_KEY / OPENAI_API_KEY)

.\.venv\Scripts\python cli.py serve      # web UI at http://127.0.0.1:8000
```

**Web UI**: start a run with *pause for plan approval* on — the graph
checkpoints and waits; edit section headings/sub-questions and approve. The
timeline streams every agent step live (SSE); the replay tab reconstructs
any run from its checkpoints; the report tab serves the final HTML with
charts inlined.

**CLI**: `run` (with `--max-llm/--max-tools/--max-seconds/--no-verify`),
`state <thread>`, `report <thread>`, `serve`.

**Kill-safety**: stop the process mid-run; `run` again with the same
`--thread` and it resumes from the last completed superstep. Finished
threads are immutable.

## Evaluation (Phase 6)

`evals/run_benchmark.py` runs a 10-question benchmark under named configs
(multi-agent ± fact-checker, single-section baseline) and reports:

- **structural metrics, computed in code**: citation validity (do prose
  markers exist in the ledger), sourced-claim ratio, verdict counts,
  llm/tool calls, wall time
- **rubric scores from an LLM judge**: coverage / structure / factuality —
  with the self-judging caveat recorded in ADR-007 (free tier = one model
  family), so structural metrics lead

```powershell
.\.venv\Scripts\python evals/run_benchmark.py --configs multi_verify,multi_noverify --limit 3 --pace 10
```

**Live numbers pending** the Gemini free-tier quota window — the harness is
resumable (`--resume evals/results/benchmark_*.jsonl`) and paced for the
~10 RPM limit. Mock-mode mechanics are fully covered in CI.

**Production outage proof (2026-07-23):** the first live shakedown hit total
quota exhaustion — every Gemini call 429'd past the backoff tiers. The run
**still finished**: planner degraded to a fallback plan, failed pages were
skipped, the writer fell back to a deterministic ledger-cited listing, and
every degradation landed in the report's run notes
(`data/runs/live-shakedown-v2/report.md`). Degrade-never-die is verified
against a real outage; live *quality* numbers land in the next quota window.

## Tests

```powershell
.\.venv\Scripts\python -m pytest tests -q    # 34 tests, all offline
```

Everything runs on the deterministic mock provider + injected fake tools:
no network, no keys, no quota. CI (`.github/workflows/ci.yml`) runs the
same suite on every push.

## Design decisions

[`docs/decisions.md`](docs/decisions.md) — 7 ADRs: SQLite-vs-Postgres
checkpointer, mock-first LLM adapter, day-0 claim ledger, code-owned report
frame, spec-driven charts (no exec), static single-file UI, benchmark
methodology & its self-judging caveat.

## Status / roadmap (per the master plan tracker)

- [x] Phase 0 — LangGraph skeleton + SQLite checkpointing + claim schema
- [x] Phase 1 — tool registry (search/fetch/pdf) + quote-grounded claims
- [x] Phase 2 — parallel section fan-out + budget governor
- [x] Phase 3 — adversarial fact-checker + honesty appendix
- [x] Phase 4 — analyst charts + critic/revision loop + HTML export
- [x] Phase 5 — HITL plan gate + SSE timeline + replay UI
- [x] Phase 6 — benchmark harness (live numbers pending quota window)
- [ ] Phase 7 — live eval table, deployed demo, video, blog post
- [ ] Postgres checkpointer + sandboxed python tool (pending Docker — ADR-001/005)
