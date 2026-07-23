"""Config-comparison benchmark (Phase 6, ADR-007).

Runs every benchmark question under named pipeline configurations and
records two metric families per run:

  structural (computed in code, no LLM):
    citation_validity  share of prose [id] markers that exist in the ledger
    sourced_ratio      share of ledger claims grounded in fetched sources
    verdicts           corroborated / contested / unverifiable counts
    llm_calls, tool_calls, duration_s

  rubric (LLM judge — same model family as the writer on the free tier, so
  treat as directional; structural metrics lead):
    coverage, structure, factuality  (0-5 each)

Incremental JSONL output; --resume FILE re-attempts only missing pairs.

Run:  python evals/run_benchmark.py [--configs multi_verify,multi_noverify]
      [--limit N] [--pace SECONDS] [--resume FILE.jsonl]
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scribeswarm.config import settings  # noqa: E402
from scribeswarm.llm.client import extract_json, generate  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"

CONFIGS: dict[str, dict] = {
    # name -> settings overrides applied for the run
    "multi_verify": {"verify_claims": True, "max_sections": 4},
    "multi_noverify": {"verify_claims": False, "max_sections": 4},
    "single_section": {"verify_claims": False, "max_sections": 1},
}

JUDGE_SYSTEM = (
    "You are the JUDGE of a research report. Score it 0-5 on each axis: "
    "coverage (does it actually answer the question, all major angles), "
    "structure (organized, no filler), factuality (statements plausible and "
    "internally consistent; penalize obvious errors). Reply with ONLY valid "
    'JSON: {"coverage": int, "structure": int, "factuality": int, '
    '"rationale": str}. No prose, no fences.'
)


def structural_metrics(state: dict) -> dict:
    claims = state.get("claims", [])
    ledger_ids = {c["id"] for c in claims}
    prose = "\n".join(d["markdown"] for d in state.get("drafts", []))
    markers = re.findall(r"\[([A-Za-z0-9_]*c\d+)\]", prose)
    valid = [m for m in markers if m in ledger_ids]
    sourced = [c for c in claims if c["source_type"] != "model_prior"]
    verdicts = {"corroborated": 0, "contested": 0, "unverifiable": 0}
    for v in state.get("verdicts", []):
        if v["status"] in verdicts:
            verdicts[v["status"]] += 1
    return {
        "citation_markers": len(markers),
        "citation_validity": round(len(valid) / len(markers), 3) if markers else None,
        "n_claims": len(claims),
        "sourced_ratio": round(len(sourced) / len(claims), 3) if claims else 0.0,
        "verdicts": verdicts,
        "n_charts": len(state.get("charts", [])),
        "llm_calls": state.get("llm_calls", 0),
        "tool_calls": state.get("tool_calls", 0),
    }


def judge(question: str, report: str) -> dict:
    raw = generate(JUDGE_SYSTEM,
                   f"QUESTION:\n{question}\n\nREPORT:\n{report[:18000]}")
    try:
        v = extract_json(raw)
        return {k: int(v.get(k, 0)) for k in ("coverage", "structure", "factuality")} | {
            "rationale": str(v.get("rationale", ""))[:400]}
    except (ValueError, TypeError):
        return {"coverage": None, "structure": None, "factuality": None,
                "rationale": "judge reply unparseable"}


def _load_previous(path: Path) -> dict[tuple[str, str], dict]:
    done = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                done[(row["qid"], row["config"])] = row
    return done


def run(config_names: list[str], limit: int | None = None,
        pace: float = 0.0, resume: Path | None = None) -> dict:
    from scribeswarm.graph.build import run as run_graph

    questions = [json.loads(l) for l in
                 (ROOT / "evals" / "benchmark.jsonl").read_text(
                     encoding="utf-8").splitlines() if l.strip()]
    if limit:
        questions = questions[:limit]

    previous = _load_previous(resume) if resume else {}
    if previous:
        print(f"Resuming: {len(previous)} runs carried over from {resume.name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"benchmark_{stamp}.jsonl"
    rows: list[dict] = []

    with out.open("a", encoding="utf-8") as fh:
        for cfg_name in config_names:
            overrides = CONFIGS[cfg_name]
            for q in questions:
                key = (q["id"], cfg_name)
                if key in previous:
                    row = previous[key]
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    continue
                print(f"[{cfg_name}] {q['id']}: {q['question'][:60]}...")
                saved = {k: getattr(settings, k) for k in overrides}
                for k, v in overrides.items():
                    setattr(settings, k, v)
                t0 = time.perf_counter()
                row = {"qid": q["id"], "config": cfg_name,
                       "thread_id": f"bench_{cfg_name}_{q['id']}_{stamp}"}
                try:
                    state = run_graph(q["question"], row["thread_id"])
                    row["duration_s"] = round(time.perf_counter() - t0, 1)
                    row.update(structural_metrics(state))
                    row["judge"] = judge(q["question"], state.get("report", ""))
                    row["ok"] = True
                except Exception as exc:
                    row["ok"] = False
                    row["error"] = str(exc)[:300]
                finally:
                    for k, v in saved.items():
                        setattr(settings, k, v)
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if pace:
                    time.sleep(pace)

    summary = _summarize(rows)
    table = _table(summary)
    (RESULTS_DIR / f"benchmark_{stamp}.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"rows: {out}")
    return {"rows": rows, "summary": summary, "output": str(out)}


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _summarize(rows: list[dict]) -> dict:
    by_cfg: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("ok"):
            by_cfg.setdefault(r["config"], []).append(r)
    out = {}
    for cfg, rs in by_cfg.items():
        out[cfg] = {
            "runs": len(rs),
            "citation_validity": _mean([r["citation_validity"] for r in rs]),
            "sourced_ratio": _mean([r["sourced_ratio"] for r in rs]),
            "corroborated": sum(r["verdicts"]["corroborated"] for r in rs),
            "contested": sum(r["verdicts"]["contested"] for r in rs),
            "unverifiable": sum(r["verdicts"]["unverifiable"] for r in rs),
            "coverage": _mean([r["judge"]["coverage"] for r in rs]),
            "structure": _mean([r["judge"]["structure"] for r in rs]),
            "factuality": _mean([r["judge"]["factuality"] for r in rs]),
            "llm_calls": _mean([r["llm_calls"] for r in rs]),
            "tool_calls": _mean([r["tool_calls"] for r in rs]),
            "duration_s": _mean([r["duration_s"] for r in rs]),
        }
    return out


def _table(summary: dict) -> str:
    cols = ["runs", "citation_validity", "sourced_ratio", "corroborated",
            "contested", "unverifiable", "coverage", "structure",
            "factuality", "llm_calls", "tool_calls", "duration_s"]
    lines = ["| config | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for cfg, s in summary.items():
        lines.append("| " + cfg + " | "
                     + " | ".join(str(s[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]

    def _opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    names = (_opt("--configs") or "multi_verify,multi_noverify").split(",")
    limit = int(_opt("--limit", 0)) or None
    pace = float(_opt("--pace", 0) or 0)
    resume = Path(_opt("--resume")) if _opt("--resume") else None
    run([n.strip() for n in names], limit=limit, pace=pace, resume=resume)
