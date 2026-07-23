"""Spec-driven chart rendering + standalone HTML export (Phase 4).

The analyst LLM never writes code — it emits a JSON chart SPEC referencing
ledger claim ids, and this module renders it with matplotlib in-process
(ADR-005): structurally safe on a machine with no Docker sandbox, and every
chart is traceable to the claims that back it.
"""

from __future__ import annotations

import base64
import html as html_mod
import re
from pathlib import Path

VALID_KINDS = ("bar", "line")


def validate_spec(spec: dict, valid_claim_ids: set[str]) -> str | None:
    """Return a rejection reason, or None if the spec is renderable."""
    if spec.get("kind") not in VALID_KINDS:
        return f"unsupported kind {spec.get('kind')!r}"
    x, y = spec.get("x"), spec.get("y")
    if not isinstance(x, list) or not isinstance(y, list) or not x or len(x) != len(y):
        return "x/y must be equal-length non-empty lists"
    try:
        [float(v) for v in y]
    except (TypeError, ValueError):
        return "y values must be numeric"
    ids = spec.get("claim_ids") or []
    if not ids or not set(ids) <= valid_claim_ids:
        return "chart must reference existing claim ids"
    if not str(spec.get("title", "")).strip():
        return "missing title"
    return None


def render_chart(spec: dict, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [str(v) for v in spec["x"]]
    y = [float(v) for v in spec["y"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    if spec["kind"] == "bar":
        ax.bar(x, y, color="#1f4d3a")
    else:
        ax.plot(x, y, color="#1f4d3a", marker="o")
    ax.set_title(spec["title"], fontsize=11)
    if spec.get("y_label"):
        ax.set_ylabel(str(spec["y_label"]), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_html(report_md: str, charts_dir: Path, out_path: Path,
                title: str = "ScribeSwarm report") -> Path:
    """Markdown -> single self-contained HTML file; chart images inlined
    as base64 so the file travels alone."""
    import markdown as md

    body = md.markdown(report_md, extensions=["tables"])

    def _inline(match: re.Match) -> str:
        src = match.group(1)
        img = charts_dir / Path(src).name
        if img.exists():
            b64 = base64.b64encode(img.read_bytes()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return match.group(0)

    body = re.sub(r'src="([^"]+\.png)"', _inline, body)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_mod.escape(title)}</title>
<style>
 body {{ max-width: 46rem; margin: 2rem auto; padding: 0 1rem;
        font: 16px/1.6 Georgia, serif; color: #1c1c1a; background: #fbfaf7; }}
 h1, h2 {{ line-height: 1.25; }}
 blockquote {{ border-left: 3px solid #1f4d3a; margin: 1rem 0;
              padding: .25rem 1rem; color: #444; background: #f2f0ea; }}
 code, pre {{ font-family: Consolas, monospace; font-size: .9em; }}
 img {{ max-width: 100%; }}
 hr {{ border: 0; border-top: 1px solid #ddd; }}
</style></head><body>
{body}
</body></html>"""
    out_path.write_text(page, encoding="utf-8")
    return out_path
