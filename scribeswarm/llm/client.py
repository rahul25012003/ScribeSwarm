"""Provider-agnostic LLM adapter (pattern proven in LexRAG).

Provider resolution (SCRIBE_LLM_PROVIDER=auto):
    ANTHROPIC_API_KEY set  -> anthropic (official SDK)
    OPENAI_API_KEY set     -> openai    (REST via httpx)
    GEMINI_API_KEY set     -> gemini    (REST via httpx)
    none of the above      -> mock      (deterministic canned outputs)

The mock provider is a first-class citizen, not a test hack: the entire
graph — plan, research, write, checkpoint, resume — runs offline with it,
so tests never touch the network and dev doesn't burn API quota.
"""

from __future__ import annotations

import json
import os
import re

import httpx

from scribeswarm.config import settings


def resolve_provider() -> str:
    configured = settings.llm_provider.lower()
    if configured != "auto":
        return configured
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return "mock"


def generate(system: str, user: str) -> str:
    provider = resolve_provider()
    if provider == "anthropic":
        return _anthropic(system, user)
    if provider == "openai":
        return _openai(system, user)
    if provider == "gemini":
        return _gemini(system, user)
    if provider == "mock":
        return _mock(system, user)
    raise ValueError(f"Unknown LLM provider: {provider}")


def extract_json(text: str):
    """Parse JSON out of an LLM reply that may wrap it in ``` fences or prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # last resort: widest brace/bracket span
    for open_ch, close_ch in ("{}", "[]"):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No parseable JSON in LLM reply: {text[:200]!r}")


# --- real providers ---------------------------------------------------------

def _anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.max_output_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this request."
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _openai(system: str, user: str) -> str:
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={
            "model": settings.openai_model,
            "max_tokens": settings.max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=settings.request_timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _gemini(system: str, user: str) -> str:
    import time

    for attempt, backoff in enumerate((15, 30, 0)):
        resp = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"maxOutputTokens": settings.max_output_tokens},
            },
            timeout=settings.request_timeout,
        )
        if resp.status_code == 429 and backoff:   # free tier ~10 RPM
            time.sleep(backoff)
            continue
        break
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates or "content" not in candidates[0]:
        reason = candidates[0].get("finishReason", "?") if candidates else "no candidates"
        raise RuntimeError(f"Gemini returned no content (finishReason: {reason})")
    parts = candidates[0]["content"].get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


# --- mock provider ----------------------------------------------------------
# Routes on the agent-role marker each node puts at the start of its system
# prompt. Outputs are deterministic functions of the input, so full-graph
# tests can assert exact structure.

def _mock(system: str, user: str) -> str:
    match = re.match(r"You are the (\w+)", system or "")
    role = match.group(1) if match else ""
    if role == "PLANNER":
        return _mock_plan(user)
    if role == "RESEARCHER":
        return _mock_claims(user)
    if role == "EXTRACTOR":
        return _mock_extract(user)
    if role == "VERIFIER":
        return _mock_verify(user)
    if role == "ANALYST":
        return _mock_analyst(user)
    if role == "CRITIC":
        return _mock_critic(user)
    if role == "WRITER":
        return _mock_draft(user)
    if role == "JUDGE":
        return json.dumps({"coverage": 3, "structure": 4, "factuality": 4,
                           "rationale": "deterministic mock judgement"})
    return "mock reply"


def _mock_plan(user: str) -> str:
    topic = user.strip().splitlines()[-1][:60] or "the topic"
    sections = [
        {"id": f"s{i}", "heading": h,
         "sub_questions": [f"{topic} {h.lower()}"]}
        for i, h in enumerate(
            ["Background", "Current landscape", "Comparison", "Outlook"], start=1
        )
    ][: settings.max_sections]
    return json.dumps({"title": f"Report: {topic}", "sections": sections})


def _mock_claims(user: str) -> str:
    match = re.search(r"section id: (\S+)", user)
    sid = match.group(1) if match else "s?"
    return json.dumps([
        {"text": f"Deterministic mock claim A for {sid}.", "confidence": 0.9},
        {"text": f"Deterministic mock claim B for {sid}.", "confidence": 0.6},
    ])


def _mock_extract(user: str) -> str:
    url_match = re.search(r"URL: (\S+)", user)
    url = url_match.group(1) if url_match else "unknown"
    body = user.split("PAGE TEXT:", 1)[-1]
    first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    return json.dumps([{
        "claim": f"Mock claim extracted from {url}.",
        "quote": first_line[:120],
        "confidence": 0.8,
    }])


def _mock_verify(user: str) -> str:
    claim_match = re.search(r"CLAIM: (.+)", user)
    claim = claim_match.group(1) if claim_match else ""
    if "disputed" in claim.lower():
        body = user.split("INDEPENDENT EVIDENCE TEXT:", 1)[-1]
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        return json.dumps({"status": "contested",
                           "reason": "mock found disagreement",
                           "counter_quote": first[:120]})
    return json.dumps({"status": "corroborated",
                       "reason": "mock corroboration",
                       "counter_quote": ""})


def _mock_analyst(user: str) -> str:
    """Build one bar chart from up to three claims that contain a number."""
    rows = re.findall(r"\[(\S+)\] \((\S+)\) (.+)", user)
    points = []
    for cid, sid, text in rows:
        num = re.search(r"\d+(?:\.\d+)?", text)
        if num:
            points.append((cid, sid, float(num.group(0))))
        if len(points) == 3:
            break
    if not points:
        return "[]"
    return json.dumps([{
        "section_id": points[0][1],
        "title": "Key figures from the claim ledger",
        "kind": "bar",
        "x": [p[0] for p in points],
        "y": [p[2] for p in points],
        "y_label": "value",
        "claim_ids": [p[0] for p in points],
    }])


def _mock_critic(user: str) -> str:
    cited = bool(re.search(r"\[[A-Za-z0-9_]*c\d+\]", user))
    if cited:
        return json.dumps({"verdict": "accept", "notes": "",
                           "scores": {"citation_density": 4, "structure": 4,
                                      "hedging": 4, "consistency": 4}})
    return json.dumps({"verdict": "revise",
                       "notes": "No inline [id] citations — cite the claims.",
                       "scores": {"citation_density": 0, "structure": 3,
                                  "hedging": 3, "consistency": 3}})


def _mock_draft(user: str) -> str:
    match = re.search(r"## (.+)", user)
    heading = match.group(1) if match else "Section"
    ids = re.findall(r"\[([A-Za-z0-9_]*c\d+)\]", user)
    markers = " ".join(f"[{i}]" for i in ids[:2])
    return f"Mock prose for {heading}, grounded in {markers}."
