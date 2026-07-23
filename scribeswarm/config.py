"""Central configuration. Every knob is overridable via env vars (SCRIBE_*)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """Export .env entries into os.environ (pydantic-settings reads the file
    for SCRIBE_* fields but does NOT export it — provider API keys like
    GEMINI_API_KEY are read from os.environ by the LLM adapter)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            if val.strip():
                os.environ.setdefault(key.strip(), val.strip())


_load_env_file(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCRIBE_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    # State / persistence
    checkpoint_db: str = str(PROJECT_ROOT / "data" / "checkpoints.sqlite")
    runs_dir: str = str(PROJECT_ROOT / "data" / "runs")

    # Plan shape
    max_sections: int = 4
    max_sub_questions: int = 3

    # Research tools (Phase 1)
    search_results_per_query: int = 4
    pages_per_section: int = 2       # fetch cap per section
    extract_chars: int = 12000       # page text given to the claim extractor
    max_claims_per_page: int = 4

    # Budget governor (Phase 2) — hard caps per run
    max_llm_calls: int = 60
    max_tool_calls: int = 80
    max_wall_seconds: float = 600.0  # 0 disables the wall clock cap

    # Adversarial fact-checking (Phase 3)
    verify_claims: bool = True
    max_verify_per_section: int = 3  # highest-confidence claims re-checked

    # Analyst + critic (Phase 4)
    max_charts: int = 2              # chart specs accepted per run
    critic_enabled: bool = True      # rubric review + one revision pass

    # LLM
    llm_provider: str = "auto"    # auto | anthropic | openai | gemini | mock
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash"
    max_output_tokens: int = 2000
    request_timeout: float = 120.0


settings = Settings()
