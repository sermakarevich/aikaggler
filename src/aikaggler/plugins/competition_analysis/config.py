from __future__ import annotations

from pathlib import Path

from aikaggler.plugins.solution_analysis.config import (
    ANALYZE_TIMEOUT,
    DEFAULT_OUTPUT_ROOT,
    MAX_SOLUTION_BODY_CHARS,
    OLLAMA_MODEL,
    OLLAMA_URL,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"

__all__ = [
    "ANALYZE_TIMEOUT",
    "DEFAULT_OUTPUT_ROOT",
    "MAX_SOLUTION_BODY_CHARS",
    "OLLAMA_MODEL",
    "OLLAMA_URL",
    "PROMPTS_DIR",
]
