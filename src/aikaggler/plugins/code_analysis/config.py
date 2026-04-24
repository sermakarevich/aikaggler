from __future__ import annotations

from pathlib import Path

from aikaggler.plugins.solution_analysis.config import (
    AGGREGATE_TIMEOUT,
    ANALYZE_TIMEOUT,
    DEFAULT_OUTPUT_ROOT,
    OLLAMA_MODEL,
    OLLAMA_URL,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"

TOP_NOTEBOOKS = 10
MAX_NOTEBOOK_BODY_CHARS = 60000

__all__ = [
    "AGGREGATE_TIMEOUT",
    "ANALYZE_TIMEOUT",
    "DEFAULT_OUTPUT_ROOT",
    "MAX_NOTEBOOK_BODY_CHARS",
    "OLLAMA_MODEL",
    "OLLAMA_URL",
    "PROMPTS_DIR",
    "TOP_NOTEBOOKS",
]
