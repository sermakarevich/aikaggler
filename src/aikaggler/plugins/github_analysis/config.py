from __future__ import annotations

from pathlib import Path

from aikaggler.plugins.solution_analysis.config import (
    DEFAULT_OUTPUT_ROOT,
    OLLAMA_MODEL,
    OLLAMA_URL,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Number of items merged per ollama call at each hierarchical aggregation
# level. With CHUNK_SIZE=10 a 50-file repo collapses as 50 -> 5 -> 1.
CHUNK_SIZE = 10

# Cap on files actually deep-analyzed per repo. Repos with more files than
# this go through a two-round LLM-driven selection step; smaller repos skip
# selection and analyze every eligible file.
MAX_FILES_PER_REPO = 20

# Round-1 cap on how many orientation files the model may ask to read before
# committing to the top-N selection.
MAX_SELECTION_REQUESTS = 5

# Per orientation-file excerpt size handed to the round-2 selection prompt.
MAX_SELECTION_BODY_CHARS = 6000

# Per-file body truncation handed to ollama (chars). qwen3.6 context is ~32k
# tokens; this leaves room for the prompt + structured response.
MAX_FILE_BODY_CHARS = 30000

# Skip files larger than this on disk (bytes) before reading. Acts as a
# binary/generated-file guard; the body truncation above bounds prompt size.
MAX_FILE_BYTES = 200_000

CLONE_TIMEOUT = 180
SELECT_TIMEOUT = 600
ANALYZE_TIMEOUT = 600
AGGREGATE_TIMEOUT = 900

# Roles to keep from solutions/*/analysis.json `github_repos[].role`.
# Default to the author's own competition code; libraries / references are
# usually third-party (pytorch, transformers, etc.) and not informative.
DEFAULT_ROLES = ("solution",)

INCLUDED_SUFFIXES = frozenset({
    ".py", ".ipynb", ".md", ".rst", ".sh", ".cfg", ".toml", ".yaml", ".yml",
})

EXCLUDED_DIR_NAMES = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "build", "dist", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".eggs", "site-packages", ".idea", ".vscode", "wandb", "outputs",
    "checkpoints", "weights", "logs", "tensorboard", "runs", "lightning_logs",
})

__all__ = [
    "AGGREGATE_TIMEOUT",
    "ANALYZE_TIMEOUT",
    "CHUNK_SIZE",
    "CLONE_TIMEOUT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_ROLES",
    "EXCLUDED_DIR_NAMES",
    "INCLUDED_SUFFIXES",
    "MAX_FILE_BODY_CHARS",
    "MAX_FILE_BYTES",
    "MAX_FILES_PER_REPO",
    "MAX_SELECTION_BODY_CHARS",
    "MAX_SELECTION_REQUESTS",
    "OLLAMA_MODEL",
    "OLLAMA_URL",
    "PROMPTS_DIR",
    "SELECT_TIMEOUT",
]
