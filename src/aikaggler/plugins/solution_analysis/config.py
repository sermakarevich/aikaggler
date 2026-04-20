from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "data"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/145.0 Safari/537.36"
)
KAGGLE_API = "https://www.kaggle.com/api/i"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:31b"

PROMPTS_DIR = Path(__file__).parent / "prompts"

MAX_SOLUTION_BODY_CHARS = 40000
ANALYZE_TIMEOUT = 600
AGGREGATE_TIMEOUT = 900
OLLAMA_RETRIES = 3
