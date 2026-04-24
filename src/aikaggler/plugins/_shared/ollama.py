from __future__ import annotations

import json
from pathlib import Path

import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "qwen3.6:latest"
DEFAULT_RETRIES = 3


def load_prompt(prompts_dir: Path, name: str, **kwargs: str) -> str:
    text = (prompts_dir / f"{name}.txt").read_text()
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", value)
    return text


def ollama_call(
    prompt: str,
    schema: dict,
    timeout: int,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    retries: int = DEFAULT_RETRIES,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
        "format": schema,
    }
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(url, json=body, timeout=timeout)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            return json.loads(content)
        except (json.JSONDecodeError, httpx.HTTPError) as e:
            last_err = e
            if attempt < retries:
                print(
                    f"  ollama call failed ({type(e).__name__}: {e}), "
                    f"retrying {attempt}/{retries - 1}..."
                )
    raise last_err  # type: ignore[misc]
