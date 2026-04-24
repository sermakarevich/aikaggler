from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path

from aikaggler.plugins._shared.ollama import load_prompt, ollama_call
from aikaggler.plugins.code_analysis.config import (
    AGGREGATE_TIMEOUT,
    ANALYZE_TIMEOUT,
    DEFAULT_OUTPUT_ROOT,
    MAX_NOTEBOOK_BODY_CHARS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROMPTS_DIR,
    TOP_NOTEBOOKS,
)
from aikaggler.plugins.code_analysis.structure import (
    AGGREGATED_NOTEBOOKS_SCHEMA,
    NOTEBOOK_SCHEMA,
)


def competition_dir(slug: str, root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    path = root / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_top_notebooks(slug: str, top: int) -> list[dict[str, str]]:
    """Return top-N notebooks for a competition, sorted by voteCount desc."""
    result = subprocess.run(
        [
            "kaggle", "kernels", "list",
            "--competition", slug,
            "--sort-by", "voteCount",
            "--page-size", str(max(top, 20)),
            "-v",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    rows.sort(key=lambda r: int(r.get("totalVotes") or 0), reverse=True)
    return rows[:top]


_SAFE_CHARS = re.compile(r"[^a-z0-9._-]+")


def _stem_for(position: int, ref: str) -> str:
    safe = _SAFE_CHARS.sub("-", ref.lower()).strip("-")
    return f"votes_{position:02d}_{safe}"


def pull_notebook(ref: str, dest: Path) -> Path | None:
    """Download notebook+metadata to dest. Returns path to .ipynb / .py or None."""
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "pull", ref, "-p", str(dest), "-m"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    pull failed for {ref}: {result.stderr.strip()}")
        return None
    for suffix in (".ipynb", ".py", ".r", ".rmd"):
        found = list(dest.glob(f"*{suffix}"))
        if found:
            return found[0]
    return None


def _ipynb_to_text(path: Path) -> str:
    try:
        nb = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return path.read_text(errors="replace")
    parts: list[str] = []
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = source.strip()
        if not source:
            continue
        kind = cell.get("cell_type", "code")
        if kind == "markdown":
            parts.append(source)
        else:
            parts.append(f"```python\n{source}\n```")
    return "\n\n".join(parts)


def _script_to_text(path: Path) -> str:
    return f"```{path.suffix.lstrip('.')}\n{path.read_text(errors='replace')}\n```"


def notebook_to_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        return _ipynb_to_text(path)
    return _script_to_text(path)


def analyze_notebook_with_ollama(
    title: str,
    author: str,
    keywords: list[str],
    body: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    prompt = load_prompt(
        PROMPTS_DIR,
        "analyze_notebook",
        title=title,
        author=author,
        keywords=json.dumps(keywords),
        body_excerpt=body[:MAX_NOTEBOOK_BODY_CHARS],
    )
    return ollama_call(
        prompt, NOTEBOOK_SCHEMA, ANALYZE_TIMEOUT, model=model, url=url
    )


def aggregate_notebooks_with_ollama(
    analyses: list[dict],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    payload = [
        {
            "votes": a.get("votes"),
            "title": a.get("title"),
            "purpose": a["analysis"].get("purpose"),
            "tldr": a["analysis"].get("tldr", ""),
            "summary": a["analysis"].get("summary", ""),
            "models": a["analysis"].get("models", []),
            "frameworks": a["analysis"].get("frameworks", []),
            "cv_strategy": a["analysis"].get("cv_strategy", ""),
            "preprocessing": a["analysis"].get("preprocessing", []),
            "feature_engineering": a["analysis"].get("feature_engineering", []),
            "augmentations": a["analysis"].get("augmentations", []),
            "loss_functions": a["analysis"].get("loss_functions", []),
            "ensemble": a["analysis"].get("ensemble", ""),
            "post_processing": a["analysis"].get("post_processing", ""),
            "what_worked": a["analysis"].get("what_worked", []),
            "what_did_not_work": a["analysis"].get("what_did_not_work", []),
            "critical_findings": a["analysis"].get("critical_findings", []),
            "notable_techniques": a["analysis"].get("notable_techniques", []),
        }
        for a in analyses
        if "error" not in a.get("analysis", {})
    ]
    prompt = load_prompt(
        PROMPTS_DIR,
        "aggregate_notebooks",
        payload_json=json.dumps(payload, indent=2),
    )
    return ollama_call(
        prompt,
        AGGREGATED_NOTEBOOKS_SCHEMA,
        AGGREGATE_TIMEOUT,
        model=model,
        url=url,
    )


def _write_notebook_files(
    notebook_dir: Path,
    row: dict,
    analysis: dict,
    body: str,
) -> None:
    ref = row.get("ref", "")
    url = f"https://www.kaggle.com/code/{ref}"
    header = (
        f"# {row.get('title', ref)}\n\n"
        f"- **Author:** {row.get('author', 'unknown')}\n"
        f"- **Votes:** {row.get('totalVotes', '0')}\n"
        f"- **Ref:** {ref}\n"
        f"- **URL:** {url}\n"
        f"- **Last run:** {row.get('lastRunTime', '')}\n\n---\n\n"
    )
    (notebook_dir / "notebook.md").write_text(header + body)
    (notebook_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))


def _render_aggregated_markdown(
    slug: str,
    agg: dict,
    analyses: list[dict],
) -> str:
    lines = [f"# {slug}: top public notebooks", ""]
    lines += [agg.get("notebooks_tldr", ""), ""]

    def _section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.extend(f"- {x}" for x in items)
        lines.append("")

    _section("Common purposes", agg.get("common_purposes", []))
    _section("Models", agg.get("all_models", []))
    _section("Frameworks", agg.get("all_frameworks", []))
    _section("CV strategies", agg.get("all_cv_strategies", []))
    _section("Preprocessing", agg.get("all_preprocessing", []))
    _section("Feature engineering", agg.get("all_feature_engineering", []))
    _section("Augmentations", agg.get("all_augmentations", []))
    _section("Loss functions", agg.get("all_loss_functions", []))
    _section("Ensemble patterns", agg.get("all_ensemble_patterns", []))
    _section("Post-processing", agg.get("all_post_processing", []))
    _section("What worked", agg.get("all_what_worked", []))
    _section("What did not work", agg.get("all_what_did_not_work", []))
    _section("Critical findings", agg.get("all_critical_findings", []))
    _section("Notable techniques", agg.get("all_notable_techniques", []))
    _section(
        "Notable individual insights",
        agg.get("notable_individual_insights", []),
    )

    lines.append("## Notebooks indexed")
    for a in sorted(
        analyses, key=lambda x: int(x.get("votes") or 0), reverse=True
    ):
        votes = a.get("votes", "?")
        stem = a["stem"]
        title = a.get("title") or a.get("ref", stem)
        ref = a.get("ref", "")
        url = f"https://www.kaggle.com/code/{ref}"
        lines.append(
            f"- #{votes} votes [[notebooks/{stem}/notebook|{title}]] "
            f"([kaggle]({url}))"
        )
    lines.append("")

    return "\n".join(lines)


def cmd_notebooks(args: argparse.Namespace) -> int:
    out = competition_dir(args.slug)

    rows = list_top_notebooks(args.slug, args.top)
    print(f"Fetched {len(rows)} notebooks for {args.slug}")
    if not rows:
        print("No notebooks found; skipping.")
        return 0

    all_analyses: list[dict] = []
    for i, row in enumerate(rows, 1):
        ref = row.get("ref", "")
        votes = int(row.get("totalVotes") or 0)
        title = row.get("title") or ref
        stem = _stem_for(i, ref)
        notebook_dir = out / "notebooks" / stem

        print(f"  [{i}/{len(rows)}] pulling {ref} (votes={votes})")
        source = pull_notebook(ref, notebook_dir)
        if source is None:
            print(f"    skipping {ref}: no source downloaded")
            continue

        body = notebook_to_text(source)
        keywords = _keywords_for(notebook_dir)

        try:
            analysis = analyze_notebook_with_ollama(
                title, row.get("author", ""), keywords, body, model=args.model
            )
        except Exception as e:
            analysis = {"error": f"{type(e).__name__}: {e}"}
        analysis["source_url"] = f"https://www.kaggle.com/code/{ref}"
        analysis["votes"] = votes
        _write_notebook_files(notebook_dir, row, analysis, body)

        all_analyses.append({
            "ref": ref,
            "title": title,
            "votes": votes,
            "stem": stem,
            "analysis": analysis,
        })

    print(f"Wrote {len(all_analyses)} notebook writeups to {out}/notebooks/")

    print(f"Aggregating {len(all_analyses)} notebooks with {args.model}...")
    try:
        agg = aggregate_notebooks_with_ollama(all_analyses, model=args.model)
    except Exception as e:
        agg = {"error": f"{type(e).__name__}: {e}"}
    (out / "aggregated_notebooks.json").write_text(json.dumps(agg, indent=2))
    if "error" not in agg:
        md = _render_aggregated_markdown(args.slug, agg, all_analyses)
        (out / "aggregated_notebooks.md").write_text(md)
        print(f"Wrote aggregated notebooks to {out}/aggregated_notebooks.md")
    else:
        print(f"Aggregation failed: {agg['error']}")
    return 0


def _keywords_for(notebook_dir: Path) -> list[str]:
    meta_path = notebook_dir / "kernel-metadata.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    raw = meta.get("keywords") or []
    return [str(k) for k in raw]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akc-notebooks")
    parser.add_argument("slug")
    parser.add_argument("--top", type=int, default=TOP_NOTEBOOKS)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args(argv)
    return cmd_notebooks(args)


if __name__ == "__main__":
    raise SystemExit(main())
