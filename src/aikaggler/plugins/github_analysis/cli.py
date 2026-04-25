from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from aikaggler.plugins._shared.ollama import load_prompt, ollama_call
from aikaggler.plugins.code_analysis.cli import notebook_to_text
from aikaggler.plugins.github_analysis.config import (
    AGGREGATE_TIMEOUT,
    ANALYZE_TIMEOUT,
    CHUNK_SIZE,
    CLONE_TIMEOUT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_ROLES,
    EXCLUDED_DIR_NAMES,
    INCLUDED_SUFFIXES,
    MAX_FILE_BODY_CHARS,
    MAX_FILE_BYTES,
    MAX_FILES_PER_REPO,
    MAX_SELECTION_BODY_CHARS,
    MAX_SELECTION_REQUESTS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROMPTS_DIR,
    SELECT_TIMEOUT,
)
from aikaggler.plugins.github_analysis.structure import (
    AGGREGATED_REPOS_SCHEMA,
    FILE_REQUEST_SCHEMA,
    FILE_SCHEMA,
    REPO_SCHEMA,
    TOP_FILES_SCHEMA,
)


_SAFE_CHARS = re.compile(r"[^a-z0-9._-]+")
_DOC_SUFFIXES = {".md", ".rst"}


def _load_valid_json(path: Path) -> dict | None:
    """Read JSON if present and parseable; treat dicts with 'error' as missing."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and "error" in data:
        return None
    return data if isinstance(data, dict) else None


def competition_dir(slug: str, root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    path = root / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def _repo_stem(owner: str, repo: str) -> str:
    safe = _SAFE_CHARS.sub("-", f"{owner}-{repo}".lower()).strip("-")
    return f"repo_{safe}"


def _flatten_path(rel: Path) -> str:
    """Flatten a relative path into a single filename stem joined by '_'.

    >>> _flatten_path(Path("src/models/resnet.py"))
    'src_models_resnet.py'
    >>> _flatten_path(Path("README.md"))
    'readme.md'
    """
    s = str(rel).lower().replace("\\", "/").replace("/", "_")
    return _SAFE_CHARS.sub("-", s).strip("-")


def _load_solution_url_roles(out: Path) -> dict[str, dict]:
    """url -> {role, rank} lookup, prefers 'solution' if multiple solutions disagree."""
    priority = {"solution": 0, "library": 1, "reference": 2}
    best: dict[str, dict] = {}
    sols = out / "solutions"
    if not sols.is_dir():
        return best
    for analysis_path in sols.glob("*/analysis.json"):
        try:
            data = json.loads(analysis_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rank = data.get("rank")
        for repo in data.get("github_repos", []):
            url = (repo.get("url") or "").rstrip("/")
            role = repo.get("role")
            if not url or not role:
                continue
            current = best.get(url)
            if current is None or priority.get(role, 9) < priority.get(current["role"], 9):
                best[url] = {"role": role, "rank": rank}
    return best


def _load_targets(
    out: Path,
    roles: tuple[str, ...],
) -> list[dict]:
    links_path = out / "github_links.json"
    if not links_path.exists():
        print(f"  no github_links.json at {links_path}; run solutions first")
        return []
    try:
        links = json.loads(links_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  failed to read github_links.json: {exc}")
        return []

    role_map = _load_solution_url_roles(out)
    seen_urls: set[str] = set()
    targets: list[dict] = []
    for link in links:
        url = (link.get("url") or "").rstrip("/")
        if not url or url in seen_urls:
            continue
        info = role_map.get(url) or {}
        role = info.get("role", "")
        if roles and role not in roles:
            continue
        seen_urls.add(url)
        targets.append({
            "url": url,
            "owner": link.get("owner", ""),
            "repo": link.get("repo", ""),
            "role": role,
            "rank": info.get("rank"),
            "topic_id": link.get("topic_id"),
            "title": link.get("title", ""),
        })
    targets.sort(key=lambda t: (t.get("rank") if isinstance(t.get("rank"), int) else 999))
    return targets


def clone_repo(url: str, dest: Path) -> Path | None:
    """Shallow-clone the repo into dest. Idempotent: if already populated, returns dest."""
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--single-branch",
                "--no-tags", url, str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        print(f"    clone failed for {url}: {err[0]}")
        if dest.is_dir():
            shutil.rmtree(dest, ignore_errors=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"    clone timed out for {url}")
        if dest.is_dir():
            shutil.rmtree(dest, ignore_errors=True)
        return None
    return dest


def _walk_repo(clone_dir: Path) -> list[Path]:
    """Return analysis-eligible files (extension + size + dir filtered), sorted
    lexicographically by full path so files from the same folder cluster together
    in chunks during hierarchical aggregation."""
    out: list[Path] = []
    for path in clone_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(clone_dir)
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        suffix = rel.suffix.lower()
        if suffix not in INCLUDED_SUFFIXES:
            continue
        if suffix in _DOC_SUFFIXES and "readme" not in rel.name.lower() and len(rel.parts) > 1:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > MAX_FILE_BYTES:
            continue
        out.append(rel)
    out.sort(key=lambda p: str(p).lower())
    return out


def _chunked(items: list, chunk_size: int) -> list[list]:
    """Split items into balanced chunks, each <= chunk_size, no tiny tails.

    >>> [len(c) for c in _chunked(list(range(11)), 5)]
    [4, 4, 3]
    >>> [len(c) for c in _chunked(list(range(6)), 5)]
    [3, 3]
    >>> [len(c) for c in _chunked(list(range(4)), 5)]
    [4]
    """
    n = len(items)
    if n == 0:
        return []
    if n <= chunk_size:
        return [list(items)]
    num_chunks = (n + chunk_size - 1) // chunk_size
    base, extra = divmod(n, num_chunks)
    out: list[list] = []
    i = 0
    for k in range(num_chunks):
        size = base + (1 if k < extra else 0)
        out.append(list(items[i:i + size]))
        i += size
    return out


def _file_to_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        return notebook_to_text(path)
    suffix = path.suffix.lstrip(".") or "txt"
    try:
        body = path.read_text(errors="replace")
    except OSError:
        return ""
    return f"```{suffix}\n{body}\n```"


def _format_size(n: int) -> str:
    """Human-readable file size for the tree shown to the selection model.

    >>> _format_size(0)
    '0B'
    >>> _format_size(900)
    '900B'
    >>> _format_size(2048)
    '2.0K'
    >>> _format_size(5 * 1024 * 1024)
    '5.0M'
    """
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}M"
    if n >= 1024:
        return f"{n / 1024:.1f}K"
    return f"{n}B"


def _format_tree(clone_dir: Path, files: list[Path]) -> str:
    """Render eligible files as `path  size` lines for the selection prompt."""
    lines: list[str] = []
    for rel in files:
        try:
            size = (clone_dir / rel).stat().st_size
        except OSError:
            size = 0
        lines.append(f"{rel}  {_format_size(size)}")
    return "\n".join(lines)


def _filter_to_known(requested: list[str], known: set[str]) -> list[str]:
    """Drop hallucinated / duplicate paths and normalize separators."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in requested or []:
        if not isinstance(raw, str):
            continue
        norm = raw.strip().lstrip("./").replace("\\", "/")
        if norm in known and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _build_orientation_excerpt(
    clone_dir: Path,
    requested: list[str],
    max_chars: int,
) -> str:
    parts: list[str] = []
    for rel_str in requested:
        body = _file_to_text(clone_dir / Path(rel_str))
        if not body.strip():
            continue
        parts.append(f"--- {rel_str} ---\n{body[:max_chars]}")
    return "\n\n".join(parts) or "(no readable orientation files)"


def select_top_files_with_ollama(
    target: dict,
    clone_dir: Path,
    files: list[Path],
    repo_dir: Path,
    max_files: int,
    max_requests: int = MAX_SELECTION_REQUESTS,
    body_chars: int = MAX_SELECTION_BODY_CHARS,
    force: bool = False,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> tuple[list[Path], dict]:
    """Two-round LLM-driven file selection.

    Round 1: model sees the file tree, requests orientation files
             (READMEs, entrypoint scripts, configs).
    Round 2: model sees the tree + the requested files' bodies, returns
             the top-N paths to deeply analyze.

    Returns (selected_paths, trace). Trace is also persisted under
    ``repo_dir/selection/`` for later inspection.
    """
    known = {str(p).replace("\\", "/") for p in files}
    tree = _format_tree(clone_dir, files)
    repo_label = f"{target['owner']}/{target['repo']}"
    sel_dir = repo_dir / "selection"

    cached = None if force else _load_valid_json(sel_dir / "round_002_selection.json")
    if cached is not None:
        cached_chosen = _filter_to_known(cached.get("top_files", []), known)[:max_files]
        if len(cached_chosen) >= max(1, max_files // 2):
            print(f"    selection cached: {len(cached_chosen)} file(s) reused")
            return [Path(p) for p in cached_chosen], {
                "cached": True,
                "selection": cached,
                "selected_filtered": cached_chosen,
            }

    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "tree.txt").write_text(tree)

    request_prompt = load_prompt(
        PROMPTS_DIR,
        "request_files",
        repo=repo_label,
        url=target["url"],
        max_files=str(max_files),
        max_requests=str(max_requests),
        tree=tree,
    )
    request = ollama_call(
        request_prompt, FILE_REQUEST_SCHEMA, SELECT_TIMEOUT, model=model, url=url,
    )
    (sel_dir / "round_001_request.json").write_text(json.dumps(request, indent=2))
    requested = _filter_to_known(request.get("files_to_read", []), known)[:max_requests]
    print(f"    selection round 1: model requests {len(requested)} orientation file(s)")

    file_bodies = _build_orientation_excerpt(clone_dir, requested, body_chars)

    select_prompt = load_prompt(
        PROMPTS_DIR,
        "select_top_files",
        repo=repo_label,
        url=target["url"],
        max_files=str(max_files),
        max_files_min=str(max(1, max_files // 2)),
        tree=tree,
        file_bodies=file_bodies,
    )
    selection = ollama_call(
        select_prompt, TOP_FILES_SCHEMA, SELECT_TIMEOUT, model=model, url=url,
    )
    (sel_dir / "round_002_selection.json").write_text(json.dumps(selection, indent=2))
    chosen = _filter_to_known(selection.get("top_files", []), known)[:max_files]
    print(f"    selection round 2: {len(chosen)} file(s) chosen")

    trace = {
        "request": request,
        "requested_filtered": requested,
        "selection": selection,
        "selected_filtered": chosen,
    }
    return [Path(p) for p in chosen], trace


def analyze_file_with_ollama(
    repo: str,
    rel_path: Path,
    body: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    prompt = load_prompt(
        PROMPTS_DIR,
        "analyze_file",
        repo=repo,
        path=str(rel_path),
        body_excerpt=body[:MAX_FILE_BODY_CHARS],
    )
    return ollama_call(
        prompt, FILE_SCHEMA, ANALYZE_TIMEOUT, model=model, url=url
    )


_FILE_PAYLOAD_FIELDS = [
    "summary", "purpose", "competition_flow", "data_reading", "data_processing",
    "features_engineering", "models", "frameworks_used", "loss_functions",
    "cv_strategy", "ensembling", "insights", "critical_findings",
    "what_did_not_work",
]
_FILE_STRING_FIELDS = {
    "summary", "purpose", "competition_flow", "cv_strategy", "ensembling",
}


def aggregate_repo_with_ollama(
    repo_label: str,
    repo_url: str,
    file_analyses: list[dict],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    payload: list[dict] = []
    for entry in file_analyses:
        analysis = entry.get("analysis", {})
        if "error" in analysis:
            continue
        item: dict = {"path": entry.get("path")}
        for field in _FILE_PAYLOAD_FIELDS:
            default = "" if field in _FILE_STRING_FIELDS else []
            item[field] = analysis.get(field, default)
        payload.append(item)
    prompt = load_prompt(
        PROMPTS_DIR,
        "aggregate_repo",
        repo=repo_label,
        url=repo_url,
        payload_json=json.dumps(payload, indent=2),
    )
    return ollama_call(
        prompt, REPO_SCHEMA, AGGREGATE_TIMEOUT, model=model, url=url,
    )


_REPO_PAYLOAD_FIELDS = [
    "summary", "competition_flow", "data_reading", "data_processing",
    "features_engineering", "models", "frameworks_used", "loss_functions",
    "cv_strategy", "ensembling", "insights", "critical_findings",
    "what_did_not_work", "training_setup",
]
_REPO_STRING_FIELDS = {
    "summary", "competition_flow", "cv_strategy", "ensembling",
}


def merge_repo_chunks_with_ollama(
    repo_label: str,
    repo_url: str,
    partial_summaries: list[dict],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    """Merge several partial REPO_SCHEMA summaries (from one repo) into one."""
    payload: list[dict] = []
    for partial in partial_summaries:
        if "error" in partial:
            continue
        item: dict = {}
        for field in _REPO_PAYLOAD_FIELDS:
            default = "" if field in _REPO_STRING_FIELDS else []
            item[field] = partial.get(field, default)
        item["notable_files"] = partial.get("notable_files", [])
        payload.append(item)
    prompt = load_prompt(
        PROMPTS_DIR,
        "merge_repo_chunks",
        repo=repo_label,
        url=repo_url,
        payload_json=json.dumps(payload, indent=2),
    )
    return ollama_call(
        prompt, REPO_SCHEMA, AGGREGATE_TIMEOUT, model=model, url=url,
    )


def hierarchical_aggregate_repo(
    repo_label: str,
    repo_url: str,
    file_analyses: list[dict],
    chunk_size: int,
    repo_dir: Path,
    force: bool = False,
    model: str = OLLAMA_MODEL,
) -> dict:
    """Tree-aggregate file analyses into a single REPO_SCHEMA summary.

    L1: chunks of file analyses -> partial REPO_SCHEMA, persisted as
        repo_dir/L1/chunk_NNN.json.
    L2+: chunks of partial REPO_SCHEMA -> merged REPO_SCHEMA, persisted as
        repo_dir/LN/chunk_NNN.json. Recurses until one summary remains.
    """
    successful = [f for f in file_analyses if "error" not in f.get("analysis", {})]
    if not successful:
        return {"error": "no per-file analyses succeeded"}

    chunks = _chunked(successful, chunk_size)
    l1_dir = repo_dir / "L1"
    l1_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"    L1: {len(successful)} files -> {len(chunks)} chunk"
        f"{'s' if len(chunks) != 1 else ''} of <= {chunk_size}"
    )
    summaries: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        chunk_path = l1_dir / f"chunk_{i:03d}.json"
        cached = None if force else _load_valid_json(chunk_path)
        if cached is not None:
            print(f"      [L1 {i}/{len(chunks)}] cached")
            summaries.append(cached)
            continue
        print(f"      [L1 {i}/{len(chunks)}] aggregating {len(chunk)} files")
        try:
            partial = aggregate_repo_with_ollama(
                repo_label, repo_url, chunk, model=model
            )
        except Exception as exc:
            partial = {"error": f"{type(exc).__name__}: {exc}"}
        chunk_path.write_text(json.dumps(partial, indent=2))
        summaries.append(partial)

    level = 2
    while len(summaries) > 1:
        valid = [s for s in summaries if "error" not in s]
        if not valid:
            return {"error": "all chunk summaries failed"}
        chunks = _chunked(valid, chunk_size)
        ln_dir = repo_dir / f"L{level}"
        ln_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"    L{level}: {len(valid)} partial summaries -> {len(chunks)} "
            f"chunk{'s' if len(chunks) != 1 else ''}"
        )
        merged: list[dict] = []
        for i, chunk in enumerate(chunks, 1):
            chunk_path = ln_dir / f"chunk_{i:03d}.json"
            cached = None if force else _load_valid_json(chunk_path)
            if cached is not None:
                print(f"      [L{level} {i}/{len(chunks)}] cached")
                merged.append(cached)
                continue
            if len(chunk) == 1:
                step = chunk[0]
            else:
                print(f"      [L{level} {i}/{len(chunks)}] merging {len(chunk)} summaries")
                try:
                    step = merge_repo_chunks_with_ollama(
                        repo_label, repo_url, chunk, model=model
                    )
                except Exception as exc:
                    step = {"error": f"{type(exc).__name__}: {exc}"}
            chunk_path.write_text(json.dumps(step, indent=2))
            merged.append(step)
        summaries = merged
        level += 1

    return summaries[0]


def aggregate_repos_with_ollama(
    repo_summaries: list[dict],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    payload: list[dict] = []
    for entry in repo_summaries:
        summary = entry.get("summary", {})
        if "error" in summary:
            continue
        item: dict = {
            "owner_repo": f"{entry.get('owner', '')}/{entry.get('repo', '')}",
            "url": entry.get("url"),
            "rank": entry.get("rank"),
            "title": entry.get("title"),
        }
        for field in _REPO_PAYLOAD_FIELDS:
            default = "" if field in _REPO_STRING_FIELDS else []
            item[field] = summary.get(field, default)
        payload.append(item)
    prompt = load_prompt(
        PROMPTS_DIR,
        "aggregate_repos",
        payload_json=json.dumps(payload, indent=2),
    )
    return ollama_call(
        prompt, AGGREGATED_REPOS_SCHEMA, AGGREGATE_TIMEOUT, model=model, url=url,
    )


def _section_md(lines: list[str], title: str, items: list[str]) -> None:
    if not items:
        return
    lines.append(f"## {title}")
    lines.extend(f"- {x}" for x in items)
    lines.append("")


def _render_repo_markdown(
    target: dict,
    summary: dict,
    file_analyses: list[dict],
) -> str:
    lines = [f"# {target['owner']}/{target['repo']}", ""]
    lines.append(f"- **URL:** {target['url']}")
    if target.get("title"):
        lines.append(f"- **Source solution:** {target['title']}")
    if target.get("role"):
        lines.append(f"- **Role:** {target['role']}")
    if isinstance(target.get("rank"), int):
        lines.append(f"- **Rank:** #{target['rank']}")
    lines.extend(["", "---", ""])
    lines.extend([summary.get("summary", ""), ""])

    flow = summary.get("competition_flow", "")
    if flow:
        lines.extend(["## Competition flow", flow, ""])

    _section_md(lines, "Models", summary.get("models", []))
    _section_md(lines, "Frameworks used", summary.get("frameworks_used", []))
    _section_md(lines, "Loss functions", summary.get("loss_functions", []))
    _section_md(lines, "Data reading", summary.get("data_reading", []))
    _section_md(lines, "Data processing", summary.get("data_processing", []))
    _section_md(
        lines, "Features engineering", summary.get("features_engineering", [])
    )
    cv = summary.get("cv_strategy", "")
    if cv:
        lines.extend(["## CV strategy", cv, ""])
    ens = summary.get("ensembling", "")
    if ens:
        lines.extend(["## Ensembling", ens, ""])
    _section_md(lines, "Training setup", summary.get("training_setup", []))
    _section_md(lines, "Insights", summary.get("insights", []))
    _section_md(lines, "Critical findings", summary.get("critical_findings", []))
    _section_md(lines, "What did not work", summary.get("what_did_not_work", []))

    notable = summary.get("notable_files", [])
    if notable:
        lines.append("## Notable files")
        for note in notable:
            lines.append(f"- `{note.get('path', '')}` — {note.get('role', '')}")
        lines.append("")

    lines.append("## Files analyzed")
    for entry in file_analyses:
        path = entry.get("path", "")
        analysis = entry.get("analysis", {}) or {}
        purpose = analysis.get("purpose", "")
        first_summary = (analysis.get("summary") or "").split("\n", 1)[0]
        suffix = f" — {first_summary}" if first_summary else ""
        purpose_str = f" _({purpose})_" if purpose else ""
        lines.append(f"- `{path}`{purpose_str}{suffix}")
    lines.append("")
    return "\n".join(lines)


def _render_aggregated_markdown(
    slug: str,
    agg: dict,
    repo_summaries: list[dict],
) -> str:
    lines = [f"# {slug}: cross-repo summary", ""]
    lines.extend([agg.get("overall_summary", ""), ""])

    _section_md(lines, "Competition flows", agg.get("competition_flows", []))
    _section_md(lines, "Data reading", agg.get("data_reading", []))
    _section_md(lines, "Data processing", agg.get("data_processing", []))
    _section_md(
        lines, "Features engineering", agg.get("features_engineering", [])
    )
    _section_md(lines, "Models", agg.get("models", []))
    _section_md(lines, "Frameworks used", agg.get("frameworks_used", []))
    _section_md(lines, "Loss functions", agg.get("loss_functions", []))
    _section_md(lines, "CV strategies", agg.get("cv_strategies", []))
    _section_md(lines, "Ensembling", agg.get("ensembling", []))
    _section_md(lines, "Training setups", agg.get("training_setups", []))
    _section_md(lines, "Insights", agg.get("insights", []))
    _section_md(lines, "Critical findings", agg.get("critical_findings", []))
    _section_md(lines, "What did not work", agg.get("what_did_not_work", []))
    _section_md(
        lines,
        "Notable individual insights",
        agg.get("notable_individual_insights", []),
    )

    lines.append("## Repos indexed")
    for entry in sorted(
        repo_summaries,
        key=lambda x: x.get("rank") if isinstance(x.get("rank"), int) else 999,
    ):
        url = entry.get("url", "")
        title = entry.get("title", "")
        stem = entry.get("stem", "")
        owner = entry.get("owner", "")
        repo = entry.get("repo", "")
        rank = entry.get("rank")
        rank_str = f"#{rank} " if isinstance(rank, int) else ""
        from_str = f" — from _{title}_" if title else ""
        lines.append(
            f"- {rank_str}[[repos/{stem}/repo_analysis|{owner}/{repo}]] "
            f"({url}){from_str}"
        )
    lines.append("")
    return "\n".join(lines)


def _process_repo(
    target: dict,
    repo_root: Path,
    chunk_size: int,
    max_files: int,
    force: bool,
    model: str,
) -> dict | None:
    stem = _repo_stem(target["owner"], target["repo"])
    repo_dir = repo_root / stem
    repo_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = repo_dir / "clone"

    print(f"  cloning {target['url']}")
    if clone_repo(target["url"], clone_dir) is None:
        return None

    all_files = _walk_repo(clone_dir)
    print(f"    {len(all_files)} files eligible for analysis")
    if not all_files:
        return None

    if len(all_files) <= max_files:
        files = all_files
        print(f"    repo within budget; analyzing all {len(files)} file(s)")
    else:
        try:
            files, _ = select_top_files_with_ollama(
                target, clone_dir, all_files, repo_dir,
                max_files=max_files, force=force, model=model,
            )
        except Exception as exc:
            print(
                f"    file selection failed: {type(exc).__name__}: {exc}; "
                f"falling back to first {max_files} by tree order"
            )
            files = all_files[:max_files]
        if not files:
            print(
                f"    selection returned no files; "
                f"falling back to first {max_files} by tree order"
            )
            files = all_files[:max_files]

    l0_dir = repo_dir / "L0"
    l0_dir.mkdir(parents=True, exist_ok=True)
    file_analyses: list[dict] = []
    for i, rel in enumerate(files, 1):
        l0_path = l0_dir / f"{_flatten_path(rel)}.json"
        cached = None if force else _load_valid_json(l0_path)
        if cached is not None and "analysis" in cached:
            print(f"    [{i}/{len(files)}] {rel} (cached)")
            file_analyses.append(cached)
            continue
        body = _file_to_text(clone_dir / rel)
        if not body.strip():
            continue
        print(f"    [{i}/{len(files)}] {rel}")
        try:
            analysis = analyze_file_with_ollama(
                f"{target['owner']}/{target['repo']}", rel, body, model=model
            )
        except Exception as exc:
            analysis = {"error": f"{type(exc).__name__}: {exc}"}
        record = {"path": str(rel), "analysis": analysis}
        l0_path.write_text(json.dumps(record, indent=2))
        file_analyses.append(record)

    meta = {**target, "files_total": len(all_files), "files_selected": len(files)}
    (repo_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    label = f"{target['owner']}/{target['repo']}"
    summary = hierarchical_aggregate_repo(
        label, target["url"], file_analyses, chunk_size, repo_dir,
        force=force, model=model,
    )
    (repo_dir / "repo_analysis.json").write_text(json.dumps(summary, indent=2))
    if "error" not in summary:
        md = _render_repo_markdown(target, summary, file_analyses)
        (repo_dir / "repo_analysis.md").write_text(md)
        print(f"    wrote {repo_dir}/repo_analysis.md")
    else:
        print(f"    repo aggregation skipped: {summary['error']}")

    return {**target, "stem": stem, "summary": summary}


def cmd_repos(args: argparse.Namespace) -> int:
    data_root = getattr(args, "data_root", DEFAULT_OUTPUT_ROOT)
    out = competition_dir(args.slug, data_root)
    roles_arg = getattr(args, "roles", list(DEFAULT_ROLES))
    roles = tuple(roles_arg) if roles_arg else ()
    chunk_size = getattr(args, "chunk_size", CHUNK_SIZE)
    max_files = getattr(args, "max_files", MAX_FILES_PER_REPO)
    force = getattr(args, "force", False)
    model = getattr(args, "model", OLLAMA_MODEL)

    targets = _load_targets(out, roles)
    role_label = ",".join(roles) if roles else "any"
    print(f"Selected {len(targets)} repos to analyze (roles={role_label})")
    if not targets:
        return 0

    repo_root = out / "repos"
    repo_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    new_repo_count = 0
    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {target['owner']}/{target['repo']}")
        stem = _repo_stem(target["owner"], target["repo"])
        repo_dir = repo_root / stem
        cached_summary = None if force else _load_valid_json(repo_dir / "repo_analysis.json")
        if cached_summary is not None:
            print(f"  reusing existing {stem}/repo_analysis.json")
            summaries.append({**target, "stem": stem, "summary": cached_summary})
            continue
        result = _process_repo(
            target, repo_root, chunk_size, max_files, force, model,
        )
        if result is not None:
            summaries.append(result)
            new_repo_count += 1

    print(
        f"\nProcessed {len(summaries)} repos for {args.slug} "
        f"({new_repo_count} new, {len(summaries) - new_repo_count} reused)"
    )
    if not summaries:
        return 0

    agg_path = out / "aggregated_repos.json"
    if not force and new_repo_count == 0:
        cached_agg = _load_valid_json(agg_path)
        if cached_agg is not None:
            print(f"  aggregated_repos.json up to date; skipping aggregation")
            return 0

    print(f"Aggregating {len(summaries)} repos with {model}...")
    try:
        agg = aggregate_repos_with_ollama(summaries, model=model)
    except Exception as exc:
        agg = {"error": f"{type(exc).__name__}: {exc}"}
    agg_path.write_text(json.dumps(agg, indent=2))
    if "error" not in agg:
        md = _render_aggregated_markdown(args.slug, agg, summaries)
        (out / "aggregated_repos.md").write_text(md)
        print(f"Wrote aggregated repos to {out}/aggregated_repos.md")
    else:
        print(f"Aggregation failed: {agg['error']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akc-repos")
    parser.add_argument("slug")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Files merged per ollama call at each aggregation level")
    parser.add_argument("--max-files", type=int, default=MAX_FILES_PER_REPO,
                        help=("Cap on files deep-analyzed per repo. Repos with more "
                              "files are first reduced via two-round LLM selection."))
    parser.add_argument("--force", action="store_true",
                        help=("Re-run every stage (selection, file analyses, L1+ "
                              "aggregation) even if cached artefacts exist"))
    parser.add_argument(
        "--roles", nargs="*", default=list(DEFAULT_ROLES),
        help=(
            "Repo roles to keep (solution|library|reference). "
            "Pass --roles with no values to disable filtering."
        ),
    )
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    return cmd_repos(args)


if __name__ == "__main__":
    raise SystemExit(main())
