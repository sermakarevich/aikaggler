"""Entry point: run full competition pipeline on every completed money-based Kaggle competition.

Pipeline per competition:
  solutions analysis → notebooks analysis → github repo analysis → classification.

Discovery pipeline:
1. List competitions via the kaggle CLI (paginated CSV).
2. Keep only USD-prize competitions whose deadline has already passed.
3. Sort by end date, most recent first.
4. Skip competitions that already have a data/<slug>/ folder.
5. Run cmd_competition on every remaining competition.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aikaggler.plugins.code_analysis.config import TOP_NOTEBOOKS
from aikaggler.plugins.competition_analysis.cli import cmd_competition
from aikaggler.plugins.github_analysis.config import (
    CHUNK_SIZE,
    DEFAULT_ROLES,
    MAX_FILES_PER_REPO,
)
from aikaggler.plugins.solution_analysis.config import DEFAULT_OUTPUT_ROOT, OLLAMA_MODEL


@dataclass
class Competition:
    slug: str
    deadline: datetime
    reward: str
    category: str
    team_count: str


def _fetch_page(page: int, page_size: int, retries: int = 3) -> list[dict[str, str]]:
    cmd = [
        "kaggle", "competitions", "list",
        "--category", "all",
        "--sort-by", "latestDeadline",
        "--page-size", str(page_size),
        "-p", str(page),
        "-v",
    ]
    for attempt in range(1, retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return list(csv.DictReader(io.StringIO(result.stdout)))
        if attempt < retries:
            time.sleep(2 * attempt)
    print(
        f"WARN: kaggle listing page {page} failed after {retries} attempts: "
        f"{result.stderr.strip() or 'no stderr'}"
    )
    return []


def _is_money_reward(reward: str) -> bool:
    return "Usd" in reward


def _slug_from_ref(ref: str) -> str:
    return ref.rstrip("/").rsplit("/", 1)[-1]


def _parse_deadline(deadline: str) -> datetime:
    return datetime.strptime(deadline, "%Y-%m-%d %H:%M:%S")


def collect_competitions(max_pages: int, page_size: int = 200) -> list[Competition]:
    """Return completed USD-prize competitions sorted by deadline descending."""
    now = datetime.now()
    seen: set[str] = set()
    out: list[Competition] = []
    empty_streak = 0
    for page in range(1, max_pages + 1):
        batch = _fetch_page(page, page_size)
        if not batch:
            empty_streak += 1
            if empty_streak >= 2:
                break
            continue
        empty_streak = 0
        for row in batch:
            slug = _slug_from_ref(row["ref"])
            if slug in seen:
                continue
            seen.add(slug)
            if not _is_money_reward(row.get("reward", "")):
                continue
            try:
                deadline = _parse_deadline(row["deadline"])
            except (KeyError, ValueError):
                continue
            if deadline >= now:
                continue
            out.append(
                Competition(
                    slug=slug,
                    deadline=deadline,
                    reward=row["reward"],
                    category=row.get("category", ""),
                    team_count=row.get("teamCount", ""),
                )
            )
    out.sort(key=lambda c: c.deadline, reverse=True)
    return out


def _already_processed(slug: str, data_root: Path) -> bool:
    return (data_root / slug).is_dir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=100,
                        help="Max kaggle listing pages to scan")
    parser.add_argument("--limit", type=int, default=20,
                        help="Solutions to analyze per competition")
    parser.add_argument("--topic-pages", type=int, default=5,
                        help="Forum pages to scan per competition")
    parser.add_argument("--top", type=int, default=TOP_NOTEBOOKS,
                        help="Top-N notebooks to pull per competition")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Files merged per ollama call at each repo aggregation level")
    parser.add_argument("--max-files", type=int, default=MAX_FILES_PER_REPO,
                        help=("Cap on files deep-analyzed per repo (selection step "
                              "kicks in for repos with more files than this)."))
    parser.add_argument(
        "--roles", nargs="*", default=list(DEFAULT_ROLES),
        help=(
            "GitHub repo roles to analyze (solution|library|reference). "
            "Pass --roles with no values to disable filtering."
        ),
    )
    parser.add_argument("--skip-repos", action="store_true",
                        help="Skip GitHub repo analysis step")
    parser.add_argument("--force", action="store_true",
                        help=("Re-run every stage of every competition even if "
                              "cached artefacts exist (bypasses the outer "
                              "data/<slug>/ skip)"))
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-competitions", type=int, default=0,
                        help="Stop after processing N competitions (0 = no cap)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List pending competitions and exit")
    args = parser.parse_args(argv)

    comps = collect_competitions(args.pages)
    if args.force:
        pending = list(comps)
        print(
            f"Found {len(comps)} completed USD-prize competitions "
            f"(--force: all {len(pending)} queued, every per-stage cache will "
            f"be ignored)"
        )
    else:
        pending = [c for c in comps if not _already_processed(c.slug, args.data_root)]
        skipped = len(comps) - len(pending)
        print(
            f"Found {len(comps)} completed USD-prize competitions "
            f"({skipped} already in {args.data_root}, {len(pending)} pending)"
        )

    if args.max_competitions > 0:
        pending = pending[: args.max_competitions]

    for i, c in enumerate(pending, 1):
        print(f"  {i:3d}. {c.deadline:%Y-%m-%d} {c.reward:<20} {c.slug}")

    if args.dry_run or not pending:
        return 0

    failures: list[tuple[str, str]] = []
    for i, c in enumerate(pending, 1):
        print(
            f"\n=== [{i}/{len(pending)}] {c.slug} "
            f"({c.deadline:%Y-%m-%d}, {c.reward}) ==="
        )
        sub_args = argparse.Namespace(
            slug=c.slug,
            limit=args.limit,
            pages=args.topic_pages,
            top=args.top,
            chunk_size=args.chunk_size,
            max_files=args.max_files,
            force=args.force,
            roles=args.roles,
            skip_repos=args.skip_repos,
            model=args.model,
            data_root=args.data_root,
        )
        try:
            cmd_competition(sub_args)
        except KeyboardInterrupt:
            print(f"\nInterrupted during {c.slug}")
            return 130
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {c.slug}: {msg}")
            failures.append((c.slug, msg))

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for slug, msg in failures:
            print(f"  {slug}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
