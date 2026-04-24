from __future__ import annotations

import argparse
import json
from pathlib import Path

from aikaggler.plugins._shared.ollama import load_prompt, ollama_call
from aikaggler.plugins.code_analysis.cli import cmd_notebooks
from aikaggler.plugins.code_analysis.config import TOP_NOTEBOOKS
from aikaggler.plugins.competition_analysis.config import (
    ANALYZE_TIMEOUT,
    DEFAULT_OUTPUT_ROOT,
    MAX_SOLUTION_BODY_CHARS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROMPTS_DIR,
)
from aikaggler.plugins.competition_analysis.structure import (
    COMPETITION_CLASSIFICATION_SCHEMA,
)
from aikaggler.plugins.solution_analysis.cli import cmd_solutions, competition_dir


def _collect_solution_titles(out: Path) -> list[str]:
    titles: list[str] = []
    solutions_dir = out / "solutions"
    if not solutions_dir.is_dir():
        return titles
    for analysis_path in sorted(solutions_dir.glob("*/analysis.json")):
        try:
            data = json.loads(analysis_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        title = analysis_path.parent.name
        solution_md = analysis_path.parent / "solution.md"
        if solution_md.exists():
            first = solution_md.read_text().splitlines()[0].lstrip("# ").strip()
            if first:
                title = first
        titles.append(title)
    return titles


def classify_competition_with_ollama(
    comp: dict,
    solution_titles: list[str],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    slim_keys = [
        "title", "briefDescription", "description", "evaluationAlgorithm",
        "categories", "hostName", "organization", "totalCompetitors",
        "totalTeams", "maxDailySubmissions", "license",
    ]
    slim = {k: comp.get(k) for k in slim_keys if comp.get(k) is not None}
    prompt = load_prompt(
        PROMPTS_DIR,
        "classify_competition",
        competition_json=json.dumps(slim, indent=2)[:MAX_SOLUTION_BODY_CHARS],
        solution_titles_json=json.dumps(solution_titles, indent=2),
    )
    return ollama_call(
        prompt,
        COMPETITION_CLASSIFICATION_SCHEMA,
        ANALYZE_TIMEOUT,
        model=model,
        url=url,
    )


def cmd_competition(args: argparse.Namespace) -> int:
    out = competition_dir(args.slug, args.data_root)

    print(f"\n[1/3] Analyzing solutions for {args.slug}")
    sol_args = argparse.Namespace(
        slug=args.slug,
        limit=args.limit,
        pages=args.pages,
        model=args.model,
    )
    try:
        cmd_solutions(sol_args)
    except Exception as e:
        print(f"Solutions analysis failed: {type(e).__name__}: {e}")

    print(f"\n[2/3] Analyzing notebooks for {args.slug}")
    nb_args = argparse.Namespace(
        slug=args.slug,
        top=args.top,
        model=args.model,
    )
    try:
        cmd_notebooks(nb_args)
    except Exception as e:
        print(f"Notebooks analysis failed: {type(e).__name__}: {e}")

    print(f"\n[3/3] Classifying {args.slug}")
    comp_path = out / "competition.json"
    if not comp_path.exists():
        print(f"Skipping classification: {comp_path} not found")
        return 1
    comp = json.loads(comp_path.read_text())
    titles = _collect_solution_titles(out)
    try:
        classification = classify_competition_with_ollama(
            comp, titles, model=args.model
        )
    except Exception as e:
        classification = {"error": f"{type(e).__name__}: {e}"}
    (out / "competition.classification.json").write_text(
        json.dumps(classification, indent=2)
    )
    if "error" not in classification:
        print(
            f"Classified: {classification.get('data_modality')}/"
            f"{classification.get('task_type')} "
            f"({classification.get('domain')})"
        )
    else:
        print(f"Classification failed: {classification['error']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_comp = sub.add_parser(
        "competition",
        help="Full pipeline: solutions + notebooks + classification",
    )
    p_comp.add_argument("slug")
    p_comp.add_argument("--limit", type=int, default=20,
                        help="Max solutions per competition")
    p_comp.add_argument("--pages", type=int, default=5,
                        help="Forum pages to scan per competition")
    p_comp.add_argument("--top", type=int, default=TOP_NOTEBOOKS,
                        help="Top-N notebooks to pull")
    p_comp.add_argument("--model", default=OLLAMA_MODEL)
    p_comp.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_comp.set_defaults(func=cmd_competition)

    p_sol = sub.add_parser("solutions", help="Solutions pipeline only")
    p_sol.add_argument("slug")
    p_sol.add_argument("--limit", type=int, default=20)
    p_sol.add_argument("--pages", type=int, default=5)
    p_sol.add_argument("--model", default=OLLAMA_MODEL)
    p_sol.set_defaults(func=cmd_solutions)

    p_nb = sub.add_parser("notebooks", help="Notebooks pipeline only")
    p_nb.add_argument("slug")
    p_nb.add_argument("--top", type=int, default=TOP_NOTEBOOKS)
    p_nb.add_argument("--model", default=OLLAMA_MODEL)
    p_nb.set_defaults(func=cmd_notebooks)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
