from __future__ import annotations

import argparse
import json
import sys

from aikaggler.plugins.knowledge_base_mcp import core


_LIST_FILTER_KEYS = {
    "slugs", "data_modality", "task_type", "label_structure", "test_split",
    "format", "dataset_scale", "domain", "data_challenges", "constraints",
}
_INT_FILTER_KEYS = {"min_teams", "min_prizes"}


def _parse_kv_pairs(items: list[str]) -> dict:
    out: dict = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--filter arg must be key=value, got {raw!r}")
        key, value = raw.split("=", 1)
        if key in _LIST_FILTER_KEYS:
            out[key] = value.split(",") if value else []
        elif key in _INT_FILTER_KEYS:
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _emit(payload) -> int:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akb",
        description="Debug CLI for the aikaggler knowledge-base MCP service.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("--filter", action="append", default=[],
                   help="repeatable key=value or key=v1,v2 filter")
    p.add_argument("--sort", default="date_enabled_desc")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("get")
    p.add_argument("slug")

    p = sub.add_parser("tags")
    p.add_argument("--field")
    p.add_argument("--with-slugs", action="store_true")

    p = sub.add_parser("aggregate")
    p.add_argument("field")
    p.add_argument("--source", default="solutions", choices=["notebooks", "solutions"])
    p.add_argument("--filter", action="append", default=[])
    p.add_argument("--flatten", action="store_true")

    p = sub.add_parser("compare")
    p.add_argument("slugs", nargs="+")
    p.add_argument("--source", default="solutions", choices=["notebooks", "solutions"])
    p.add_argument("--fields", nargs="*")

    p = sub.add_parser("get-aggregated")
    p.add_argument("slug")
    p.add_argument("--source", default="solutions", choices=["notebooks", "solutions"])
    p.add_argument("--fields", nargs="*")

    p = sub.add_parser("notebooks")
    p.add_argument("slug")

    p = sub.add_parser("solutions")
    p.add_argument("slug")

    p = sub.add_parser("notebook")
    p.add_argument("slug")
    p.add_argument("dir_name")

    p = sub.add_parser("solution")
    p.add_argument("slug")
    p.add_argument("dir_name")

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--source", choices=["notebooks", "solutions"])
    p.add_argument("--fields", nargs="*")
    p.add_argument("--slugs", nargs="*")
    p.add_argument("--limit", type=int, default=100)

    p = sub.add_parser("github")
    p.add_argument("--slugs", nargs="*")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        return _emit(core.list_competitions(_parse_kv_pairs(args.filter), args.sort, args.limit))
    if args.cmd == "get":
        return _emit(core.get_competition(args.slug))
    if args.cmd == "tags":
        return _emit(core.list_tags(args.field, args.with_slugs))
    if args.cmd == "aggregate":
        return _emit(core.aggregate_field(args.field, args.source,
                                          _parse_kv_pairs(args.filter), args.flatten))
    if args.cmd == "compare":
        return _emit(core.compare_competitions(args.slugs, args.fields, args.source))
    if args.cmd == "get-aggregated":
        return _emit(core.get_aggregated(args.slug, args.source, args.fields))
    if args.cmd == "notebooks":
        return _emit(core.list_notebooks(args.slug))
    if args.cmd == "solutions":
        return _emit(core.list_solutions(args.slug))
    if args.cmd == "notebook":
        return _emit(core.get_notebook(args.slug, args.dir_name))
    if args.cmd == "solution":
        return _emit(core.get_solution(args.slug, args.dir_name))
    if args.cmd == "search":
        return _emit(core.search(args.query, args.source, args.fields, args.slugs, args.limit))
    if args.cmd == "github":
        return _emit(core.list_github_links(args.slugs))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
