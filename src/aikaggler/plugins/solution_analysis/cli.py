from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from aikaggler.plugins._shared.ollama import load_prompt as _load_prompt, ollama_call
from aikaggler.plugins.solution_analysis.config import (
    AGGREGATE_TIMEOUT,
    ANALYZE_TIMEOUT,
    DEFAULT_OUTPUT_ROOT,
    KAGGLE_API,
    MAX_SOLUTION_BODY_CHARS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROMPTS_DIR,
    USER_AGENT,
)
from aikaggler.plugins.solution_analysis.structure import (
    AGGREGATED_SCHEMA,
    FILTER_SCHEMA,
    GITHUB_URL_RE,
    NON_REPO_OWNERS,
    SOLUTION_SCHEMA,
)


def load_prompt(name: str, **kwargs: str) -> str:
    return _load_prompt(PROMPTS_DIR, name, **kwargs)


def competition_dir(slug: str, root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    path = root / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


class KaggleRPC:
    """Thin httpx client for Kaggle's internal Twirp-style RPC endpoints."""

    def __init__(self, slug: str) -> None:
        self.slug = slug
        self.client = httpx.Client(
            follow_redirects=True, timeout=30, headers={"User-Agent": USER_AGENT}
        )
        self.client.get(f"https://www.kaggle.com/competitions/{slug}/discussion")
        self.headers = {
            "x-xsrf-token": self.client.cookies.get("XSRF-TOKEN"),
            "content-type": "application/json",
            "accept": "application/json",
            "origin": "https://www.kaggle.com",
            "referer": f"https://www.kaggle.com/competitions/{slug}/discussion",
        }

    def __enter__(self) -> "KaggleRPC":
        return self

    def __exit__(self, *_: object) -> None:
        self.client.close()

    def post(self, path: str, body: dict) -> dict:
        r = self.client.post(
            f"{KAGGLE_API}/{path}", json=body, headers=self.headers
        )
        r.raise_for_status()
        return r.json()

    def competition(self) -> dict:
        return self.post(
            "competitions.CompetitionService/GetCompetition",
            {"competitionName": self.slug},
        )

    def topic_page(self, forum_id: int, page: int) -> list[dict]:
        body = {
            "category": "TOPIC_LIST_CATEGORY_ALL",
            "group": "TOPIC_LIST_GROUP_ALL",
            "customGroupingIds": [],
            "author": "TOPIC_LIST_AUTHOR_UNSPECIFIED",
            "myActivity": "TOPIC_LIST_MY_ACTIVITY_UNSPECIFIED",
            "recency": "TOPIC_LIST_RECENCY_UNSPECIFIED",
            "filterCategoryIds": [],
            "searchQuery": "",
            "sortBy": "TOPIC_LIST_SORT_BY_TOP",
            "page": page,
            "forumId": forum_id,
        }
        return self.post(
            "discussions.DiscussionsService/GetTopicListByForumId", body
        ).get("topics", [])

    def topic(self, topic_id: int) -> dict:
        return self.post(
            "discussions.DiscussionsService/GetForumTopicById",
            {"forumTopicId": topic_id, "includeComments": True},
        )


def ask_ollama_to_filter(
    topics: list[dict], model: str = OLLAMA_MODEL, url: str = OLLAMA_URL
) -> list[dict]:
    slim = [
        {"id": t.get("id"), "title": t.get("title"), "votes": t.get("votes")}
        for t in topics
    ]
    prompt = load_prompt(
        "filter_topics", topics_json=json.dumps(slim, indent=2)
    )
    result = ollama_call(
        prompt, FILTER_SCHEMA, ANALYZE_TIMEOUT, model=model, url=url
    )
    return result.get("solutions", [])


def analyze_solution_with_ollama(
    title: str,
    body: str,
    regex_links: list[dict],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    prompt = load_prompt(
        "analyze_solution",
        title=title,
        regex_urls_json=json.dumps([l["url"] for l in regex_links]),
        body_excerpt=body[:MAX_SOLUTION_BODY_CHARS],
    )
    return ollama_call(
        prompt, SOLUTION_SCHEMA, ANALYZE_TIMEOUT, model=model, url=url
    )


_SOLUTION_PAYLOAD_FIELDS = [
    "summary", "competition_flow", "data_reading", "data_processing",
    "features_engineering", "models", "frameworks_used", "loss_functions",
    "cv_strategy", "ensembling", "insights", "critical_findings",
    "what_did_not_work",
]


def aggregate_analyses_with_ollama(
    analyses: list[dict],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    payload = []
    for a in analyses:
        analysis = a.get("analysis", {})
        if "error" in analysis:
            continue
        entry: dict = {"rank": analysis.get("rank"), "title": a["title"]}
        for field in _SOLUTION_PAYLOAD_FIELDS:
            entry[field] = analysis.get(field, [] if field not in {
                "summary", "competition_flow", "cv_strategy", "ensembling"
            } else "")
        payload.append(entry)
    prompt = load_prompt(
        "aggregate_analyses", payload_json=json.dumps(payload, indent=2)
    )
    return ollama_call(
        prompt, AGGREGATED_SCHEMA, AGGREGATE_TIMEOUT, model=model, url=url
    )


def extract_github_links(detail: dict) -> list[dict]:
    text = json.dumps(detail)
    seen: set[str] = set()
    found: list[dict] = []
    for match in GITHUB_URL_RE.finditer(text):
        owner = match.group(1)
        repo = match.group(2).rstrip(".").removesuffix(".git")
        if not repo or owner.lower() in NON_REPO_OWNERS:
            continue
        key = f"{owner}/{repo}".lower()
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "owner": owner,
            "repo": repo,
            "url": f"https://github.com/{owner}/{repo}",
        })
    return found


def _rank_stem(rank: int | None, topic_id: int) -> str:
    return f"rank_{rank:02d}" if isinstance(rank, int) else f"rank_xx_{topic_id}"


def _write_solution_files(
    out_dir: Path,
    topic_id: int,
    title: str,
    detail: dict,
    slug: str,
    model: str = OLLAMA_MODEL,
) -> tuple[list[dict], dict]:
    topic = detail.get("forumTopic", {})
    first = topic.get("firstMessage", {}) or {}
    author = first.get("author", {}).get("displayName") or topic.get(
        "authorUserDisplayName", "unknown"
    )
    date = first.get("postDate") or topic.get("postDate", "")
    url = f"https://www.kaggle.com/competitions/{slug}/discussion/{topic_id}"
    links = extract_github_links(detail)
    body = first.get("rawMarkdown", "") or "(no first message content)"

    try:
        analysis = analyze_solution_with_ollama(title, body, links, model=model)
    except Exception as e:
        analysis = {"error": f"{type(e).__name__}: {e}"}
    analysis["source_url"] = url

    stem = _rank_stem(analysis.get("rank"), topic_id)
    solution_dir = out_dir / "solutions" / stem
    solution_dir.mkdir(parents=True, exist_ok=True)

    links_md = (
        "\n\n**GitHub links found:**\n"
        + "\n".join(f"- {l['url']}" for l in links)
        + "\n"
    ) if links else ""
    header = (
        f"# {title}\n\n"
        f"- **Author:** {author}\n"
        f"- **Date:** {date}\n"
        f"- **Topic ID:** {topic_id}\n"
        f"- **URL:** {url}"
        f"{links_md}\n---\n\n"
    )
    (solution_dir / "solution.md").write_text(header + body)
    (solution_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))

    link_entries = [{"topic_id": topic_id, "title": title, **l} for l in links]
    analysis_entry = {"topic_id": topic_id, "title": title, "analysis": analysis}
    return link_entries, analysis_entry


def _render_aggregated_markdown(
    slug: str,
    agg: dict,
    analyses: list[dict],
    all_links: list[dict],
) -> str:
    lines = [f"# {slug}: cross-solution summary", ""]
    lines += [agg.get("overall_summary", ""), ""]

    def _section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.extend(f"- {x}" for x in items)
        lines.append("")

    _section("Competition flows", agg.get("competition_flows", []))
    _section("Data reading", agg.get("data_reading", []))
    _section("Data processing", agg.get("data_processing", []))
    _section("Features engineering", agg.get("features_engineering", []))
    _section("Models", agg.get("models", []))
    _section("Frameworks used", agg.get("frameworks_used", []))
    _section("Loss functions", agg.get("loss_functions", []))
    _section("CV strategies", agg.get("cv_strategies", []))
    _section("Ensembling", agg.get("ensembling", []))
    _section("Insights", agg.get("insights", []))
    _section("Critical findings", agg.get("critical_findings", []))
    _section("What did not work", agg.get("what_did_not_work", []))
    _section(
        "Notable individual insights",
        agg.get("notable_individual_insights", []),
    )

    lines.append("## Solutions indexed")
    for a in sorted(analyses, key=lambda x: x["analysis"].get("rank") or 999):
        rank = a["analysis"].get("rank")
        rank_str = f"#{rank}" if rank else "?"
        stem = _rank_stem(rank, a["topic_id"])
        lines.append(
            f"- {rank_str} [[solutions/{stem}/solution|{a['title']}]]"
        )
    lines.append("")

    if all_links:
        role_by_key = {
            (a["topic_id"], r.get("url")): r.get("role")
            for a in analyses
            for r in a["analysis"].get("github_repos", [])
        }
        stem_by_topic = {
            a["topic_id"]: _rank_stem(a["analysis"].get("rank"), a["topic_id"])
            for a in analyses
        }
        lines.append("## GitHub links")
        for l in all_links:
            stem = stem_by_topic.get(l["topic_id"], "")
            role = role_by_key.get((l["topic_id"], l["url"]))
            role_str = f" _({role})_" if role else ""
            lines.append(
                f"- [{l['owner']}/{l['repo']}]({l['url']}){role_str}"
                f" — from [[solutions/{stem}/solution|{l['title']}]]"
            )
        lines.append("")

    all_papers: dict[str, dict] = {}
    for a in analyses:
        for p in a["analysis"].get("papers", []):
            key = p.get("arxiv_id") or p.get("title") or p.get("url", "")
            if key and key not in all_papers:
                all_papers[key] = p
    if all_papers:
        lines.append("## Papers cited")
        for p in all_papers.values():
            title = p.get("title", "?")
            url = p.get("url", "")
            lines.append(f"- [{title}]({url})" if url else f"- {title}")
        lines.append("")

    return "\n".join(lines)


def _solutions_already_run(out: Path) -> bool:
    sol_dir = out / "solutions"
    return sol_dir.is_dir() and any(sol_dir.iterdir())


def cmd_solutions(args: argparse.Namespace) -> int:
    out = competition_dir(args.slug)

    if not getattr(args, "force", False) and _solutions_already_run(out):
        print(
            f"  solutions already analyzed in {out}/solutions/; "
            f"skipping (use --force to rerun)"
        )
        return 0

    with KaggleRPC(args.slug) as rpc:
        comp = rpc.competition()
        (out / "competition.json").write_text(json.dumps(comp, indent=2))
        forum_id = comp["forumId"]
        print(f"forumId={forum_id}")

        all_topics: list[dict] = []
        for page in range(1, args.pages + 1):
            batch = rpc.topic_page(forum_id, page)
            if not batch:
                break
            all_topics.extend(batch)
        unique = list({t["id"]: t for t in all_topics}.values())
        unique.sort(key=lambda t: t.get("votes") or 0, reverse=True)
        print(f"Fetched {len(unique)} unique topics across {args.pages} pages")

        selected = ask_ollama_to_filter(unique[: args.limit], model=args.model)
        print(f"Ollama ({args.model}) selected {len(selected)} solution threads")

        all_links: list[dict] = []
        all_analyses: list[dict] = []
        for i, s in enumerate(selected, 1):
            tid = int(s["id"])
            detail = rpc.topic(tid)
            print(f"  [{i}/{len(selected)}] analyzing {tid}: {s['title'][:60]}")
            links, analysis = _write_solution_files(
                out, tid, s["title"], detail, args.slug, model=args.model
            )
            all_links.extend(links)
            all_analyses.append(analysis)
        (out / "github_links.json").write_text(json.dumps(all_links, indent=2))
        print(f"Wrote {len(selected)} solution writeups to {out}/solutions/")
        print(
            f"Extracted {len(all_links)} GitHub links to {out}/github_links.json"
        )

        print(f"Aggregating {len(all_analyses)} analyses with {args.model}...")
        try:
            agg = aggregate_analyses_with_ollama(all_analyses, model=args.model)
        except Exception as e:
            agg = {"error": f"{type(e).__name__}: {e}"}
        (out / "aggregated_solutions.json").write_text(json.dumps(agg, indent=2))
        if "error" not in agg:
            md = _render_aggregated_markdown(
                args.slug, agg, all_analyses, all_links
            )
            (out / "aggregated_solutions.md").write_text(md)
            print(f"Wrote aggregated solutions to {out}/aggregated_solutions.md")
        else:
            print(f"Aggregation failed: {agg['error']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akc-solutions")
    parser.add_argument("slug")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if solutions/ folder is non-empty")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args(argv)
    return cmd_solutions(args)


if __name__ == "__main__":
    raise SystemExit(main())
