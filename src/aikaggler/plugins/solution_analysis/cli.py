from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx

from aikaggler.plugins.solution_analysis.config import (
    AGGREGATE_TIMEOUT,
    ANALYZE_TIMEOUT,
    DEFAULT_OUTPUT_ROOT,
    KAGGLE_API,
    MAX_SOLUTION_BODY_CHARS,
    OLLAMA_MODEL,
    OLLAMA_RETRIES,
    OLLAMA_URL,
    PROMPTS_DIR,
    USER_AGENT,
)
from aikaggler.plugins.solution_analysis.structure import (
    AGGREGATED_SCHEMA,
    COMPETITION_CLASSIFICATION_SCHEMA,
    FILTER_SCHEMA,
    GITHUB_URL_RE,
    NON_REPO_OWNERS,
    SOLUTION_SCHEMA,
)


def load_prompt(name: str, **kwargs: str) -> str:
    text = (PROMPTS_DIR / f"{name}.txt").read_text()
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", value)
    return text


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


def _ollama_call(
    prompt: str,
    schema: dict,
    timeout: int,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
    retries: int = OLLAMA_RETRIES,
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
            out = json.loads(content)
            return out
        
        except (json.JSONDecodeError, httpx.HTTPError) as e:
            last_err = e
            if attempt < retries:
                print(
                    f"  ollama call failed ({type(e).__name__}: {e}), "
                    f"retrying {attempt}/{retries - 1}..."
                )
    raise last_err  # type: ignore[misc]


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
    result = _ollama_call(
        prompt, FILTER_SCHEMA, ANALYZE_TIMEOUT, model=model, url=url
    )
    return result.get("solutions", [])


def classify_competition_with_ollama(
    comp: dict,
    solution_titles: list[str],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    slim_keys = [
        "title", "briefDescription", "evaluationAlgorithm", "categories",
        "hostName", "organization", "totalCompetitors", "totalTeams",
        "maxDailySubmissions", "license",
    ]
    slim = {k: comp.get(k) for k in slim_keys if comp.get(k) is not None}
    prompt = load_prompt(
        "classify_competition",
        competition_json=json.dumps(slim, indent=2)[:MAX_SOLUTION_BODY_CHARS],
        solution_titles_json=json.dumps(solution_titles, indent=2),
    )
    return _ollama_call(
        prompt,
        COMPETITION_CLASSIFICATION_SCHEMA,
        ANALYZE_TIMEOUT,
        model=model,
        url=url,
    )


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
    return _ollama_call(
        prompt, SOLUTION_SCHEMA, ANALYZE_TIMEOUT, model=model, url=url
    )


def aggregate_analyses_with_ollama(
    analyses: list[dict],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> dict:
    payload = [
        {
            "rank": a["analysis"].get("rank"),
            "title": a["title"],
            "tldr": a["analysis"].get("tldr", ""),
            "summary": a["analysis"].get("summary", ""),
            "models": a["analysis"].get("models", []),
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
        }
        for a in analyses
        if "error" not in a.get("analysis", {})
    ]
    prompt = load_prompt(
        "aggregate_analyses", payload_json=json.dumps(payload, indent=2)
    )
    return _ollama_call(
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
    classification: dict | None = None,
) -> str:
    lines = [f"# {slug}: cross-solution summary", ""]
    lines += [agg.get("competition_tldr", ""), ""]

    if classification and "error" not in classification:
        lines.append("## Competition profile")
        lines.append(
            f"- **Modality / task:** {classification.get('data_modality', '?')} / "
            f"{classification.get('task_type', '?')}"
        )
        lines.append(f"- **Domain:** {classification.get('domain', '?')}")
        lines.append(f"- **Metric:** {classification.get('metric', '?')}")
        lines.append(
            f"- **Labels:** {classification.get('label_structure', '?')}"
        )
        lines.append(
            f"- **Test split:** {classification.get('test_split', '?')}"
        )
        lines.append(f"- **Format:** {classification.get('format', '?')}")
        lines.append(
            f"- **Dataset scale:** {classification.get('dataset_scale', '?')}"
        )
        challenges = classification.get("data_challenges") or []
        if challenges:
            lines.append(f"- **Data challenges:** {', '.join(challenges)}")
        constraints = classification.get("constraints") or []
        if constraints:
            lines.append(f"- **Constraints:** {', '.join(constraints)}")
        lines.append("")

    def _section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.extend(f"- {x}" for x in items)
        lines.append("")

    _section("Key challenges", agg.get("key_challenges", []))
    _section("Models", agg.get("all_models", []))
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


def cmd_solutions(args: argparse.Namespace) -> int:
    out = competition_dir(args.slug)

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

        try:
            classification = classify_competition_with_ollama(
                comp, [s["title"] for s in selected], model=args.model
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
        (out / "aggregated_analysis.json").write_text(json.dumps(agg, indent=2))
        if "error" not in agg:
            md = _render_aggregated_markdown(
                args.slug, agg, all_analyses, all_links, classification
            )
            (out / "aggregated_summary.md").write_text(md)
            print(f"Wrote aggregated summary to {out}/aggregated_summary.md")
        else:
            print(f"Aggregation failed: {agg['error']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("solutions")
    p.add_argument("slug")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--pages", type=int, default=5)
    p.add_argument("--model", default=OLLAMA_MODEL)
    p.set_defaults(func=cmd_solutions)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
