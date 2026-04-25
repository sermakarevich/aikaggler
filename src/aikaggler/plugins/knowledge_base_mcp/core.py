from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from aikaggler.plugins.knowledge_base_mcp.catalog import (
    Catalog,
    CompetitionRecord,
    default_catalog,
    list_notebook_dirs,
    list_solution_dirs,
    load_notebook_analysis,
    load_notebook_metadata,
    load_solution_analysis,
)
from aikaggler.plugins.knowledge_base_mcp.config import (
    AGG_FIELDS_NOTEBOOKS,
    AGG_FIELDS_SOLUTIONS,
    CLASSIFICATION_FIELDS_LIST,
    CLASSIFICATION_FIELDS_SCALAR,
    DEFAULT_LIST_LIMIT,
    DEFAULT_SORT,
)


def _agg_fields(source: str) -> tuple[str, ...]:
    if source == "notebooks":
        return AGG_FIELDS_NOTEBOOKS
    if source == "solutions":
        return AGG_FIELDS_SOLUTIONS
    raise ValueError(f"source must be 'notebooks' or 'solutions', got {source!r}")


def _agg_doc(record: CompetitionRecord, source: str) -> dict | None:
    if source == "notebooks":
        return record.aggregated_notebooks
    if source == "solutions":
        return record.aggregated_solutions
    raise ValueError(f"source must be 'notebooks' or 'solutions', got {source!r}")


def _passes_filters(record: CompetitionRecord, filters: dict | None) -> bool:
    if not filters:
        return True
    if (slugs := filters.get("slugs")) and record.slug not in set(slugs):
        return False
    cls = record.classification
    for field_name in CLASSIFICATION_FIELDS_SCALAR:
        wanted = filters.get(field_name)
        if wanted and cls.get(field_name) not in set(wanted):
            return False
    for field_name in CLASSIFICATION_FIELDS_LIST:
        wanted = filters.get(field_name)
        if wanted and not (set(wanted) & set(cls.get(field_name) or [])):
            return False
    comp = record.competition
    if (after := filters.get("deadline_after")) and (comp.get("deadline") or "") < after:
        return False
    if (before := filters.get("deadline_before")) and (comp.get("deadline") or "") > before:
        return False
    if (after := filters.get("date_enabled_after")) and (comp.get("dateEnabled") or "") < after:
        return False
    if (before := filters.get("date_enabled_before")) and (comp.get("dateEnabled") or "") > before:
        return False
    if (min_teams := filters.get("min_teams")) is not None and (comp.get("totalTeams") or 0) < min_teams:
        return False
    if (min_prizes := filters.get("min_prizes")) is not None and (comp.get("numPrizes") or 0) < min_prizes:
        return False
    return True


_SORT_KEYS: dict[str, tuple[str | tuple[str, str], bool]] = {
    "deadline_asc": ("deadline", False),
    "deadline_desc": ("deadline", True),
    "date_enabled_asc": ("dateEnabled", False),
    "date_enabled_desc": ("dateEnabled", True),
    "total_teams_desc": ("totalTeams", True),
    "total_competitors_desc": ("totalCompetitors", True),
    "reward_desc": (("reward", "quantity"), True),
}


def _sort_value(comp: dict, key: str | tuple[str, str]) -> Any:
    if isinstance(key, tuple):
        outer = comp.get(key[0]) or {}
        return outer.get(key[1]) or 0
    return comp.get(key) or ""


def _sort_records(records: list[CompetitionRecord], sort: str) -> list[CompetitionRecord]:
    if sort not in _SORT_KEYS:
        raise ValueError(f"unknown sort key {sort!r}")
    key, reverse = _SORT_KEYS[sort]
    return sorted(records, key=lambda r: _sort_value(r.competition, key), reverse=reverse)


def _summarize(record: CompetitionRecord) -> dict:
    comp = record.competition
    cls = record.classification
    return {
        "slug": record.slug,
        "title": comp.get("title", ""),
        "brief_description": comp.get("briefDescription", ""),
        "deadline": comp.get("deadline", ""),
        "date_enabled": comp.get("dateEnabled", ""),
        "total_teams": comp.get("totalTeams", 0),
        "total_competitors": comp.get("totalCompetitors", 0),
        "reward": comp.get("reward") or {},
        "data_modality": cls.get("data_modality"),
        "task_type": cls.get("task_type"),
        "domain": cls.get("domain"),
        "format": cls.get("format"),
    }


def _filter_and_sort(
    catalog: Catalog,
    filters: dict | None,
    sort: str,
) -> list[CompetitionRecord]:
    matches = [r for r in catalog.all() if _passes_filters(r, filters)]
    return _sort_records(matches, sort)


def list_competitions(
    filters: dict | None = None,
    sort: str = DEFAULT_SORT,
    limit: int | None = DEFAULT_LIST_LIMIT,
    catalog: Catalog | None = None,
) -> dict:
    cat = catalog or default_catalog()
    matches = _filter_and_sort(cat, filters, sort)
    total = len(matches)
    if limit is not None:
        matches = matches[:limit]
    return {
        "total": total,
        "returned": len(matches),
        "sort": sort,
        "competitions": [_summarize(r) for r in matches],
    }


def get_competition(slug: str, catalog: Catalog | None = None) -> dict:
    cat = catalog or default_catalog()
    record = cat.get(slug)
    if record is None:
        return {"error": f"unknown slug {slug!r}", "available": cat.slugs()[:10]}
    comp = record.competition
    keep_keys = (
        "id", "competitionName", "title", "briefDescription", "dateEnabled",
        "deadline", "totalTeams", "totalCompetitors", "totalSubmissions",
        "totalJoinedUsers", "numPrizes", "maxDailySubmissions", "maxTeamSize",
        "reward", "license", "categories", "organization", "hostName",
        "evaluationAlgorithm", "onlyAllowKernelSubmissions",
        "requiredSubmissionFilename", "maxCpuRuntimeMinutes",
        "maxGpuRuntimeMinutes",
    )
    competition_slim = {k: comp[k] for k in keep_keys if k in comp}
    return {
        "slug": record.slug,
        "competition": competition_slim,
        "classification": record.classification,
        "classification_present": record.classification_present,
        "counts": {
            "notebooks": len(list_notebook_dirs(record.folder)),
            "solutions": len(list_solution_dirs(record.folder)),
            "github_links": len(record.github_links),
            "has_aggregated_notebooks": record.aggregated_notebooks is not None,
            "has_aggregated_solutions": record.aggregated_solutions is not None,
        },
    }


def list_tags(
    field: str | None = None,
    include_slugs: bool = False,
    filters: dict | None = None,
    catalog: Catalog | None = None,
) -> dict:
    cat = catalog or default_catalog()
    records = [r for r in cat.all() if _passes_filters(r, filters)]
    fields = (field,) if field else CLASSIFICATION_FIELDS_SCALAR + CLASSIFICATION_FIELDS_LIST
    invalid = [f for f in fields if f not in CLASSIFICATION_FIELDS_SCALAR + CLASSIFICATION_FIELDS_LIST]
    if invalid:
        return {"error": f"unknown field(s): {invalid}"}
    out: dict[str, Any] = {}
    for f in fields:
        counts: Counter[str] = Counter()
        slugs_per_value: dict[str, list[str]] = {}
        for r in records:
            value = r.classification.get(f)
            values = value if isinstance(value, list) else ([value] if value else [])
            for v in values:
                counts[v] += 1
                if include_slugs:
                    slugs_per_value.setdefault(v, []).append(r.slug)
        if include_slugs:
            out[f] = {v: {"count": c, "slugs": slugs_per_value.get(v, [])} for v, c in counts.most_common()}
        else:
            out[f] = dict(counts.most_common())
    return out if field is None else {field: out[field]}


def aggregate_field(
    field: str,
    source: str,
    filters: dict | None = None,
    flatten: bool = False,
    catalog: Catalog | None = None,
) -> dict:
    valid = _agg_fields(source)
    if field not in valid:
        return {"error": f"field {field!r} not in {source} schema; valid: {list(valid)}"}
    cat = catalog or default_catalog()
    records = [r for r in cat.all() if _passes_filters(r, filters)]
    per_comp: list[dict] = []
    flat_counts: Counter[str] = Counter()
    for r in records:
        doc = _agg_doc(r, source)
        if doc is None:
            continue
        value = doc.get(field)
        if value is None or value == "" or value == []:
            continue
        items = value if isinstance(value, list) else [value]
        per_comp.append({"slug": r.slug, "items": items})
        if flatten:
            for item in items:
                flat_counts[item] += 1
    result: dict[str, Any] = {
        "field": field,
        "source": source,
        "total_competitions_with_data": len(per_comp),
    }
    if flatten:
        result["flattened"] = dict(flat_counts.most_common())
    else:
        result["per_competition"] = per_comp
    return result


def compare_competitions(
    slugs: list[str],
    fields: list[str] | None = None,
    source: str = "solutions",
    catalog: Catalog | None = None,
) -> dict:
    valid = _agg_fields(source)
    selected = tuple(fields) if fields else valid
    invalid = [f for f in selected if f not in valid]
    if invalid:
        return {"error": f"unknown field(s) for {source}: {invalid}; valid: {list(valid)}"}
    cat = catalog or default_catalog()
    matrix: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for slug in slugs:
        record = cat.get(slug)
        if record is None:
            missing.append(slug)
            continue
        doc = _agg_doc(record, source) or {}
        matrix[slug] = {f: doc.get(f) for f in selected}
    return {"source": source, "fields": list(selected), "matrix": matrix, "missing": missing}


def get_aggregated(
    slug: str,
    source: str,
    fields: list[str] | None = None,
    catalog: Catalog | None = None,
) -> dict:
    valid = _agg_fields(source)
    cat = catalog or default_catalog()
    record = cat.get(slug)
    if record is None:
        return {"error": f"unknown slug {slug!r}"}
    doc = _agg_doc(record, source)
    if doc is None:
        return {"slug": slug, "source": source, "data": None}
    if fields:
        invalid = [f for f in fields if f not in valid]
        if invalid:
            return {"error": f"unknown field(s) for {source}: {invalid}; valid: {list(valid)}"}
        data = {f: doc.get(f) for f in fields}
    else:
        data = doc
    return {"slug": slug, "source": source, "data": data}


def list_notebooks(slug: str, catalog: Catalog | None = None) -> dict:
    cat = catalog or default_catalog()
    record = cat.get(slug)
    if record is None:
        return {"error": f"unknown slug {slug!r}"}
    entries: list[dict] = []
    for dir_name in list_notebook_dirs(record.folder):
        analysis = load_notebook_analysis(record.folder, dir_name) or {}
        meta = load_notebook_metadata(record.folder, dir_name) or {}
        entries.append({
            "dir": dir_name,
            "title": meta.get("title", ""),
            "votes": analysis.get("votes"),
            "purpose": analysis.get("purpose"),
            "models": analysis.get("models", []),
            "frameworks_used": analysis.get("frameworks_used", []),
            "source_url": analysis.get("source_url", ""),
        })
    return {"slug": slug, "count": len(entries), "notebooks": entries}


def list_solutions(slug: str, catalog: Catalog | None = None) -> dict:
    cat = catalog or default_catalog()
    record = cat.get(slug)
    if record is None:
        return {"error": f"unknown slug {slug!r}"}
    entries: list[dict] = []
    for dir_name in list_solution_dirs(record.folder):
        analysis = load_solution_analysis(record.folder, dir_name) or {}
        entries.append({
            "dir": dir_name,
            "rank": analysis.get("rank"),
            "summary": analysis.get("summary", "")[:240],
            "models": analysis.get("models", []),
            "frameworks_used": analysis.get("frameworks_used", []),
            "source_url": analysis.get("source_url", ""),
        })
    return {"slug": slug, "count": len(entries), "solutions": entries}


def get_notebook(slug: str, dir_name: str, catalog: Catalog | None = None) -> dict:
    cat = catalog or default_catalog()
    record = cat.get(slug)
    if record is None:
        return {"error": f"unknown slug {slug!r}"}
    analysis = load_notebook_analysis(record.folder, dir_name)
    if analysis is None:
        return {"error": f"notebook {dir_name!r} not found in {slug}"}
    return {
        "slug": slug,
        "dir": dir_name,
        "analysis": analysis,
        "kernel_metadata": load_notebook_metadata(record.folder, dir_name) or {},
    }


def get_solution(slug: str, dir_name: str, catalog: Catalog | None = None) -> dict:
    cat = catalog or default_catalog()
    record = cat.get(slug)
    if record is None:
        return {"error": f"unknown slug {slug!r}"}
    analysis = load_solution_analysis(record.folder, dir_name)
    if analysis is None:
        return {"error": f"solution {dir_name!r} not found in {slug}"}
    return {"slug": slug, "dir": dir_name, "analysis": analysis}


_SEARCHABLE_FIELDS_AGG: tuple[str, ...] = (
    "overall_summary",
    "competition_flows",
    "models",
    "frameworks_used",
    "insights",
    "critical_findings",
    "what_did_not_work",
    "notable_individual_insights",
    "data_processing",
    "ensembling",
)


def _iter_text(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_text(item)


def search(
    query: str,
    source: str | None = None,
    fields: list[str] | None = None,
    slugs: list[str] | None = None,
    limit: int = 100,
    catalog: Catalog | None = None,
) -> dict:
    needle = (query or "").strip().lower()
    if not needle:
        return {"error": "query must be non-empty"}
    sources = (source,) if source else ("notebooks", "solutions")
    for s in sources:
        if s not in ("notebooks", "solutions"):
            return {"error": f"source must be 'notebooks' or 'solutions', got {s!r}"}
    cat = catalog or default_catalog()
    target_slugs = set(slugs) if slugs else None
    hits: list[dict] = []
    for record in cat.all():
        if target_slugs is not None and record.slug not in target_slugs:
            continue
        for s in sources:
            doc = _agg_doc(record, s)
            if doc is None:
                continue
            valid = _agg_fields(s)
            search_fields = tuple(fields) if fields else _SEARCHABLE_FIELDS_AGG
            for f in search_fields:
                if f not in valid:
                    continue
                for text in _iter_text(doc.get(f)):
                    if needle in text.lower():
                        hits.append({
                            "slug": record.slug,
                            "source": s,
                            "field": f,
                            "match": text,
                        })
                        if len(hits) >= limit:
                            return {"query": query, "hit_count": len(hits), "hits": hits}
    return {"query": query, "hit_count": len(hits), "hits": hits}


def list_github_links(
    slugs: list[str] | None = None,
    catalog: Catalog | None = None,
) -> dict:
    cat = catalog or default_catalog()
    target = set(slugs) if slugs else None
    out: dict[str, list[dict]] = {}
    for record in cat.all():
        if target is not None and record.slug not in target:
            continue
        if record.github_links:
            out[record.slug] = record.github_links
    return {"competitions": out, "total_links": sum(len(v) for v in out.values())}
