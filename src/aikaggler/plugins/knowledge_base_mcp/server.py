from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from aikaggler.plugins.knowledge_base_mcp import core
from aikaggler.plugins.knowledge_base_mcp.config import (
    DEFAULT_LIST_LIMIT,
    DEFAULT_SORT,
)

mcp = FastMCP("aikaggler-knowledge-base")


@mcp.tool()
def list_competitions(
    filters: dict | None = None,
    sort: str = DEFAULT_SORT,
    limit: int | None = DEFAULT_LIST_LIMIT,
) -> dict:
    """List Kaggle competitions in the knowledge base.

    filters: optional dict with any of:
      slugs, data_modality, task_type, label_structure, test_split, format,
      dataset_scale, domain (lists; OR within a field, AND across fields),
      data_challenges, constraints (lists; any-of intersect),
      deadline_after / deadline_before / date_enabled_after /
      date_enabled_before (ISO 8601 strings),
      min_teams, min_prizes (ints).
    sort: deadline_asc | deadline_desc | date_enabled_asc | date_enabled_desc
          | total_teams_desc | total_competitors_desc | reward_desc.
    limit: cap on returned rows; null returns all.
    """
    return core.list_competitions(filters=filters, sort=sort, limit=limit)


@mcp.tool()
def get_competition(slug: str) -> dict:
    """Full record for one competition: slimmed competition.json + classification + counts."""
    return core.get_competition(slug)


@mcp.tool()
def list_tags(
    field: str | None = None,
    include_slugs: bool = False,
    filters: dict | None = None,
) -> dict:
    """Counts of classification tag values across competitions.

    field: one of data_modality, task_type, label_structure, test_split,
           format, dataset_scale, domain, data_challenges, constraints.
           Omit to get all fields at once.
    include_slugs: when true returns {value: {count, slugs}} per value.
    filters: same shape as list_competitions filters; restricts the population.
    """
    return core.list_tags(field=field, include_slugs=include_slugs, filters=filters)


@mcp.tool()
def aggregate_field(
    field: str,
    source: str,
    filters: dict | None = None,
    flatten: bool = False,
) -> dict:
    """Pull one synthesis field from aggregated_notebooks or aggregated_solutions across competitions.

    source: 'notebooks' or 'solutions'.
    field: e.g. models, frameworks_used, insights, what_did_not_work,
           critical_findings, competition_flows, ensembling, cv_strategies,
           data_processing, data_reading, features_engineering,
           loss_functions, common_purposes (notebooks only),
           overall_summary, notable_individual_insights.
    filters: same shape as list_competitions filters.
    flatten: when true returns deduped global counts; otherwise returns per-competition lists.
    """
    return core.aggregate_field(field=field, source=source, filters=filters, flatten=flatten)


@mcp.tool()
def compare_competitions(
    slugs: list[str],
    fields: list[str] | None = None,
    source: str = "solutions",
) -> dict:
    """Side-by-side matrix of aggregated_notebooks/solutions fields across given slugs.

    fields: list of synthesis field names; omit for all.
    """
    return core.compare_competitions(slugs=slugs, fields=fields, source=source)


@mcp.tool()
def get_aggregated(
    slug: str,
    source: str,
    fields: list[str] | None = None,
) -> dict:
    """Full aggregated_notebooks/solutions for one competition; optional field subset."""
    return core.get_aggregated(slug=slug, source=source, fields=fields)


@mcp.tool()
def list_notebooks(slug: str) -> dict:
    """Index of per-notebook entries (votes_NN) for a competition: dir, title, votes, purpose, models, source_url."""
    return core.list_notebooks(slug=slug)


@mcp.tool()
def list_solutions(slug: str) -> dict:
    """Index of per-solution entries (rank_xx) for a competition: dir, rank, brief summary, models, source_url."""
    return core.list_solutions(slug=slug)


@mcp.tool()
def get_notebook(slug: str, dir_name: str) -> dict:
    """Per-notebook analysis.json + kernel-metadata.json for a single votes_NN entry."""
    return core.get_notebook(slug=slug, dir_name=dir_name)


@mcp.tool()
def get_solution(slug: str, dir_name: str) -> dict:
    """Per-solution analysis.json for a single rank_xx entry."""
    return core.get_solution(slug=slug, dir_name=dir_name)


@mcp.tool()
def search(
    query: str,
    source: str | None = None,
    fields: list[str] | None = None,
    slugs: list[str] | None = None,
    limit: int = 100,
) -> dict:
    """Case-insensitive substring search across aggregated_notebooks/solutions.

    source: 'notebooks' or 'solutions'; omit to search both.
    fields: limit search to specific synthesis fields; omit for the default set
            (overall_summary, competition_flows, models, frameworks_used,
             insights, critical_findings, what_did_not_work,
             notable_individual_insights, data_processing, ensembling).
    slugs: restrict to a competition subset.
    limit: cap on hits returned.
    """
    return core.search(query=query, source=source, fields=fields, slugs=slugs, limit=limit)


@mcp.tool()
def list_github_links(slugs: list[str] | None = None) -> dict:
    """github_links.json grouped by competition slug; optional slug filter."""
    return core.list_github_links(slugs=slugs)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
