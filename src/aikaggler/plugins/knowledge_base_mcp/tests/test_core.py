from __future__ import annotations

from aikaggler.plugins.knowledge_base_mcp import core
from aikaggler.plugins.knowledge_base_mcp.catalog import Catalog
from aikaggler.plugins.knowledge_base_mcp.config import (
    AGG_FIELDS_NOTEBOOKS,
    AGG_FIELDS_SOLUTIONS,
    CLASSIFICATION_FIELDS_ALL,
    DEFAULT_DATA_ROOT,
)


def _catalog() -> Catalog:
    catalog = Catalog(DEFAULT_DATA_ROOT)
    catalog.refresh()
    return catalog


def test_list_competitions_default_returns_20() -> None:
    cat = _catalog()
    result = core.list_competitions(catalog=cat)
    assert result["sort"] == "date_enabled_desc"
    assert result["returned"] == min(20, result["total"])
    assert all("slug" in c for c in result["competitions"])


def test_list_competitions_filter_by_modality() -> None:
    cat = _catalog()
    result = core.list_competitions({"data_modality": ["image"]}, limit=None, catalog=cat)
    assert result["total"] >= 1
    assert all(c["data_modality"] == "image" for c in result["competitions"])


def test_list_tags_full_inventory_covers_taxonomy() -> None:
    cat = _catalog()
    tags = core.list_tags(catalog=cat)
    assert set(tags.keys()) == set(CLASSIFICATION_FIELDS_ALL)
    assert tags["data_modality"]


def test_drawing_with_llms_surfaces_as_other() -> None:
    cat = _catalog()
    result = core.get_competition("drawing-with-llms", catalog=cat)
    assert result["classification_present"] is False
    assert result["classification"]["data_modality"] == "other"


def test_get_competition_known_slug() -> None:
    cat = _catalog()
    result = core.get_competition("arc-prize-2025", catalog=cat)
    assert result["slug"] == "arc-prize-2025"
    assert result["counts"]["notebooks"] >= 1


def test_aggregate_field_per_competition() -> None:
    cat = _catalog()
    result = core.aggregate_field("models", "solutions", catalog=cat)
    assert "per_competition" in result
    assert all("slug" in row and "items" in row for row in result["per_competition"])


def test_aggregate_field_flatten_returns_counts() -> None:
    cat = _catalog()
    result = core.aggregate_field(
        "frameworks_used", "notebooks", flatten=True, catalog=cat
    )
    assert "flattened" in result
    assert all(isinstance(c, int) for c in result["flattened"].values())


def test_aggregate_field_invalid_field() -> None:
    cat = _catalog()
    result = core.aggregate_field("nope", "solutions", catalog=cat)
    assert "error" in result


def test_compare_competitions() -> None:
    cat = _catalog()
    slugs = ["arc-prize-2025", "birdclef-2025"]
    result = core.compare_competitions(slugs, fields=["models"], catalog=cat)
    assert set(result["matrix"].keys()) == set(slugs)
    assert result["fields"] == ["models"]


def test_search_qwen_finds_arc_prize_2025() -> None:
    cat = _catalog()
    result = core.search("Qwen", catalog=cat)
    assert result["hit_count"] > 0
    assert any(h["slug"] == "arc-prize-2025" for h in result["hits"])


def test_list_notebooks_arc() -> None:
    cat = _catalog()
    result = core.list_notebooks("arc-prize-2025", catalog=cat)
    assert result["count"] >= 1
    first = result["notebooks"][0]
    assert "votes" in first and "source_url" in first


def test_list_solutions_arc() -> None:
    cat = _catalog()
    result = core.list_solutions("arc-prize-2025", catalog=cat)
    assert result["count"] >= 1


def test_github_links_arc() -> None:
    cat = _catalog()
    result = core.list_github_links(["arc-prize-2025"], catalog=cat)
    assert "arc-prize-2025" in result["competitions"]
    assert result["total_links"] >= 1


def test_agg_fields_all_known() -> None:
    for f in AGG_FIELDS_SOLUTIONS:
        assert f in AGG_FIELDS_SOLUTIONS
    assert "common_purposes" in AGG_FIELDS_NOTEBOOKS
