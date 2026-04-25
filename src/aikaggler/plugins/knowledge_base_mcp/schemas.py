from __future__ import annotations

from typing import Any, Literal, TypedDict

Source = Literal["notebooks", "solutions"]
SortKey = Literal[
    "deadline_asc",
    "deadline_desc",
    "date_enabled_asc",
    "date_enabled_desc",
    "total_teams_desc",
    "total_competitors_desc",
    "reward_desc",
]


class Filters(TypedDict, total=False):
    slugs: list[str]
    data_modality: list[str]
    task_type: list[str]
    label_structure: list[str]
    test_split: list[str]
    format: list[str]
    dataset_scale: list[str]
    domain: list[str]
    data_challenges: list[str]
    constraints: list[str]
    deadline_after: str
    deadline_before: str
    date_enabled_after: str
    date_enabled_before: str
    min_teams: int
    min_prizes: int


class CompetitionSummary(TypedDict):
    slug: str
    title: str
    brief_description: str
    deadline: str
    date_enabled: str
    total_teams: int
    total_competitors: int
    reward: dict[str, Any]
    data_modality: str
    task_type: str
    domain: str
    format: str
