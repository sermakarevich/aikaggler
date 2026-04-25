from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from aikaggler.plugins.knowledge_base_mcp.config import (
    DEFAULT_CLASSIFICATION,
    DEFAULT_DATA_ROOT,
)


@dataclass
class CompetitionRecord:
    slug: str
    folder: Path
    competition: dict
    classification: dict
    aggregated_notebooks: dict | None
    aggregated_solutions: dict | None
    github_links: list[dict]
    classification_present: bool
    folder_mtime: float = field(default=0.0)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _safe_read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _folder_mtime(folder: Path) -> float:
    latest = folder.stat().st_mtime
    for child in folder.iterdir():
        if child.is_file():
            latest = max(latest, child.stat().st_mtime)
    return latest


def load_competition(folder: Path) -> CompetitionRecord | None:
    comp_path = folder / "competition.json"
    if not comp_path.is_file():
        return None
    competition = _safe_read_json(comp_path)
    if competition is None:
        return None
    classification_raw = _safe_read_json(folder / "competition.classification.json")
    classification_present = classification_raw is not None
    classification = (
        {**DEFAULT_CLASSIFICATION, **classification_raw}
        if classification_present
        else dict(DEFAULT_CLASSIFICATION)
    )
    return CompetitionRecord(
        slug=folder.name,
        folder=folder,
        competition=competition,
        classification=classification,
        aggregated_notebooks=_safe_read_json(folder / "aggregated_notebooks.json"),
        aggregated_solutions=_safe_read_json(folder / "aggregated_solutions.json"),
        github_links=_safe_read_json(folder / "github_links.json") or [],
        classification_present=classification_present,
        folder_mtime=_folder_mtime(folder),
    )


class Catalog:
    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self._records: dict[str, CompetitionRecord] = {}
        self._mtimes: dict[str, float] = {}
        self._loaded: bool = False

    def _competition_dirs(self) -> list[Path]:
        if not self.data_root.is_dir():
            return []
        return sorted(p for p in self.data_root.iterdir() if p.is_dir())

    def refresh(self) -> None:
        seen: set[str] = set()
        for folder in self._competition_dirs():
            slug = folder.name
            seen.add(slug)
            mtime = _folder_mtime(folder)
            if self._mtimes.get(slug) == mtime and slug in self._records:
                continue
            record = load_competition(folder)
            if record is None:
                self._records.pop(slug, None)
                self._mtimes.pop(slug, None)
                continue
            self._records[slug] = record
            self._mtimes[slug] = mtime
        for slug in list(self._records):
            if slug not in seen:
                self._records.pop(slug, None)
                self._mtimes.pop(slug, None)
        self._loaded = True

    def all(self) -> list[CompetitionRecord]:
        if not self._loaded:
            self.refresh()
        return list(self._records.values())

    def get(self, slug: str) -> CompetitionRecord | None:
        if not self._loaded:
            self.refresh()
        return self._records.get(slug)

    def slugs(self) -> list[str]:
        if not self._loaded:
            self.refresh()
        return sorted(self._records.keys())


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    catalog = Catalog()
    catalog.refresh()
    return catalog


def load_notebook_analysis(folder: Path, dir_name: str) -> dict | None:
    return _safe_read_json(folder / "notebooks" / dir_name / "analysis.json")


def load_notebook_metadata(folder: Path, dir_name: str) -> dict | None:
    return _safe_read_json(folder / "notebooks" / dir_name / "kernel-metadata.json")


def load_solution_analysis(folder: Path, dir_name: str) -> dict | None:
    return _safe_read_json(folder / "solutions" / dir_name / "analysis.json")


def list_notebook_dirs(folder: Path) -> list[str]:
    nb = folder / "notebooks"
    if not nb.is_dir():
        return []
    return sorted(p.name for p in nb.iterdir() if p.is_dir())


def list_solution_dirs(folder: Path) -> list[str]:
    sl = folder / "solutions"
    if not sl.is_dir():
        return []
    return sorted(p.name for p in sl.iterdir() if p.is_dir())
