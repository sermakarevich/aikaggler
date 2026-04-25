from __future__ import annotations

from pathlib import Path

from aikaggler.plugins.solution_analysis.config import DEFAULT_OUTPUT_ROOT

DEFAULT_DATA_ROOT: Path = DEFAULT_OUTPUT_ROOT
DEFAULT_LIST_LIMIT: int = 20
DEFAULT_SORT: str = "date_enabled_desc"

CLASSIFICATION_FIELDS_SCALAR: tuple[str, ...] = (
    "data_modality",
    "task_type",
    "label_structure",
    "test_split",
    "format",
    "dataset_scale",
    "domain",
)
CLASSIFICATION_FIELDS_LIST: tuple[str, ...] = (
    "data_challenges",
    "constraints",
)
CLASSIFICATION_FIELDS_ALL: tuple[str, ...] = (
    *CLASSIFICATION_FIELDS_SCALAR,
    *CLASSIFICATION_FIELDS_LIST,
)

OTHER_VALUE: str = "other"
DEFAULT_CLASSIFICATION: dict = {
    "description": "",
    "data_modality": OTHER_VALUE,
    "task_type": OTHER_VALUE,
    "label_structure": OTHER_VALUE,
    "metric": "",
    "test_split": OTHER_VALUE,
    "format": OTHER_VALUE,
    "dataset_scale": OTHER_VALUE,
    "domain": OTHER_VALUE,
    "data_challenges": [],
    "constraints": [],
}

AGG_FIELDS_NOTEBOOKS: tuple[str, ...] = (
    "overall_summary",
    "common_purposes",
    "competition_flows",
    "data_reading",
    "data_processing",
    "features_engineering",
    "models",
    "frameworks_used",
    "loss_functions",
    "cv_strategies",
    "ensembling",
    "insights",
    "critical_findings",
    "what_did_not_work",
    "notable_individual_insights",
)
AGG_FIELDS_SOLUTIONS: tuple[str, ...] = (
    "overall_summary",
    "competition_flows",
    "data_reading",
    "data_processing",
    "features_engineering",
    "models",
    "frameworks_used",
    "loss_functions",
    "cv_strategies",
    "ensembling",
    "insights",
    "critical_findings",
    "what_did_not_work",
    "notable_individual_insights",
)
