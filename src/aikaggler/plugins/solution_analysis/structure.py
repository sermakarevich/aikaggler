from __future__ import annotations

import re

GITHUB_URL_RE = re.compile(
    r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
)
NON_REPO_OWNERS = {
    "about", "collections", "enterprise", "features", "issues", "marketplace",
    "orgs", "pricing", "pulls", "readme", "search", "settings", "sponsors",
    "topics", "trending",
}

FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "solutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["id", "title"],
            },
        },
    },
    "required": ["solutions"],
}

SOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tldr": {"type": "string"},
        "summary": {"type": "string"},
        "extensive_summary": {"type": "string"},
        "rank": {"type": ["integer", "null"]},
        "models": {"type": "array", "items": {"type": "string"}},
        "cv_strategy": {"type": "string"},
        "preprocessing": {"type": "array", "items": {"type": "string"}},
        "feature_engineering": {"type": "array", "items": {"type": "string"}},
        "augmentations": {"type": "array", "items": {"type": "string"}},
        "loss_functions": {"type": "array", "items": {"type": "string"}},
        "ensemble": {"type": "string"},
        "post_processing": {"type": "string"},
        "what_worked": {"type": "array", "items": {"type": "string"}},
        "what_did_not_work": {"type": "array", "items": {"type": "string"}},
        "critical_findings": {"type": "array", "items": {"type": "string"}},
        "github_repos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["solution", "library", "reference"],
                    },
                },
                "required": ["url", "role"],
            },
        },
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "arxiv_id": {"type": ["string", "null"]},
                },
                "required": ["title", "url"],
            },
        },
        "other_solution_refs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["description", "url"],
            },
        },
    },
    "required": [
        "tldr", "summary", "extensive_summary", "rank", "models",
        "cv_strategy", "preprocessing", "feature_engineering", "augmentations",
        "loss_functions", "ensemble", "post_processing", "what_worked",
        "what_did_not_work", "critical_findings", "github_repos", "papers",
        "other_solution_refs",
    ],
}

AGGREGATED_SCHEMA = {
    "type": "object",
    "properties": {
        "competition_tldr": {"type": "string"},
        "key_challenges": {"type": "array", "items": {"type": "string"}},
        "all_models": {"type": "array", "items": {"type": "string"}},
        "all_cv_strategies": {"type": "array", "items": {"type": "string"}},
        "all_preprocessing": {"type": "array", "items": {"type": "string"}},
        "all_feature_engineering": {"type": "array", "items": {"type": "string"}},
        "all_augmentations": {"type": "array", "items": {"type": "string"}},
        "all_loss_functions": {"type": "array", "items": {"type": "string"}},
        "all_ensemble_patterns": {"type": "array", "items": {"type": "string"}},
        "all_post_processing": {"type": "array", "items": {"type": "string"}},
        "all_what_worked": {"type": "array", "items": {"type": "string"}},
        "all_what_did_not_work": {"type": "array", "items": {"type": "string"}},
        "all_critical_findings": {
            "type": "array", "items": {"type": "string"}
        },
        "notable_individual_insights": {
            "type": "array", "items": {"type": "string"}
        },
    },
    "required": [
        "competition_tldr", "key_challenges", "all_models",
        "all_cv_strategies", "all_preprocessing", "all_feature_engineering",
        "all_augmentations", "all_loss_functions", "all_ensemble_patterns",
        "all_post_processing", "all_what_worked", "all_what_did_not_work",
        "all_critical_findings", "notable_individual_insights",
    ],
}
