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
        "summary": {"type": "string"},
        "extensive_summary": {"type": "string"},
        "rank": {"type": ["integer", "null"]},
        "competition_flow": {"type": "string"},
        "data_reading": {"type": "array", "items": {"type": "string"}},
        "data_processing": {"type": "array", "items": {"type": "string"}},
        "features_engineering": {"type": "array", "items": {"type": "string"}},
        "models": {"type": "array", "items": {"type": "string"}},
        "frameworks_used": {"type": "array", "items": {"type": "string"}},
        "loss_functions": {"type": "array", "items": {"type": "string"}},
        "cv_strategy": {"type": "string"},
        "ensembling": {"type": "string"},
        "insights": {"type": "array", "items": {"type": "string"}},
        "critical_findings": {"type": "array", "items": {"type": "string"}},
        "what_did_not_work": {"type": "array", "items": {"type": "string"}},
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
        "summary", "extensive_summary", "rank", "competition_flow",
        "data_reading", "data_processing", "features_engineering", "models",
        "frameworks_used", "loss_functions", "cv_strategy", "ensembling",
        "insights", "critical_findings", "what_did_not_work", "github_repos",
        "papers", "other_solution_refs",
    ],
}

AGGREGATED_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary": {"type": "string"},
        "competition_flows": {"type": "array", "items": {"type": "string"}},
        "data_reading": {"type": "array", "items": {"type": "string"}},
        "data_processing": {"type": "array", "items": {"type": "string"}},
        "features_engineering": {"type": "array", "items": {"type": "string"}},
        "models": {"type": "array", "items": {"type": "string"}},
        "frameworks_used": {"type": "array", "items": {"type": "string"}},
        "loss_functions": {"type": "array", "items": {"type": "string"}},
        "cv_strategies": {"type": "array", "items": {"type": "string"}},
        "ensembling": {"type": "array", "items": {"type": "string"}},
        "insights": {"type": "array", "items": {"type": "string"}},
        "critical_findings": {"type": "array", "items": {"type": "string"}},
        "what_did_not_work": {"type": "array", "items": {"type": "string"}},
        "notable_individual_insights": {
            "type": "array", "items": {"type": "string"}
        },
    },
    "required": [
        "overall_summary", "competition_flows", "data_reading",
        "data_processing", "features_engineering", "models",
        "frameworks_used", "loss_functions", "cv_strategies",
        "ensembling", "insights", "critical_findings",
        "what_did_not_work", "notable_individual_insights",
    ],
}
