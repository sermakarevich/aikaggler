from __future__ import annotations

NOTEBOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "purpose": {
            "type": "string",
            "enum": [
                "eda", "baseline", "feature_engineering", "training",
                "inference", "ensemble", "tutorial", "utility", "other",
            ],
        },
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
    },
    "required": [
        "summary", "purpose", "competition_flow", "data_reading",
        "data_processing", "features_engineering", "models", "frameworks_used",
        "loss_functions", "cv_strategy", "ensembling", "insights",
        "critical_findings", "what_did_not_work",
    ],
}

AGGREGATED_NOTEBOOKS_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary": {"type": "string"},
        "common_purposes": {"type": "array", "items": {"type": "string"}},
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
        "overall_summary", "common_purposes", "competition_flows",
        "data_reading", "data_processing", "features_engineering",
        "models", "frameworks_used", "loss_functions",
        "cv_strategies", "ensembling", "insights",
        "critical_findings", "what_did_not_work",
        "notable_individual_insights",
    ],
}
