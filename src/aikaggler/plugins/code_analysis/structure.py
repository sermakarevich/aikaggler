from __future__ import annotations

NOTEBOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "tldr": {"type": "string"},
        "summary": {"type": "string"},
        "extensive_summary": {"type": "string"},
        "purpose": {
            "type": "string",
            "enum": [
                "eda", "baseline", "feature_engineering", "training",
                "inference", "ensemble", "tutorial", "utility", "other",
            ],
        },
        "models": {"type": "array", "items": {"type": "string"}},
        "frameworks": {"type": "array", "items": {"type": "string"}},
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
        "notable_techniques": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "tldr", "summary", "extensive_summary", "purpose", "models",
        "frameworks", "cv_strategy", "preprocessing", "feature_engineering",
        "augmentations", "loss_functions", "ensemble", "post_processing",
        "what_worked", "what_did_not_work", "critical_findings",
        "notable_techniques",
    ],
}

AGGREGATED_NOTEBOOKS_SCHEMA = {
    "type": "object",
    "properties": {
        "notebooks_tldr": {"type": "string"},
        "common_purposes": {"type": "array", "items": {"type": "string"}},
        "all_models": {"type": "array", "items": {"type": "string"}},
        "all_frameworks": {"type": "array", "items": {"type": "string"}},
        "all_cv_strategies": {"type": "array", "items": {"type": "string"}},
        "all_preprocessing": {"type": "array", "items": {"type": "string"}},
        "all_feature_engineering": {"type": "array", "items": {"type": "string"}},
        "all_augmentations": {"type": "array", "items": {"type": "string"}},
        "all_loss_functions": {"type": "array", "items": {"type": "string"}},
        "all_ensemble_patterns": {"type": "array", "items": {"type": "string"}},
        "all_post_processing": {"type": "array", "items": {"type": "string"}},
        "all_what_worked": {"type": "array", "items": {"type": "string"}},
        "all_what_did_not_work": {"type": "array", "items": {"type": "string"}},
        "all_critical_findings": {"type": "array", "items": {"type": "string"}},
        "all_notable_techniques": {"type": "array", "items": {"type": "string"}},
        "notable_individual_insights": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "notebooks_tldr", "common_purposes", "all_models", "all_frameworks",
        "all_cv_strategies", "all_preprocessing", "all_feature_engineering",
        "all_augmentations", "all_loss_functions", "all_ensemble_patterns",
        "all_post_processing", "all_what_worked", "all_what_did_not_work",
        "all_critical_findings", "all_notable_techniques",
        "notable_individual_insights",
    ],
}
