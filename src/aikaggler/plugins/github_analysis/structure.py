from __future__ import annotations

# Two-round LLM-driven selection of the most informative files per repo.
# Round 1: model sees the file tree and asks for orientation files (README,
# entrypoints, configs) it wants to read before committing.
FILE_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "files_to_read": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "files_to_read"],
}

# Round 2: model sees the tree + the requested files' contents and returns
# the final top-N paths to deeply analyze.
TOP_FILES_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "top_files": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "top_files"],
}

# Mirrors NOTEBOOK_SCHEMA with the only required diff being the file-specific
# 'purpose' enum (a file plays a structural role inside the repo, not a kernel
# narrative role like 'eda' / 'baseline' / 'tutorial').
FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "purpose": {
            "type": "string",
            "enum": [
                "data_loading", "preprocessing", "model_definition", "training",
                "inference", "ensemble", "evaluation", "config", "utility",
                "entrypoint", "documentation", "other",
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

# Mirrors SOLUTION_SCHEMA minus the citation pointers (rank, github_repos,
# papers, other_solution_refs, extensive_summary — those describe references
# from a forum writeup, not facts derivable from repo code itself), plus two
# repo-specific diffs: training_setup (hardware/optimizer/schedule notes) and
# notable_files (entry-point map).
REPO_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
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
        "training_setup": {"type": "array", "items": {"type": "string"}},
        "notable_files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["path", "role"],
            },
        },
    },
    "required": [
        "summary", "competition_flow", "data_reading", "data_processing",
        "features_engineering", "models", "frameworks_used", "loss_functions",
        "cv_strategy", "ensembling", "insights", "critical_findings",
        "what_did_not_work", "training_setup", "notable_files",
    ],
}

# Mirrors AGGREGATED_SCHEMA (solutions cross-aggregate) with one required diff:
# 'training_setups' as the union of per-repo training_setup entries.
AGGREGATED_REPOS_SCHEMA = {
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
        "training_setups": {"type": "array", "items": {"type": "string"}},
        "notable_individual_insights": {
            "type": "array", "items": {"type": "string"}
        },
    },
    "required": [
        "overall_summary", "competition_flows", "data_reading",
        "data_processing", "features_engineering", "models", "frameworks_used",
        "loss_functions", "cv_strategies", "ensembling", "insights",
        "critical_findings", "what_did_not_work", "training_setups",
        "notable_individual_insights",
    ],
}
