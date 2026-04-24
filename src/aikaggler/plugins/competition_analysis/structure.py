from __future__ import annotations

COMPETITION_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "data_modality": {
            "type": "string",
            "enum": [
                "tabular", "image", "video", "text", "audio",
                "time_series", "graph", "multimodal", "other",
            ],
        },
        "task_type": {
            "type": "string",
            "enum": [
                "classification", "regression", "segmentation", "detection",
                "ranking", "forecasting", "generation", "metric_learning",
                "clustering", "rl", "other",
            ],
        },
        "label_structure": {
            "type": "string",
            "enum": [
                "single_label", "multi_label", "hierarchical",
                "none", "unknown",
            ],
        },
        "metric": {"type": "string"},
        "test_split": {
            "type": "string",
            "enum": [
                "random", "temporal", "grouped", "stratified", "unknown",
            ],
        },
        "format": {
            "type": "string",
            "enum": ["standard", "code_comp", "two_stage", "simulation"],
        },
        "dataset_scale": {
            "type": "string",
            "enum": ["small", "medium", "large", "xlarge", "unknown"],
        },
        "domain": {"type": "string"},
        "data_challenges": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "description", "data_modality", "task_type", "label_structure",
        "metric", "test_split", "format", "dataset_scale", "domain",
        "data_challenges", "constraints",
    ],
}
