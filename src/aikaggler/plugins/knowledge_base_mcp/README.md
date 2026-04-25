# knowledge_base_mcp

Read-only MCP service that exposes the per-competition artifacts produced by
`akc` (`competition.json`, `competition.classification.json`,
`aggregated_notebooks.json`, `aggregated_solutions.json`, `github_links.json`,
and per-notebook / per-solution `analysis.json`) under `aikaggler/data/`.

## Tools

| Tool | Purpose |
|---|---|
| `list_competitions` | Filter + sort + paginate (default: latest 20 by `date_enabled_desc`). |
| `get_competition` | Slim `competition.json` + classification + counts for one slug. |
| `list_tags` | Counts per classification tag value (one field or all). |
| `aggregate_field` | Pull a synthesis field (`models`, `insights`, `what_did_not_work`, …) from `notebooks` or `solutions` across filtered competitions. |
| `compare_competitions` | Side-by-side matrix of synthesis fields across slugs. |
| `get_aggregated` | Full aggregated_notebooks/solutions for one slug; optional field subset. |
| `list_notebooks` / `list_solutions` | Lightweight per-entry index. |
| `get_notebook` / `get_solution` | Per-entry `analysis.json` (+ kernel metadata for notebooks). |
| `search` | Case-insensitive substring across summaries / insights / models. |
| `list_github_links` | `github_links.json` grouped by slug. |

Competitions without `competition.classification.json` (e.g.
`drawing-with-llms`) surface every classification field as `"other"` /
empty list, so they remain filterable.

## Filter shape

Every cross-cutting tool accepts the same `filters` dict:

```jsonc
{
  "slugs": ["..."],
  "data_modality": ["image","audio"],
  "task_type": ["classification","segmentation"],
  "label_structure": ["multi_label"],
  "test_split": ["temporal"],
  "format": ["code_comp"],
  "dataset_scale": ["small"],
  "domain": ["finance"],
  "data_challenges": ["class_imbalance"],   // any-of
  "constraints": ["kernels_only"],           // any-of
  "deadline_after":  "2025-09-01T00:00:00Z",
  "deadline_before": "2026-12-31T23:59:59Z",
  "date_enabled_after":  "2025-01-01T00:00:00Z",
  "date_enabled_before": "2026-12-31T23:59:59Z",
  "min_teams": 100,
  "min_prizes": 1
}
```

Within a field: OR. Across fields: AND.

## Install

The plugin ships with `aikaggler`:

```sh
just install            # uv sync
just kb-test            # smoke tests (14)
just kb list            # latest 20 competitions, default sort
just kb-mcp             # boot the stdio MCP server
```

## Wire into Claude Code

Add the server to Claude Code's MCP config:

```sh
claude mcp add aikaggler-kb -- uv --directory /home/sergii/kaggle/aikaggler run akb-mcp
```

Or hand-edit `~/.claude.json`:

```json
{
  "mcpServers": {
    "aikaggler-kb": {
      "command": "uv",
      "args": ["--directory", "/home/sergii/kaggle/aikaggler", "run", "akb-mcp"]
    }
  }
}
```

The 12 tools then appear as `mcp__aikaggler-kb__list_competitions`, etc.

## Use from Ollama

Ollama doesn't speak MCP natively. Two integration options:

1. **Tool-calling**: register each `core.*` function as an Ollama tool. The
   `core.py` module is a plain Python API — same signatures as the MCP tools,
   no transport coupling.
2. **CLI shim**: invoke `akb <subcommand>` via shell from a tool-calling agent
   and parse the JSON. Every MCP tool has a CLI counterpart printing JSON to
   stdout.

Example (Ollama Python client):

```python
from aikaggler.plugins.knowledge_base_mcp import core

def kb_list_competitions(filters=None, sort="date_enabled_desc", limit=20):
    return core.list_competitions(filters=filters, sort=sort, limit=limit)

# register kb_list_competitions, kb_get_competition, kb_list_tags, ...
# under your Ollama tool schema and dispatch by tool name.
```

## CLI reference

```sh
akb list --filter data_modality=image --filter format=code_comp --limit 5
akb tags --field domain --with-slugs
akb get arc-prize-2025
akb aggregate models --source solutions --filter data_modality=image --flatten
akb compare arc-prize-2025 birdclef-2025 --fields models frameworks_used
akb get-aggregated arc-prize-2025 --source notebooks --fields models insights
akb notebooks arc-prize-2025
akb notebook arc-prize-2025 votes_01_boristown-agi-compressarc
akb solutions arc-prize-2025
akb solution arc-prize-2025 rank_xx_651671
akb search "Qwen" --source solutions --limit 10
akb github --slugs arc-prize-2025
```
