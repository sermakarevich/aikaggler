<p align="center">
  <img src="assets/image.png" alt="aikaggler" width="480">
</p>

# aikaggler

Toolkit for competing in Kaggle competitions.

## Install

```sh
uv sync
```

Requires a running Ollama instance at `localhost:11434` with the chosen model pulled, and a configured `kaggle` CLI (`~/.kaggle/kaggle.json`).

## Usage

### Full competition pipeline (solutions + notebooks + classification)

```sh
just competition <competition-slug>
# or
uv run akc competition <competition-slug> [--limit 20] [--pages 5] [--top 10] [--model gemma4:latest]
```

### Single stage

```sh
uv run akc solutions <competition-slug>    # forum writeups only
uv run akc notebooks <competition-slug>    # top voted notebooks only
```

### Bulk over every completed USD-prize competition

```sh
just run-competitions --dry-run            # preview the queue
just run-competitions                      # run the full back-catalog
# flags: --max-competitions N, --pages N, --limit N, --topic-pages N, --top N, --model ...
```

The bulk runner lists competitions via the kaggle CLI, keeps USD-prize + past-deadline entries, sorts by end date, skips any slug already present in `data/`, and runs the full `competition` pipeline on each.

## Output layout

```
data/<slug>/
├── competition.json                   # raw Kaggle metadata
├── competition.classification.json    # structured tags + prose description
├── aggregated_solutions.json          # cross-solution superset (LLM)
├── aggregated_solutions.md            # human-readable solutions digest
├── aggregated_notebooks.json          # cross-notebook superset (LLM)
├── aggregated_notebooks.md            # human-readable notebooks digest
├── github_links.json                  # links extracted from solutions
├── solutions/
│   └── rank_NN/
│       ├── solution.md                # cleaned writeup
│       └── analysis.json              # per-solution structured fields
└── notebooks/
    └── votes_NN_<owner-kernel>/
        ├── notebook.md                # notebook as markdown + code
        ├── analysis.json              # per-notebook structured fields
        ├── kernel-metadata.json       # kaggle metadata
        └── <kernel>.ipynb             # original notebook
```

## Plugins

- `competition_analysis` — orchestrator. Runs solutions + notebooks, then classifies the competition (structured tags + prose description).
- `solution_analysis` — scrapes the discussion forum via Kaggle's internal RPC, filters to real solution writeups with a local LLM, extracts structured per-solution analyses, aggregates.
- `code_analysis` — lists top-voted notebooks via `kaggle kernels list`, pulls each, converts `.ipynb` to readable markdown+code, extracts structured per-notebook analyses, aggregates.
- `knowledge_base_mcp` — read-only MCP server (and matching `akb` CLI) over the harvested `data/<slug>/` artifacts. Exposes ~12 tools for filtering, comparing, and aggregating across competitions (`list_competitions`, `get_competition`, `aggregate_field`, `compare_competitions`, `search`, …). See [its README](src/aikaggler/plugins/knowledge_base_mcp/README.md) for full details.
- `_shared/ollama.py` — unified `ollama_call` + prompt loader used by all plugins.

## Knowledge Base MCP

Query the local knowledge base from Claude Code (or any MCP client) without re-reading per-competition JSON by hand.

```sh
just kb-mcp                          # boot the stdio MCP server locally
just kb list                         # CLI equivalent: latest 20 competitions

# Register with Claude Code:
claude mcp add aikaggler-kb -- uv --directory /home/sergii/kaggle/aikaggler run akb-mcp
```

Tools then appear as `mcp__aikaggler-kb__list_competitions`, `mcp__aikaggler-kb__aggregate_field`, etc.
