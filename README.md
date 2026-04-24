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
- `_shared/ollama.py` — unified `ollama_call` + prompt loader used by all plugins.
