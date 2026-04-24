default:
    @just --list

install:
    uv sync

competition slug="human-protein-atlas-image-classification":
    uv run akc competition {{slug}}

solutions slug="human-protein-atlas-image-classification":
    uv run akc solutions {{slug}}

notebooks slug="human-protein-atlas-image-classification":
    uv run akc notebooks {{slug}}

run-competitions *args="":
    uv run python run_competitions.py {{args}}
