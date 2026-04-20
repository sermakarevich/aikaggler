default:
    @just --list

install:
    uv sync

solutions slug="human-protein-atlas-image-classification":
    uv run akc solutions {{slug}}
