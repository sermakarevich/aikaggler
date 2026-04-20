default:
    @just --list

install:
    uv sync

solutions slug:
    uv run akc solutions {{slug}}
