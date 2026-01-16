# AGENTS

Repo-specific notes for automation and maintenance:
- Python target is 3.12; use `uv sync` for installs and `uv run pytest tests/` for tests.
- Docs use MkDocs Material; preview with `mkdocs serve` and keep headings plain Markdown (no span wrappers).
- Prefer `rg` for searches and avoid touching binary assets unless requested.
- Commit messages follow Conventional Commits (e.g., `feat: ...`, `fix: ...`, `chore: ...`).
