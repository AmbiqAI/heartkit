# AGENTS

Repo-specific notes for automation and maintenance:
- Python target is 3.12; use `uv sync` for installs and `uv run pytest tests/` for tests.
- Docs use Astro/Starlight in `astro-site/`, generated from Markdown under `docs/`, saved notebooks and Python docstrings. Edit sources rather than generated content.
- Use Node 24. From `astro-site/`, run `npm ci`, `npm run check`, `npm run build`, `npm run check:output` and `npm test`. Builds need Python and uv for static API extraction; notebook training is not executed.
- The documentation workflow deploys Pages from main independently of package releases. Preserve historical URL redirects and keep headings plain Markdown.
- Prefer `rg` for searches and avoid touching binary assets unless requested.
- Commit messages follow Conventional Commits (e.g., `feat: ...`, `fix: ...`, `chore: ...`).
