# heartKIT Astro migration

## Goal and scope

Migrate public docs to Astro/Starlight using the sleepKIT layout and conversion fixes. Preserve content and URLs, render saved notebook outputs, generate public Python reference, and deploy docs independently of package releases. Runtime updates, model refreshes and Hugging Face deployment are separate follow-ups.

## References

- Issue: https://github.com/AmbiqAI/heartkit/issues/43 (creation approved).
- Worktree: /Users/adam.page/Ambiq/adks/heartkit-docs
- Branch: codex/heartkit-astro; baseline 64cd51b, version 1.8.0.
- Preview: http://127.0.0.1:8777/heartkit/
- Primary checkout untouched. PR: https://github.com/AmbiqAI/heartkit/pull/44 (250c9e7). Both independent reviews complete; findings resolved. Python CI and documentation CI passed on 250c9e7; heartKIT merge remains for user approval.

## Implemented

Astro site under astro-site, scoped navigation, branded dark hero and independent Pages workflow. Preserved MkDocs sources and repaired malformed syntax, missing model-zoo snippet includes and docstring formatting. Python edits affect docstrings only.

Migrated 44 standalone Markdown pages, five notebooks and 122 public API modules (161 catalog symbols). All five docs/notebooks pairs are identical. Downloads preserve original bytes; all 18 saved PNG figures render. Notebooks were not executed. Rich HTML outputs use plain-text fallbacks. Private modules are excluded from API pages and exports. Historical API and notebook URLs redirect.

## Verified

Production build and internal links across 334 HTML documents pass. Astro check: zero errors, warnings or hints. Four converter tests and seven browser tests pass. All 49 authored/notebook routes loaded at desktop and mobile widths without horizontal overflow or broken images; selected landing, Quickstart and notebook screenshots inspected. All 32 copied non-theme assets are byte-identical. git diff --check and notebook-renderer Ruff checks pass.

## Follow-up refinements

Restored the shared heliaEDGE/heartKIT red token mapping and a brighter red hero accent. Added uvx/pipx installation tabs with reduced-motion-aware transitions, replaced task recap tabs with a comparison table, and removed obsolete code annotation markers. Missing snippet includes now fail the build instead of silently emitting placeholder content. Desktop/mobile hero and installation screenshots inspected; installation tabs exercised. Retained interactive ECG traces and confusion matrices.

Hero copy approved: “Turn heart signals into on-device intelligence.” Introduction describes heartKIT as a Python-based AI Development Kit for heart monitoring on Ambiq devices.

Latest browser feedback resolved: mobile section switcher with only active-section pages, clearer workflow labels and no duplicate modes entry, compact footer pagination and explicit source link. Workflow recap is a comparison table; rhythm descriptions use headings. Shared configuration snippet was mislabeled JavaScript; now validated JSON with collapsed preview/download everywhere included. Train/evaluate/export diagrams use readable vertical flows. Added browser regressions for mobile section switching and configuration expansion/downloads.

Published-site audit: all 190 original sitemap routes now resolve; added 19 missing legacy redirects (API summary and standalone snippets). Checked 50 authored content tables and 348 public API names with no missing content. Original assets page was also empty; docstrings now explain bundled noise resources. Legacy route fixture guards URL coverage. See MIGRATION.md for evidence and review limits.

## Review and release status

Two independent content and delivery reviews completed. Fixed BYOT introduction loss from badge-cell skipping and preserved query/fragment on legacy redirects. Added two notebook regression tests and an eighth browser test. Delivery reviewer rechecked redirect security and JavaScript-disabled fallback; no remaining findings. Python behavior is unchanged; Ruff 0.11.12 passed.

Shared UI #185 and release PR #186 are merged. Publication workflow 36793695398 passed; v0.1.0-alpha.22 points to a62e8d45505dd3bbcdf1c4a03dfd1ec863322ecf. heartKIT package.json and regenerated lockfile pin that exact released commit. This replaces the temporary local preview package. Compact terminals within tabs retain copy controls without redundant headers.

Clean npm ci, Astro check, build, output checks and all eight browser tests pass on alpha.22. Rendered installation panel inspected; screenshot /tmp/heartkit-alpha22-terminal.png. CI on the dependency update is the remaining qualification step before final user merge approval.

## Next steps and limits

Push the dependency update and verify GitHub CI, then request final owner approval for heartKIT #44. Do not merge heartKIT without approval. Keep package release workflows unchanged. Other product consistency PRs follow heartKIT landing. Runtime updates, model refreshes and Hugging Face deployment are separate follow-ups. External links, runtime examples, dataset access, historical metrics and training were not revalidated. See astro-site/MIGRATION.md and README.md for coverage and commands.
