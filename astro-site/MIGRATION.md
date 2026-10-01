# heartKIT migration

Tracking: AmbiqAI/heartkit#43. Baseline 64cd51b (1.8.0).

## Coverage

44 standalone authored Markdown pages, five notebook pages and 122 public Python API modules (161 catalog entries). Markdown fragments under assets are included in pages. Five notebook pairs are identical; original files remain unchanged. All 18 saved PNG figures are rendered and downloads are byte-identical. Rich HTML outputs use plain-text fallbacks. No training or hardware checks performed.

## Repairs

Reused sleepKIT conversion fixes for tabs, Material callouts, tables, image dimensions, icon shortcodes and large configuration previews/downloads. Restored missing task model-zoo snippets from the existing model-zoo overview without changing reported values. Fixed the malformed architecture fragment and labeled commented class maps as JSONC. Repaired two unterminated docstring fences, a plotting example typo and a Material example wrapper. Corrected an inherited beat model-zoo link and Python requirement wording. No runtime behavior changed.

Landing uses heartKIT branding and a version from pyproject.toml, with scoped navigation and the shared heliaEDGE/heartKIT red accent. Installation includes uv, uvx, pipx, pip and Git. Task recap tabs became a comparison table, and obsolete code annotation markers were removed. Private implementation modules do not generate pages, catalog rows, search entries or machine-readable exports. Historical public API and notebook URLs redirect to their new pages.

## Validation

Production build, internal links across 334 HTML documents, four converter tests, two notebook regression tests and eight browser tests pass. Astro check reports zero errors/warnings. All 49 authored/notebook pages loaded at 1440px and 390px without viewport overflow or broken images. Landing, Quickstart and notebook screenshots inspected. Broader example runtime correctness, external links, dataset access and historical metrics have not been revalidated.

## Delivery

PR preparation is approved. Merge and production deployment await user approval. The Pages workflow is independent of package publishing.

## Published-site comparison

Compared the published MkDocs sitemap and downloaded HTML with the Astro output. All 190 original URLs have a page or redirect, enforced by `scripts/legacy-routes.json`. This includes the API summary and 18 standalone snippet URLs previously published by MkDocs. Snippet URLs lead to the pages that include their content; the diagnostic results snippet had only an empty table header and redirects to the diagnostic overview.

Checked the 50 non-code tables in authored pages for retained cell content, and 348 distinct public API headings for retained names. No missing table content or public API names found. Five notebook sources and their 18 saved PNG figures are preserved. The duplicate homepage logo images were intentionally replaced by the approved hero.

Inspected original-site screenshots for the homepage, assets API, guide index and a model-zoo page, plus corresponding local content. The original assets API was also an empty heading; package docstrings now explain the bundled noise resources and the NstdbNoise interface. This is route/content parity and representative visual review, not exhaustive visual review of every page or runtime verification of the examples. Interactive Plotly signal plots remain; rich notebook HTML outputs use plain-text fallbacks.

## Independent reviews

Content review identified two migration regressions: removing a Colab toolbar discarded BYOT prose in the same cell, and static redirects discarded API symbol fragments. The renderer now removes only toolbar markup; redirects preserve query strings and fragments with a meta-refresh fallback when JavaScript is disabled. Both changes have regression coverage. Delivery review checked Pages permissions and triggers, shared section matching, notebook assets, public API coverage and Python AST parity. No additional blocking findings remained after fix review.

Dependency qualification: shared UI alpha.21 is pinned by immutable commit `6cdbea0c594c955e6aeef232af1fbb15e395ab2d` (AmbiqAI/helia-ui#183 and #184). Clean installation, type checks, build, output checks and all eight browser tests pass with this dependency.

## Canonical sources

The one-time MkDocs adapter has been retired. Authored pages are now in `src/content/docs/`, navigation in `src/navigation.mjs`, and static assets in `public/`. API and notebook generation remain. See README.md for source ownership.
