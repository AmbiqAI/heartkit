# Canonical Astro documentation

Issue: AmbiqAI/heartkit#47. Branch: codex/canonical-astro-docs.

Goal: retire the MkDocs compatibility layer without changing published content or runtime behavior.

Baseline build passed. Source and output baselines are saved under /tmp/heartkit-canonical-baseline. Worktree is isolated from prior migration work.

Plan: keep authored MDX and navigation; retain API/notebook generation; remove unused MkDocs config/dependencies; verify content, links, downloads and browser behavior. Publish PR for review; do not merge without approval.

Implementation: authored MDX and static public assets are tracked; navigation is direct Astro configuration; static redirects are checked in. API, notebook and example-download generation remain. Removed MkDocs config, converters, duplicate notebook sources and docs dependency group; lock regenerated. sleepKIT retains golden-contract evidence at docs/evidence and private notes in docs-maintainers.

Verified: baseline and migrated route sets identical; all static assets, notebook downloads and example downloads byte-identical; authored MDX unchanged. Build/type/output checks passed, including generation-preserves-authored-source regression. Browser suites passed (heartKIT 8, sleepKIT 17). Clean install with generated outputs removed, build/type/output checks and visual review passed. No third-party lockfile package versions changed. heartKIT lock regeneration also aligns its project version with pyproject.toml. PR publication pending. EDGE cleanup is already landed. AOT MkDocs is an active exported-model offline-doc feature, intentionally retained.
