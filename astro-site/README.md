# heartKIT documentation

Astro/Starlight renders Markdown/MDX from `src/content/docs/`, five saved notebooks and a static Griffe Python API reference. Runtime training dependencies are not imported. Private implementation modules are excluded before rendering.

Use Node24, Python3.12 and uv. From this directory:

```sh
npm ci
npx playwright install chromium
npm run dev -- --port 8779
npm run check
npm run build
npm run check:output
npm test
```

Edit authored Markdown/MDX in `src/content/docs/`, navigation in `src/navigation.mjs`, static redirects in `src/redirects.json`, and static assets in `public/`. These are canonical sources and are never replaced by the build.

`prepare:docs` generates only downloadable configuration examples, notebook guides/assets, and Python API pages/data. API output under `src/content/docs/reference/`, notebook `.md` pages under `guides/`, `src/data/`, and `public/{reference,notebooks,examples}/` are ignored. Edit Python docstrings or the notebooks in `../notebooks/` for those outputs. There is no MkDocs configuration or Markdown conversion step.

The five notebooks in `notebooks/` supply rendered pages and byte-identical downloads. Duplicate documentation copies have been removed. Saved outputs include 18 PNG figures, logs and plain-text fallbacks for rich HTML. No notebook execution occurs during builds. Notebook timestamps and measurements are historical, not current model qualification.

PRs build and test the site. Main pushes and manual main dispatches publish Pages independently of package releases. Package release workflows are unchanged.
