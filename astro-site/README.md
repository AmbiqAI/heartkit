# heartKIT documentation

Astro/Starlight renders Markdown from `../docs`, five saved notebooks and a static Griffe Python API reference. Runtime training dependencies are not imported. Private implementation modules are excluded before rendering.

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

Edit source Markdown, notebook sources or owning scripts. `src/content/docs`, `src/data`, `public`, `.cache` and `dist` are generated. Existing navigation labels come from `mkdocs.yml`; `src/navigation.mjs` assigns public pages to five scoped sections.

The five notebook pairs in `docs/guides` and `notebooks` were identical at migration. Documentation copies supply the rendered pages and byte-identical downloads. Saved outputs include 18 PNG figures, logs and plain-text fallbacks for rich HTML. No notebook execution occurs during builds. Notebook timestamps and measurements are historical, not current model qualification.

PRs build and test the site. Main pushes and manual main dispatches publish Pages independently of package releases. Package release workflows are unchanged.
