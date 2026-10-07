# Home-page consistency handoff

Goal: align heartKIT with the KIT documentation sites and physioKIT.

State:
- Done locally in `codex/kit-home-consistency`: replace the hero dot with the existing product icon, render the links below the hero with the shared helia-ui Button, and set `header.titleRegularPrefix` so the prefix is regular and KIT stays bold. The home body follows the shared order: overview, four task cards with small red icons, a short uv installation path, and four shared navigation cards. Mode, dataset and model details remain on their dedicated pages.
- Verified: Astro check and build pass, 11 Playwright tests pass, and output contract and route checks pass. Desktop, 785px and 390px mobile views were inspected in light and dark mode; no horizontal overflow. Browser regression checks verify readable shared quick buttons in both themes and navigation from card-body clicks. The mobile test follows the home-page CTA to the quickstart installation tab.
- Preview: http://127.0.0.1:8777/heartkit/
- Tracked by AmbiqAI/heartkit#54. Draft PR #55 is open; no release or deployment has occurred.

Decisions:
- Reuse the repository's existing product icon and keep its current hero color and destination links.
- Use the shared Button component for the links under the hero; do not introduce a site-specific pill design.
- Keep the home page introductory; direct readers to Quickstart, Tasks, Guides and API Reference for detail.
- Pin helia-ui v0.1.0-alpha.24, published from f7158bf, with an npm 11.19.0 lockfile. Clean installs reproduce the shared header without a local patch.

Next: finish checks against the released package, final review and CI, merge under Adam's authorization, then verify the Pages deployment.

Not-found handling: restrict the product hero to the home route so unknown routes show the 404 page; built output and browser checks cover the fallback.


Release validation: clean installation of alpha.24 passed. Final check/build/output checks pass, with 11 rendered acceptance checks passing against the clean released dependency. User authorized merging after green CI.
