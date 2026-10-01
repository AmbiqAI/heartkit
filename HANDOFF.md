# Canonical Astro documentation

Goal: retire the MkDocs compatibility layer while preserving public content and routes.

PR: https://github.com/AmbiqAI/heartkit/pull/48. Authored Markdown/MDX, navigation, redirects and static assets now belong to Astro. API pages, notebooks and downloads remain generated. MkDocs configuration and unused dependencies are removed.

Review: notebook source links, download command examples and generation regression coverage corrected. Local build, type checks, content checks and browser tests pass. Independent final review and CI on the fix commit gate merging. CompressionKIT follows this cleanup.

Ownership: edit astro-site/src/content/docs for authored pages, src/navigation.mjs for navigation and notebooks/ for notebook sources. See astro-site/README.md for generated paths and validation commands.
