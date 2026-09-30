import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import starlight from "@astrojs/starlight";
import react from "@astrojs/react";
import { heliaStarlight } from "@ambiqai/helia-ui/starlight";
import rehypeMermaid from "rehype-mermaid";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import redirects from "./src/data/redirects.json" with { type: "json" };
import { sections } from "./src/navigation.mjs";
import { fileURLToPath } from "node:url";
export default defineConfig({
  vite: {
    plugins: [
      {
        name: "heartkit-section-membership",
        enforce: "pre",
        resolveId(source, importer) {
          // Historical page URLs do not share the section's URL prefix.
          if (
            source === "./sections" &&
            importer?.includes("/@ambiqai/helia-ui/starlight/")
          )
            return fileURLToPath(
              new URL("./src/section-matcher.ts", import.meta.url),
            );
        },
      },
    ],
  },
  site: "https://ambiqai.github.io",
  base: "/heartkit",
  redirects: {
    ...redirects,
    "/api": "/heartkit/reference/",
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, [rehypeMermaid, { strategy: "inline-svg" }]],
    }),
  },
  integrations: [
    react(),
    starlight({
      components: { Hero: "./src/components/HomeHero.astro" },
      title: "heartKIT",
      description: "AI development kit for heart monitoring on Ambiq devices.",
      favicon: "/assets/favicon.png",
      customCss: [
        "./src/styles/site.css",
        "@ambiqai/helia-ui/mermaid.css",
        "katex/dist/katex.min.css",
      ],
      plugins: [
        heliaStarlight({
          accent: "kit-heart",
          sections,
          sidebar: "always",
          header: {
            title: "heartKIT",
            hub: {
              label: "HELIA",
              href: "https://ambiqai.github.io/helia-developer-hub/",
            },
          },
          discoverability: {
            markdown: true,
            llms: true,
            jsonLd: true,
            ogImage: true,
          },
          footer: {
            logo: "ambiq",
            tagline: "Part of the Ambiq HELIA AI platform",
            links: [
              { label: "heartKIT source on GitHub", href: "https://github.com/AmbiqAI/heartkit" },
            ],
          },
        }),
      ],
    }),
  ],
});
