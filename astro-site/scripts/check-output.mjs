import { internalPages, isPrivateModule } from "./public-docs.mjs";
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { resolve, relative, join } from "node:path";
import { parseHTML } from "linkedom";
import assert from "node:assert/strict";
const root = resolve("dist");
const files = (dir) =>
  readdirSync(dir).flatMap((n) =>
    statSync(join(dir, n)).isDirectory() ? files(join(dir, n)) : [join(dir, n)],
  );
const errors = [];
const cache = new Map();
const doc = (file) => {
  if (!cache.has(file))
    cache.set(file, parseHTML(readFileSync(file, "utf8")).document);
  return cache.get(file);
};
for (const file of files(root).filter((f) => f.endsWith(".html"))) {
  const from = new URL(
    "/heartkit/" + relative(root, file).replace(/index\.html$/, ""),
    "https://ambiqai.github.io",
  );
  for (const a of doc(file).querySelectorAll(
    "a[href], img[src], iframe[src]",
  )) {
    const raw = a.getAttribute("href") ?? a.getAttribute("src");
    if (!raw || raw.startsWith("mailto:")) continue;
    const url = new URL(raw, from);
    if (url.origin !== from.origin || !url.pathname.startsWith("/heartkit/"))
      continue;
    const path = resolve(
      root,
      decodeURIComponent(url.pathname.slice("/heartkit/".length)),
    );
    const target = [path, join(path, "index.html"), path + ".html"].find(
      (p) => existsSync(p) && statSync(p).isFile(),
    );
    if (!target) errors.push(`${relative(root, file)} -> ${raw}`);
    else if (
      url.hash &&
      target.endsWith(".html") &&
      !doc(target).getElementById(decodeURIComponent(url.hash.slice(1)))
    )
      errors.push(`${relative(root, file)} -> missing anchor ${raw}`);
  }
}
const index = JSON.parse(readFileSync("src/data/api-index.json", "utf8"));
assert(index.rows.length >= 100, "API coverage unexpectedly reduced");
assert(index.rows.every((row) => !isPrivateModule(row.module)), "Private API in catalog");
const apiModel = JSON.parse(readFileSync("dist/reference/api/reference.json", "utf8"));
function checkPublic(module) {
  assert(!isPrivateModule(module.path), `Private module published: ${module.path}`);
  (module.submodules ?? []).forEach(checkPublic);
}
apiModel.modules.forEach(checkPublic);
for (const file of files(root)) {
  const route = relative(root, file).replaceAll("\\", "/");
  assert(!route.split("/").some((part) => part.startsWith("_") && !part.startsWith("_astro")), `Private route published: ${route}`);
}
assert(existsSync("dist/llms-full.txt"), "Missing LLM export");
assert(existsSync("dist/pagefind/pagefind.js"), "Missing search index");
assert(
  existsSync("dist/notebooks/train-arrhythmia-model.ipynb"),
  "Missing notebook download",
);
if (errors.length)
  throw Error(
    `Broken internal links (${errors.length}):\n${[...new Set(errors)].join("\n")}`,
  );
console.log(
  `Verified internal links across ${cache.size} pages, ${index.rows.length} API symbols and discovery exports.`,
);

for (const path of files(resolve("src/content/docs")).filter((p) => /\.mdx?$/.test(p))) {
  const rel = relative(resolve("src/content/docs"), path).replace(/(?:index)?\.mdx?$/, "");
  assert(existsSync(join(root, rel, "index.html")), `Missing authored route: ${rel}`);
}
for (const path of internalPages) {
  const rel = path.replace(/(?:index)?\.md$/, "");
  assert(!existsSync(join(root, rel, "index.html")), `Internal page published: ${rel}`);
}
const reference = readFileSync(
  "dist/reference/api/heartkit/defines/index.md",
  "utf8",
);
assert(
  reference.includes("TaskParams"),
  "API Markdown lost parameter documentation",
);

for (const file of files(root).filter((f) => f.endsWith("/index.html"))) {
  const content = doc(file).querySelector(".sl-markdown-content");
  if (!content) continue;
  const prose = content.cloneNode(true);
  prose
    .querySelectorAll("pre, code, script, style")
    .forEach((node) => node.remove());
  assert(
    !/(?:^|\n)\s*(?:!!!|\?\?\?|===|--8<--)/m.test(prose.textContent),
    `Unconverted Material block: ${relative(root, file)}`,
  );
  assert(
    !/\{\s*width\s*=/.test(prose.textContent),
    `Leaked image attribute: ${relative(root, file)}`,
  );
  for (const p of prose.querySelectorAll("p"))
    assert(
      !/^\s*\|.+\|/s.test(p.textContent),
      `Unrendered table: ${relative(root, file)}`,
    );
}
for (const file of files(root).filter(
  (f) => f.endsWith("/index.html") && f !== join(root, "index.html"),
)) {
  const document = doc(file);
  if (!document.querySelector(".sl-markdown-content")) continue;
  assert(
    document.querySelector("[data-helia-sidebar-heading]"),
    `Page has no assigned section: ${relative(root, file)}`,
  );
}

for (const file of files(root).filter((f) => f.endsWith("/index.html"))) {
  const document = doc(file);
  const prose = document.querySelector("main")?.cloneNode(true);
  if (!prose) continue;
  prose
    .querySelectorAll("pre, code, script, style")
    .forEach((node) => node.remove());
  assert(
    !/:(?:material|simple|fontawesome|octicons)-[\w-]+:/.test(
      prose.textContent,
    ),
    `Unconverted icon shortcode: ${relative(root, file)}`,
  );
  assert(
    !/:(?:material|simple|fontawesome|octicons)-[\w-]+:/.test(document.title),
    `Icon shortcode in page title: ${relative(root, file)}`,
  );
}

for (const file of files(join(root, "examples")).filter((f) =>
  f.endsWith(".json"),
)) {
  JSON.parse(readFileSync(file, "utf8"));
}

for (const name of readdirSync("../notebooks").filter(n=>n.endsWith(".ipynb"))) {
 const original=readFileSync("../notebooks/"+name);
 assert.deepEqual(original,readFileSync("dist/notebooks/"+name));
 const notebook=JSON.parse(original);
 const expected=notebook.cells.flatMap(c=>c.outputs??[]).filter(o=>o.data?.["image/png"]).length;
 const page=doc(join(root,"guides",name.replace(".ipynb",""),"index.html"));
 assert.equal(page.querySelectorAll('img[src*="/notebooks/"]').length,expected);
 assert(!page.querySelector("main").textContent.includes("\x1b"));
}

const legacy = JSON.parse(readFileSync("scripts/legacy-routes.json", "utf8"));
for (const route of legacy.routes)
  assert(existsSync(join(root, route, "index.html")), `Missing published route: ${route}`);
console.log(`Verified all ${legacy.routes.length} original published routes remain available.`);

for (const path of files(resolve("../notebooks")).filter((p) => p.endsWith(".ipynb"))) {
  const notebook = JSON.parse(readFileSync(path, "utf8"));
  for (const cell of notebook.cells) {
    if (cell.cell_type !== "markdown") continue;
    const source = Array.isArray(cell.source) ? cell.source.join("") : cell.source;
    for (const match of source.matchAll(/(?:github\.com|github)\/AmbiqAI\/heartkit\/blob\/main\/([^\s)"<>]+)/g)) {
      assert(existsSync(resolve("..", match[1])), `Notebook source link is missing: ${path} -> ${match[1]}`);
    }
  }
}
