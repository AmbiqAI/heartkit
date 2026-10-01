import { internalPages } from "./public-docs.mjs";
import {
  existsSync,
  readFileSync,
  writeFileSync,
  mkdirSync,
  readdirSync,
  cpSync,
  rmSync,
} from "node:fs";
import { dirname, resolve, relative } from "node:path";
import {
  convertPage,
  createState,
  parseNav,
  buildSidebar,
} from "../node_modules/@ambiqai/helia-ui/scripts/lib/mkdocs-convert-render.mjs";
import { normalizeMarkdown, expandSnippets } from "./normalize-markdown.mjs";
const source = resolve("../docs");
const out = resolve("src/content/docs");
const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(resolve(dir, e.name)) : [resolve(dir, e.name)],
  );
const state = createState();
const titles = {};
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
rmSync("public/examples", { recursive: true, force: true });
mkdirSync("src/data", { recursive: true });
function readSnippet(file) {
  const path = resolve(source, file);
  if (!path.startsWith(source + "/"))
    throw Error(`Snippet outside docs: ${file}`);
  if (!existsSync(path)) throw Error(`Missing snippet: ${file}`);
  if (file.endsWith(".html"))
    return `<iframe title="Interactive example" src="/heartkit/${file}" loading="lazy" style={{width:'100%',height:'480px',border:0}} />`;
  return readFileSync(path, "utf8");
}
for (const file of walk(source)) {
  const rel = relative(source, file);
  if (/^(css|js|overrides)\//.test(rel) || internalPages.has(rel)) continue;
  if (!rel.endsWith(".md")) {
    if (!rel.endsWith(".ipynb")) {
      mkdirSync(dirname("public/" + rel), { recursive: true });
      cpSync(file, "public/" + rel);
    }
    continue;
  }
  if (rel.startsWith("assets/")) continue;
  let raw = expandSnippets(
    readFileSync(file, "utf8").replace(/<!--[\s\S]*?-->/g, ""),
    readSnippet,
  );
  raw = raw.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (all, label, href) => {
    if (/^(https?:|mailto:|#|\/)/.test(href)) return all;
    const target = relative(source, resolve(dirname(file), href.split("#")[0]));
    return internalPages.has(target) ? label : all;
  });
  raw = raw.replace(/\]\(([^)]+)\)/g, (all, href) => {
    if (/^(https?:|mailto:|#|\/)/.test(href)) return all;
    const [path, hash] = href.split("#");
    const target = relative(
      source,
      resolve(dirname(file), path.replace(/\.md\/$/, ".md")),
    );
    if (target.startsWith("../"))
      return `](https://github.com/AmbiqAI/heartkit/blob/main/${relative(resolve(".."), resolve(dirname(file), path))}${hash ? "#" + hash : ""})`;
    const route = target.replace(/(?:index)?\.md$/, "").replace(/\.ipynb$/, "");
    const suffix =
      /\.(md|ipynb)\/?$/.test(path) && route && !route.endsWith("/") ? "/" : "";
    return `](/heartkit/${route}${suffix}${hash ? "#" + hash : ""})`;
  });
  raw = raw.replace(/^#\s*$/m, "# heartKIT");
  raw = raw.replace(/:(?:material|simple|fontawesome|octicons)-[\w-]+:/g, "");
  raw = raw.replace(/\]\(([^)]+)\.ipynb\)/g, "]($1/)");
  raw = raw.replace(
    /\[([^\]]+)\]\(([^)]+)\)\{\s*\.md-button\s*\}/g,
    '<a className="md-button" href="$2">$1</a>',
  );
  const intro = rel === "index.md" ? raw.match(/<div class="heartkit-intro">\s*([\s\S]*?)\s*<\/div>/)?.[1].trim() : undefined;
  if (intro) raw = raw.replace(/<div class="heartkit-intro">[\s\S]*?<\/div>/, "");
  raw = normalizeMarkdown(raw);
  const page = convertPage(raw, rel, { state, base: "/heartkit" });
  let text = page.text
    .replace(/#only-light/g, "#only-light")
    .replace(/<br\s*>/g, "<br />");
  if (rel === "index.md")
    text = text.replace(
      /^description:.*$/m,
      'description: \"AI development kit for heart monitoring on Ambiq devices.\"',
    );
  if (intro) text = text.replace(/^description:.*$/m, (line) => `${line}\nhero:\n  tagline: ${JSON.stringify(intro)}`);
  let blockIndex = 0;
  text = text.replace(
    /^([ \t]*)(`{3,})([\w+-]+)([^\n]*)\n([\s\S]*?)^\1\2[ \t]*$/gm,
    (all, indent, fence, lang, meta, body) => {
      if (["mermaid", "text"].includes(lang)) return all;
      const extension =
        {
          python: "py",
          py: "py",
          bash: "sh",
          sh: "sh",
          console: "sh",
          json: "json",
          jsonc: "jsonc",
          yaml: "yaml",
          yml: "yaml",
        }[lang] || lang;
      const code = body
        .split("\n")
        .map((l) => (l.startsWith(indent) ? l.slice(indent.length) : l))
        .join("\n")
        .trimEnd();
      if (lang === "json") {
        try {
          JSON.parse(code);
        } catch (error) {
          throw new Error(`Invalid JSON example in ${rel}: ${error.message}`);
        }
      }
      if (meta.includes("fragment"))
        return `${indent}${fence}${lang} title="Configuration fragment"\n${body}${indent}${fence}`;
      const lines = code.split("\n").length;
      const name = `${rel.replace(/\.md$/, "").replaceAll("/", "-")}-${++blockIndex}.${extension}`;
      if (
        (["json", "jsonc", "yaml", "yml"].includes(lang) && lines > 16) ||
        (["py", "python"].includes(lang) && lines > 45)
      ) {
        mkdirSync("public/examples", { recursive: true });
        writeFileSync("public/examples/" + name, code + "\n");
        return `${indent}<ConfigExample code={${JSON.stringify(code)}} lang=${JSON.stringify(lang === "py" ? "python" : lang)} filename=${JSON.stringify(name)} download=${JSON.stringify("/heartkit/examples/" + name)}>\n\n${indent}${fence}${lang} title="${name}"\n${body}${indent}${fence}\n\n${indent}</ConfigExample>`;
      }
      const title =
        lang === "py" || lang === "python"
          ? "Python example"
          : ["bash", "sh", "console"].includes(lang)
            ? "Terminal"
            : lang.toUpperCase() + " example";
      return `${indent}${fence}${lang}${meta.includes("title=") ? meta : ' title="' + title + '"' + meta}\n${body}${indent}${fence}`;
    },
  );
  if (text.includes("<ConfigExample "))
    text = text.replace(
      /(---\n[\s\S]*?\n---)/,
      '$1\n\nimport ConfigExample from "' +
        relative(
          dirname(resolve(out, rel)),
          resolve("src/components/ConfigExample.astro"),
        ).replaceAll("\\", "/") +
        '";',
    );
  const dest = resolve(out, rel.replace(/\.md$/, ".mdx"));
  mkdirSync(dirname(dest), { recursive: true });
  writeFileSync(dest, text);
  titles[rel] = page.title;
  for (const warning of page.handPass)
    state.handPass.push(`${rel}: ${warning}`);
}
const nav = parseNav(readFileSync("../mkdocs.yml", "utf8"));
writeFileSync(
  "src/data/sidebar.json",
  JSON.stringify(buildSidebar(nav, titles), null, 2),
);
mkdirSync(".cache", { recursive: true });
writeFileSync(".cache/conversion.json", JSON.stringify(state, null, 2));
console.log(
  `Converted ${state.counts.pages} authored pages; manual checks recorded in .cache/conversion.json.`,
);
