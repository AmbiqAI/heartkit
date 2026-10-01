import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join, basename } from "node:path";

const walk = (dir) => readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
  entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
);
rmSync("public/examples", { recursive: true, force: true });
mkdirSync("public/examples", { recursive: true });
const downloads = new Map();
for (const file of walk("src/content/docs").filter((path) => path.endsWith(".mdx"))) {
  const source = readFileSync(file, "utf8");
  for (const match of source.matchAll(/<ConfigExample code=\{("(?:[^"\\]|\\.)*")\} lang="[^"]+" filename=("(?:[^"\\]|\\.)*")/g)) {
    const code = JSON.parse(match[1]);
    const name = JSON.parse(match[2]);
    if (basename(name) !== name) throw new Error(`Invalid example filename in ${file}`);
    if (downloads.has(name) && downloads.get(name) !== code) throw new Error(`Conflicting example: ${name}`);
    if (name.endsWith(".json")) JSON.parse(code);
    downloads.set(name, code);
  }
}
for (const [name, code] of downloads) writeFileSync(join("public/examples", name), code + "\n");
console.log(`Generated ${downloads.size} downloads from authored examples.`);
