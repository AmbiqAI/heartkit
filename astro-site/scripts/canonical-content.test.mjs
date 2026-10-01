import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import test from "node:test";

const walk = (dir) => readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
  entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
);
const snapshot = () => Object.fromEntries([
  ...walk("src/content/docs").filter((p) => p.endsWith(".mdx") && !p.startsWith("src/content/docs/reference/")),
  "src/navigation.mjs", "src/redirects.json",
  ...walk("public").filter((p) => !/^public\/(reference|notebooks|examples)\//.test(p)),
].map((p) => [relative(".", p), readFileSync(p).toString("base64")]));

test("generation preserves authored pages, navigation, redirects and static assets", () => {
  const before = snapshot();
  assert(Object.keys(before).length > 40, "Missing canonical content");
  execFileSync("npm", ["run", "prepare:docs"], { stdio: "pipe" });
  assert.deepEqual(snapshot(), before);
});
