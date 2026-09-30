import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeMarkdown, expandSnippets } from "./normalize-markdown.mjs";
import {
  convertPage,
  createState,
} from "../node_modules/@ambiqai/helia-ui/scripts/lib/mkdocs-convert-render.mjs";
const convert = (raw) =>
  convertPage(normalizeMarkdown(raw), "example.md", {
    state: createState(),
    base: "/heartkit",
  }).text;
test("snippet expansion keeps table rows adjacent and preserves surrounding blank lines", () => {
  const table = "| Class | Label |\n| --- | --- |\n| 0 | Wake |";
  assert.equal(
    expandSnippets('Before\n\n--8<-- "table.md"\n\nAfter', () => table),
    `Before\n\n${table}\n\nAfter`,
  );
  assert.equal(
    expandSnippets('    --8<-- "table.md"', () => table),
    table
      .split("\n")
      .map((l) => "    " + l)
      .join("\n"),
  );
  assert.throws(
    () => expandSnippets('--8<-- "loop"', () => '--8<-- "loop"'),
    /Cyclic/,
  );
});
test("mixed-case examples and installation wrappers become ordinary content", () => {
  for (const type of ["Example", "example", "install"]) {
    const result = convert(
      `# Test\n\n!!! ${type} "Example title"\n\n    === "Python"\n\n        Useful example.`,
    );
    assert(!result.includes("!!!"));
    assert(!result.includes(":::note"));
    assert(result.includes("<Tabs>"));
    assert(result.includes("**Example title**"));
  }
});
test("actual notes remain callouts and collapsed admonitions retain open state", () => {
  assert(
    convert('# Test\n\n!!! Note "Important"\n\n    Keep this note.').includes(
      ":::note[Important]",
    ),
  );
  const closed = convert(
    '# Test\n\n??? Example "Advanced"\n\n    Hidden detail.',
  );
  assert(closed.includes("<details>"));
  assert(closed.includes("<summary>Advanced</summary>"));
  assert(
    convert('# Test\n\n???+ note "Expanded"\n\n    Detail.').includes(
      "<details open>",
    ),
  );
});
test("image attributes become image properties; code samples remain verbatim", () => {
  const image = normalizeMarkdown('![Watch](/watch.webp){ width="540" }');
  assert.equal(image, '<img src="/watch.webp" alt="Watch" width="540" />');
  const fenced = '```text\n!!! Example\n{ width="540" }\n```';
  assert.equal(normalizeMarkdown(fenced), fenced);
});
