/** Normalize Material block syntax before the shared MDX converter runs. */
export function normalizeMarkdown(text) {
  const lines = text.split("\n");
  function blocks(input) {
    const out = [];
    let fence;
    for (let i = 0; i < input.length; i++) {
      const line = input[i];
      const token = line.match(/^\s*(`{3,}|~{3,})/);
      if (token) {
        if (!fence) fence = token[1];
        else if (token[1].startsWith(fence)) fence = undefined;
        out.push(line);
        continue;
      }
      if (fence) {
        out.push(line);
        continue;
      }
      const match = line.match(
        /^([ \t]*)(!!!|\?\?\?\+?)\s+([\w-]+)(?:\s+(.*))?$/,
      );
      if (match) {
        const [, indent, marker, rawType, rawTitle = ""] = match;
        const type = rawType.toLowerCase();
        const title = rawTitle.replace(/^["']|["']$/g, "");
        const body = [];
        let j = i + 1;
        while (
          j < input.length &&
          (!input[j].trim() || input[j].startsWith(indent + "    "))
        ) {
          body.push(input[j].trim() ? input[j].slice(indent.length + 4) : "");
          j++;
        }
        const inner = blocks(body).join("\n").trim();
        if (marker.startsWith("???")) {
          const label = title || type[0].toUpperCase() + type.slice(1);
          out.push(
            "",
            `${indent}<details${marker.endsWith("+") ? " open" : ""}>`,
            `${indent}<summary>${label}</summary>`,
            "",
            ...inner.split("\n").map((l) => indent + l),
            "",
            `${indent}</details>`,
            "",
          );
        } else if (["example", "install"].includes(type)) {
          if (title) out.push("", `${indent}**${title}**`, "");
          out.push(...inner.split("\n").map((l) => indent + l), "");
        } else {
          out.push(
            `${indent}!!! ${type}${title ? " " + JSON.stringify(title) : ""}`,
            "",
            ...inner.split("\n").map((l) => indent + "    " + l),
            "",
          );
        }
        i = j - 1;
        continue;
      }
      out.push(
        line.replace(
          /(!\[[^\]]*\]\([^)]*\))\{\s*width=["']?(\d+)["']?\s*\}/g,
          (_, image, width) => {
            const [, alt, src] = image.match(/!\[([^\]]*)\]\(([^)]*)\)/);
            return `<img src="${src}" alt="${alt}" width="${width}" />`;
          },
        ),
      );
    }
    return out;
  }
  return blocks(lines).join("\n");
}

/** Keep snippet indentation local to the directive's line. */
export function expandSnippets(text, read, seen = []) {
  return text.replace(
    /^([ \t]*)--8<--[ \t]+"([^"]+)"[ \t]*$/gm,
    (_, indent, file) => {
      if (seen.includes(file)) throw Error(`Cyclic snippet: ${file}`);
      return expandSnippets(read(file), read, [...seen, file])
        .split("\n")
        .map((line) => (line ? indent + line : ""))
        .join("\n");
    },
  );
}
