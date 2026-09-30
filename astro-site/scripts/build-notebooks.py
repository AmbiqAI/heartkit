"""Render saved notebook outputs without executing training code."""

import base64
import json
import re
import shutil
from pathlib import Path

assets = Path("public/notebooks")
shutil.rmtree(assets, ignore_errors=True)
assets.mkdir(parents=True)
for source in sorted(Path("../docs/guides").glob("*.ipynb")):
    notebook = json.loads(source.read_text())
    shutil.copyfile(source, assets / source.name)
    title = next(
        (
            re.search(r"^# (.+)", "".join(c.get("source", [])), re.MULTILINE).group(1)
            for c in notebook["cells"]
            if c["cell_type"] == "markdown" and re.search(r"^# (.+)", "".join(c.get("source", [])), re.MULTILINE)
        ),
        source.stem.replace("-", " ").title(),
    )
    parts = [
        f"---\ntitle: {json.dumps(title)}\ndescription: Saved heartKIT notebook example with code and outputs.\n---",
        f'<div class="heartkit-actions"><a class="md-button" href="/heartkit/notebooks/{source.name}">Download notebook</a> <a class="md-button" href="https://github.com/AmbiqAI/heartkit/blob/main/notebooks/{source.name}">View source</a> <a class="md-button" href="https://colab.research.google.com/github/AmbiqAI/heartkit/blob/main/notebooks/{source.name}">Open in Colab</a></div>',
        "This example displays saved outputs. Building the documentation does not run training. Check dataset paths for your notebook working directory before running.",
    ]
    for ci, cell in enumerate(notebook["cells"]):
        text = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            text = re.sub(
                r'<div\b[^>]*class="grid cards"[^>]*>.*?</div>',
                lambda match: "" if "View in Colab" in match[0] or "colab-badge.svg" in match[0] else match[0],
                text,
                flags=re.DOTALL,
            )
            text = re.sub(r"^# .+\n?", "", text, flags=re.MULTILINE)
            text = re.sub(r":(?:material|simple|fontawesome|octicons)-[\w-]+:", "", text)
            text = re.sub(r"\{\s*\.[^}]+\}", "", text)
            parts.append(text)
        elif cell["cell_type"] == "code":
            parts.append("```python\n" + text.rstrip() + "\n```")
            for oi, output in enumerate(cell.get("outputs", [])):
                data = output.get("data", {})
                if "image/png" in data:
                    name = f"{source.stem}-{ci}-{oi}.png"
                    (assets / name).write_bytes(base64.b64decode("".join(data["image/png"])))
                    parts.append(f"![Saved figure from cell {ci + 1}](/heartkit/notebooks/{name})")
                else:
                    raw = "".join(output.get("text", data.get("text/plain", [])))
                    raw = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", raw)
                    raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
                    if raw.strip():
                        block = "```text\n" + raw.rstrip() + "\n```"
                        parts.append(
                            "<details>\n<summary>Saved output</summary>\n\n" + block + "\n\n</details>"
                            if len(raw.splitlines()) > 12
                            else block
                        )
    Path(f"src/content/docs/guides/{source.stem}.md").write_text("\n\n".join(parts) + "\n")
