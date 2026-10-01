"""Check notebook prose survives removal of the replaced navigation toolbar."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("build-notebooks.py")
GUIDES = SCRIPT.parents[2] / "docs" / "guides"


class NotebookRenderingTest(unittest.TestCase):
    def test_badge_toolbar_preserves_byot_introduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(GUIDES, root / "docs" / "guides")
            site = root / "site"
            pages = site / "src" / "content" / "docs" / "guides"
            pages.mkdir(parents=True)
            subprocess.run([sys.executable, str(SCRIPT)], cwd=site, check=True)

            rendered = (pages / "byot.md").read_text()
            original = json.loads((GUIDES / "byot.ipynb").read_text())
            introduction = "".join(original["cells"][0]["source"]).split("</div>", 1)[1]
            for paragraph in introduction.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph and not paragraph.startswith("# "):
                    self.assertIn(paragraph, rendered)
            self.assertNotIn('class="grid cards"', rendered)
            self.assertNotIn("View in Colab", rendered)
            self.assertIn("Open in Colab", rendered)

    def test_colab_mention_does_not_drop_ordinary_prose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guides = root / "docs" / "guides"
            guides.mkdir(parents=True)
            prose = "Choose View in Colab to run this example.\n\nKeep your dataset paths configured."
            (guides / "example.ipynb").write_text(json.dumps({
                "cells": [{"cell_type": "markdown", "source": ["# Example\n\n" + prose]}],
            }))
            site = root / "site"
            pages = site / "src" / "content" / "docs" / "guides"
            pages.mkdir(parents=True)
            subprocess.run([sys.executable, str(SCRIPT)], cwd=site, check=True)
            self.assertIn(prose, (pages / "example.md").read_text())


if __name__ == "__main__":
    unittest.main()
