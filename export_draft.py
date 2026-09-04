#!/usr/bin/env python
"""Export a notebook's prose to .docx for text editing, then paste the edits back.

Produces a Word document containing only what you would actually edit: the markdown
prose, the figure images, the `#| fig-cap:` captions, and any printed output the
prose cites. Code cell *source* is dropped -- it is noise when you are editing text.

Every prose block is preceded by a marker such as [[cell 12]] giving its index in
the notebook, so an edited paragraph can be pasted back into the right cell. Search
for "[[cell" in Word to walk them.

Quarto syntax (`:::{.callout-note}`, `@fig-x`, `$math$`) is left verbatim rather
than prettified, so that text pasted back is valid notebook source as-is.

Math is left to pandoc by default, which converts it to native Word equations.
Pass --dollar-math to emit it as literal ``$$ ... $$`` text instead, for Google
Docs extensions that render markdown math themselves. (AutoLaTeX crashed on that
form; the default is the safe one.)

Usage:
    python export_draft.py index.ipynb [more.ipynb ...]     # -> draft_export/*.docx
    python export_draft.py index.ipynb --dollar-math        # literal $$...$$ math
"""

from __future__ import annotations

import base64
import json
import pathlib
import re
import subprocess
import sys

OUT_DIR = pathlib.Path("draft_export")
MAX_STREAM_LINES = 25  # printed output is context, not the thing being edited

FIG_CAP = re.compile(r'^#\|\s*fig-cap:\s*"(.*)"\s*$', re.M)
LABEL = re.compile(r"^#\|\s*label:\s*(\S+)\s*$", re.M)

MATH_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
MATH_INLINE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")


def wrap_math(text: str) -> str:
    r"""Rewrite every math span as ``$$ ... $$`` inside a code span.

    Both delimiters become ``$$`` so a Google Docs markdown-rendering extension
    picks up inline and display math alike. The surrounding backticks are what
    make the LaTeX survive pandoc byte-exact -- without them the markdown reader
    eats ``\mathbf``, ``\underbrace``, ``\;`` and ``\,``, and turns ``_..._``
    into emphasis. Escaping the dollars instead (``\$\$``) is lossy for the same
    reason. Display math is collapsed onto one line because a code span cannot
    contain a line break; LaTeX is whitespace-insensitive apart from ``\\`` row
    separators, which survive.
    """

    def repl(match: re.Match) -> str:
        body = " ".join(match.group(1).split())
        tick = "``" if "`" in body else "`"
        pad = " " if body.startswith("`") or body.endswith("`") else ""
        return f"{tick}{pad}$${body}$${pad}{tick}"

    return MATH_INLINE.sub(repl, MATH_DISPLAY.sub(repl, text))


def _images(cell, cell_idx, media_dir):
    """Write each PNG output to disk; return markdown image references."""
    refs = []
    for out_idx, output in enumerate(cell.get("outputs", [])):
        png = output.get("data", {}).get("image/png")
        if png is None:
            continue
        path = media_dir / f"cell{cell_idx:02d}_{out_idx}.png"
        path.write_bytes(base64.b64decode(png))
        refs.append(f"![]({path.name}){{width=100%}}")
    return refs


def _streams(cell):
    """Printed stdout, truncated -- some prose quotes these numbers."""
    text = "".join(
        "".join(o.get("text", [])) for o in cell.get("outputs", []) if o.get("output_type") == "stream"
    ).rstrip()
    if not text:
        return None
    lines = text.splitlines()
    if len(lines) > MAX_STREAM_LINES:
        lines = lines[:MAX_STREAM_LINES] + [f"... ({len(text.splitlines()) - MAX_STREAM_LINES} more lines)"]
    return "```\n" + "\n".join(lines) + "\n```"


def notebook_to_markdown(nb_path: pathlib.Path, media_dir: pathlib.Path, dollar_math: bool = False) -> str:
    math = wrap_math if dollar_math else (lambda t: t)
    cells = json.loads(nb_path.read_text())["cells"]
    parts = [f"# {nb_path.name} — prose export\n"]
    for i, cell in enumerate(cells):
        source = "".join(cell["source"])
        if cell["cell_type"] == "raw":
            continue  # YAML front matter; not prose
        if cell["cell_type"] == "markdown":
            parts += [f"\n**[[cell {i}]]**\n", math(source.rstrip())]
            continue
        # code cell: keep the figure, its caption, and any printed numbers
        refs = _images(cell, i, media_dir)
        cap = FIG_CAP.search(source)
        stream = _streams(cell)
        if not (refs or cap or stream):
            continue
        label = LABEL.search(source)
        parts.append(f"\n**[[cell {i}]]**{f'  `{label.group(1)}`' if label else ''}\n")
        parts += refs
        if cap:
            parts.append(
                f"\n*Caption (edit here, lives in the `#| fig-cap:` line):* {math(cap.group(1))}\n"
            )
        if stream:
            parts.append(stream)
    return "\n\n".join(parts) + "\n"


def export(nb_path: pathlib.Path, dollar_math: bool = False) -> pathlib.Path:
    media_dir = OUT_DIR / f"{nb_path.stem}_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"{nb_path.stem}.md"
    md_path.write_text(notebook_to_markdown(nb_path, media_dir, dollar_math))
    docx_path = OUT_DIR / f"{nb_path.stem}.docx"
    subprocess.run(
        [
            "pandoc",
            str(md_path),
            "-o",
            str(docx_path),
            # '@fig-x' is a Quarto cross-reference, not a pandoc citation
            "-f",
            "markdown-citations",
            "--resource-path",
            str(media_dir),
            "--toc",
        ],
        check=True,
    )
    return docx_path


if __name__ == "__main__":
    args = sys.argv[1:]
    dollar_math = "--dollar-math" in args
    targets = [a for a in args if not a.startswith("-")] or ["index.ipynb"]
    OUT_DIR.mkdir(exist_ok=True)
    for name in targets:
        path = pathlib.Path(name)
        docx = export(path, dollar_math)
        mode = "$$-wrapped literal math" if dollar_math else "native Word equations"
        print(f"{path}  ->  {docx}  ({docx.stat().st_size / 1e6:.2f} MB, {mode})")
