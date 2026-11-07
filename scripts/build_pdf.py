#!/usr/bin/env python3
"""
Build a preprint-style PDF of the English edition.

The chapters in en/ are concatenated in book order, lightly preprocessed for
LaTeX, converted with pandoc and compiled with tectonic (XeLaTeX). The LaTeX
source and its figures are kept next to the PDF, so the same folder can be
uploaded to arXiv, which wants the source rather than a TeX-built PDF.

Requires pandoc, tectonic and rsvg-convert (brew install pandoc tectonic librsvg).

Usage (from the repo root):
    python scripts/build_pdf.py            # -> build/pdf/offline-rl-book.pdf
    python scripts/build_pdf.py --tex-only # LaTeX source only
"""
import argparse
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent / "pdf"
SITE = "https://corba777.github.io"
CHAPTERS = [f"en/chapter{i}.md" for i in range(1, 14)] + ["en/appendix.md"]


def need(tool):
    if shutil.which(tool) is None:
        sys.exit(f"{tool} not found; install it with: brew install pandoc tectonic librsvg")


def preprocess(text, figdir):
    """Turn one chapter's Markdown into pandoc-friendly Markdown for LaTeX."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)                  # front matter
    text = re.sub(r"^# (?:Chapter \d+|Appendix): (.*)$", r"# \1", text, count=1, flags=re.M)
    text = re.sub(r"^---\s*$", "", text, flags=re.M)                          # horizontal rules
    text = text.replace("📄 ", "")
    # GitHub-style ```math fences (Chapter 2) become display math.
    text = re.sub(r"^```math\n(.*?)\n```$", lambda m: "$$" + m.group(1).strip() + "$$",
                  text, flags=re.S | re.M)

    def figure(m):
        alt, path = m.group(1), m.group(2)
        src = (ROOT / "en" / path).resolve()
        if src.suffix == ".svg":
            dst = figdir / (src.parent.name + "_" + src.stem + ".pdf")
            subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(dst), str(src)], check=True)
        else:
            dst = figdir / (src.parent.name + "_" + src.name)
            shutil.copy(src, dst)
        return f"![{alt}](figures/{dst.name}){{width=95%}}"

    text = re.sub(r"!\[([^\]]*)\]\((\.\./figures/[^)]+)\)", figure, text)
    text = re.sub(r"\]\((/offline-rl-book/[^)]*)\)", lambda m: f"]({SITE}{m.group(1)})", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "pdf"))
    ap.add_argument("--tex-only", action="store_true")
    args = ap.parse_args()

    need("pandoc"); need("rsvg-convert")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    figdir = out / "figures"
    figdir.mkdir(parents=True)

    parts = []
    for rel in CHAPTERS:
        md = preprocess((ROOT / rel).read_text(encoding="utf-8"), figdir)
        if rel.endswith("appendix.md"):
            md = "\n```{=latex}\n\\appendix\n```\n\n" + md
        parts.append(md)
    (out / "book.md").write_text("\n\n".join(parts), encoding="utf-8")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "unknown"
    date = datetime.date.today().strftime("%B %-d, %Y")
    title = (HERE / "title.tex").read_text().replace("$date$", date).replace("$commit$", commit)
    (out / "title.tex").write_text(title)
    shutil.copy(HERE / "preamble.tex", out / "preamble.tex")

    subprocess.run([
        "pandoc", "book.md", "-o", "offline-rl-book.tex", "--standalone",
        "--from", "markdown+tex_math_dollars+pipe_tables+lists_without_preceding_blankline-implicit_figures+raw_tex",
        "--top-level-division=chapter", "--number-sections", "--toc", "--toc-depth=1",
        "--include-in-header=preamble.tex", "--include-before-body=title.tex",
        "--highlight-style=tango",
        "-V", "documentclass=report", "-V", "classoption=11pt",
        "-V", "geometry:margin=1in", "-V", "colorlinks=true",
        "-V", "linkcolor=blue!50!black", "-V", "urlcolor=blue!50!black",
        "-M", "pagetitle=Offline RL: From Theory to Industrial Practice",
        "-M", "author-meta=Artem Zvyagintsev",
    ], cwd=out, check=True)

    # Long inline code (paths, calls) may break after / _ . = instead of overflowing.
    tex = out / "offline-rl-book.tex"
    def breakable(m):
        body = m.group(1)
        for ch in ("/", r"\_", ".", "="):
            body = body.replace(ch, ch + r"\allowbreak{}")
        return r"\texttt{" + body + "}"
    tex.write_text(re.sub(r"\\texttt\{([^{}]*)\}", breakable, tex.read_text(encoding="utf-8")),
                   encoding="utf-8")

    if args.tex_only:
        print(f"LaTeX source: {out / 'offline-rl-book.tex'}")
        return
    need("tectonic")
    subprocess.run(["tectonic", "--keep-logs", "offline-rl-book.tex"], cwd=out, check=True)
    print(f"PDF: {out / 'offline-rl-book.pdf'}")


if __name__ == "__main__":
    main()
