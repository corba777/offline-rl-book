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
    python scripts/build_pdf.py --epub     # -> build/epub/offline-rl-book.epub (pandoc only)
    python scripts/build_pdf.py --cover    # PDF with the cover page in front
    python scripts/build_pdf.py --leanpub  # -> build/leanpub/: EPUB and PDF, both with the cover,
                                           #    and the Russian edition in build/leanpub/ru/
    python scripts/build_pdf.py --lang ru  # any of the above from ru/ (Leanpub only)

The EPUB is the ebook sold on Leanpub (pay what you want, free minimum); CI
does not build or publish it, nor the PDF with a cover page: the PDF on GitHub
starts with the title page. The Russian edition is built for Leanpub only.
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
RU_MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
             "сентября", "октября", "ноября", "декабря"]
DOI = "10.5281/zenodo.24045871"
# What differs between the English and the Russian edition.
LANG = {
    "en": dict(title="Offline RL: From Theory to Industrial Practice",
               subtitle="Practical notes on offline deep reinforcement learning",
               title_tex="title.tex", cover="cover.svg", cover_label="Cover",
               site=f"{SITE}/offline-rl-book/", about="About this book", toc="Contents",
               date=lambda d: d.strftime("%B %-d, %Y"),
               rights=["Version {version}", "Text and figures: CC BY 4.0. Code: MIT.",
                       "Free online: <{site}>", "Source and code: <https://github.com/corba777/offline-rl-book>",
                       "DOI: [{doi}](https://doi.org/{doi})", "Built from commit {commit}."]),
    "ru": dict(title="Offline RL: от теории к промышленной практике",
               subtitle="Практические заметки об offline deep RL",
               title_tex="title_ru.tex", cover="cover_ru.svg", cover_label="Обложка",
               site=f"{SITE}/offline-rl-book/ru/", about="Об этой книге", toc="Содержание",
               date=lambda d: f"{d.day} {RU_MONTHS[d.month - 1]} {d.year} г.",
               rights=["Версия {version}", "Текст и рисунки: CC BY 4.0. Код: MIT.",
                       "Бесплатно онлайн: <{site}>", "Исходники и код: <https://github.com/corba777/offline-rl-book>",
                       "DOI: [{doi}](https://doi.org/{doi})", "Собрано из коммита {commit}."]),
}


def chapters(lang):
    return [f"{lang}/chapter{i}.md" for i in range(1, 14)] + [f"{lang}/appendix.md"]


def edition():
    """(version, commit) for the title pages. The version is the latest release tag,
    with the number of commits since it when the build is not exactly on a tag
    (1.1.1+6); without tags, the version in CITATION.cff."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    m = re.fullmatch(r"v?(.+?)(?:-(\d+)-g[0-9a-f]+)?", git("describe", "--tags", "--match", "v*"))
    if m:
        version = m.group(1) + (f"+{m.group(2)}" if m.group(2) else "")
    else:
        cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        version = re.search(r"^version:\s*(\S+)", cff, flags=re.M).group(1)
    return version, commit


def need(tool):
    if shutil.which(tool) is None:
        sys.exit(f"{tool} not found; install it with: brew install pandoc tectonic librsvg")


def chapter_id(rel):
    """Anchor for a chapter file: en/chapter4.md -> ch4, en/appendix.md -> appendix."""
    stem = Path(rel).stem
    return "appendix" if stem == "appendix" else "ch" + stem.removeprefix("chapter")


def preprocess(text, figdir, anchor, epub=False, raster_svg=False, lang="en"):
    """Turn one chapter's Markdown into pandoc-friendly Markdown for LaTeX (or EPUB).
    raster_svg: diagrams as 300 dpi PNGs instead of vector PDFs (see build_pdf)."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)                  # front matter
    # "Приложение: выбор алгоритма" loses its prefix, so the title gets its capital letter.
    text = re.sub(r"^# (?:Chapter \d+|Appendix|Глава \d+|Приложение): (.*)$",
                  lambda m: f"# {m.group(1)[:1].upper()}{m.group(1)[1:]} {{#{anchor}}}", text, count=1, flags=re.M)
    text = re.sub(r"^---\s*$", "", text, flags=re.M)                          # horizontal rules
    text = text.replace("📄 ", "")
    # Inside math, \_ and \* are escapes for the website's Markdown; LaTeX would print them.
    def unescape(math):
        # text-mode groups such as \text{next\_s1} need the escape and keep it
        pieces = re.split(r"(\\(?:text|mathrm|texttt|operatorname)\{[^{}]*\})", math)
        return "".join(q if j % 2 else q.replace(r"\_", "_").replace(r"\*", "*")
                       for j, q in enumerate(pieces))

    def unescape_math(chunk):
        return re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$", lambda m: unescape(m.group(0)), chunk, flags=re.S)
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    text = "".join(p if i % 2 else unescape_math(p) for i, p in enumerate(parts))
    # GitHub-style ```math fences (Chapter 2) become display math.
    text = re.sub(r"^```math\n(.*?)\n```$", lambda m: "$$" + m.group(1).strip() + "$$",
                  text, flags=re.S | re.M)
    if epub:
        # pandoc's MathML writer rejects \Bigl( ... \Bigr) and \bigl| ... \bigr| (\left/\right pairs
        # render the same delimiters), and turns \tfrac into a <mfrac displaystyle> that EPUB 3
        # validators reject (\frac reads the same on a phone screen).
        def mathml_safe(chunk):
            chunk = re.sub(r"\\[Bb]igl\b", r"\\left", chunk)
            chunk = re.sub(r"\\[Bb]igr\b", r"\\right", chunk)
            return re.sub(r"\\[td]frac\b", r"\\frac", chunk)
        parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
        text = "".join(p if i % 2 else mathml_safe(p) for i, p in enumerate(parts))

    def figure(m):
        alt, path = m.group(1), m.group(2)
        src = (ROOT / lang / path).resolve()
        if src.suffix == ".svg" and epub:
            # EPUB 3 readers render SVG; keep it vector, with the same font substitution.
            dst = figdir / (src.parent.name + "_" + src.name)
            dst.write_text(re.sub(r'font-family="[^"]*"', 'font-family="DejaVu Sans, Helvetica, Arial, sans-serif"',
                                  src.read_text(encoding="utf-8")), encoding="utf-8")
        elif src.suffix == ".svg":
            dst = figdir / (src.parent.name + "_" + src.stem + (".png" if raster_svg else ".pdf"))
            # One sans font for every diagram, whatever the build machine has installed.
            svg = re.sub(r'font-family="[^"]*"', 'font-family="DejaVu Sans, Helvetica, Arial, sans-serif"',
                         src.read_text(encoding="utf-8"))
            fmt = ["-f", "png", "-w", "1900"] if raster_svg else ["-f", "pdf"]  # ~300 dpi at 95% width
            subprocess.run(["rsvg-convert", *fmt, "-o", str(dst)], input=svg.encode(), check=True)
        else:
            dst = figdir / (src.parent.name + "_" + src.name)
            shutil.copy(src, dst)
        return f"![{alt}](figures/{dst.name})" + ("" if epub else "{width=95%}")

    text = re.sub(r"!\[([^\]]*)\]\((\.\./figures/[^)]+)\)", figure, text)
    # Links to other chapters jump inside the PDF; other site links stay absolute.
    text = re.sub(rf"\]\(/offline-rl-book/{lang}/((?:chapter\d+|appendix))\.html(?:#[^)]*)?\)",
                  lambda m: f"](#{chapter_id(m.group(1) + '.md')})", text)
    text = re.sub(r"\]\((/offline-rl-book/[^)]*)\)", lambda m: f"]({SITE}{m.group(1)})", text)
    return text


COVERS = ROOT / "figures" / "readme"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "pdf"))
    ap.add_argument("--tex-only", action="store_true")
    ap.add_argument("--epub", action="store_true", help="build the EPUB instead of the PDF")
    ap.add_argument("--cover", action="store_true", help="put the cover page in front of the PDF")
    ap.add_argument("--leanpub", action="store_true",
                    help="build the EPUB and the PDF, both with the cover, into build/leanpub/ "
                         "(English) and build/leanpub/ru/ (Russian)")
    ap.add_argument("--lang", choices=sorted(LANG), default="en", help="edition (default: en)")
    args = ap.parse_args()
    custom_out = args.out != ap.get_default("out")
    if args.leanpub:
        dest = Path(args.out) if custom_out else ROOT / "build" / "leanpub"
        for lang, sub in (("en", dest), ("ru", dest / "ru")):
            build_epub(sub / "epub", lang=lang)
            build_pdf(sub / "pdf", cover=True, raster_svg=True, lang=lang)
            for f in (sub / "epub" / "offline-rl-book.epub", sub / "epub" / "cover.png",
                      sub / "pdf" / "offline-rl-book.pdf"):
                shutil.copy(f, sub / f.name)
            (sub / "toc.html").write_text(leanpub_toc(lang), encoding="utf-8")
            print(f"Leanpub files ({lang}): {sub}")
        return
    if args.epub:
        out = Path(args.out) if custom_out else ROOT / "build" / ("epub" if args.lang == "en" else "epub-ru")
        return build_epub(out, lang=args.lang)
    out = Path(args.out) if custom_out or args.lang == "en" else ROOT / "build" / "pdf-ru"
    build_pdf(out, tex_only=args.tex_only, cover=args.cover, lang=args.lang)


def leanpub_toc(lang="en"):
    """Table of contents for the Leanpub landing page, in Leanpub's own markup:
    chapters and their sections, numbered as in the PDF (appendix A, A.1, ...).
    The per-chapter reference lists are left out."""
    import html
    skip = {"References", "Литература", "Ссылки"}
    items = []
    for k, rel in enumerate(chapters(lang), start=1):
        text = re.sub(r"```.*?```", "", (ROOT / rel).read_text(encoding="utf-8"), flags=re.S)
        title = re.search(r"^# (?:Chapter \d+|Appendix|Глава \d+|Приложение): (.*)$", text, flags=re.M).group(1)
        title = title[:1].upper() + title[1:]
        num = "A" if rel.endswith("appendix.md") else str(k)
        subs = [f'      <li>\n        <span class="section-number">{num}.{j} </span>{html.escape(h.strip())}\n      </li>'
                for j, h in enumerate(re.findall(r"^## (.+)$", text, flags=re.M), start=1)
                if h.strip() not in skip]
        sub = ("\n    <ul>\n" + "\n".join(subs) + "\n    </ul>") if subs else ""
        items.append(f'  <li>\n    <span class="section-number">{num}{"" if num == "A" else "."} </span>'
                     f'{html.escape(title)}{sub}\n  </li>')
    return '<ul class="toc no-parts">\n' + "\n".join(items) + "\n</ul>\n"


def build_pdf(out, tex_only=False, cover=False, raster_svg=False, lang="en"):
    """The preprint-style PDF; with cover=True, the cover page comes first.
    raster_svg=True renders the diagrams as images: newer cairo versions draw the
    spaces of SVG text with empty Type 3 glyphs. Used for the Leanpub PDF built
    locally; CI keeps vector diagrams."""
    need("pandoc"); need("rsvg-convert")
    if out.exists():
        shutil.rmtree(out)
    figdir = out / "figures"
    figdir.mkdir(parents=True)

    L = LANG[lang]
    parts = []
    for rel in chapters(lang):
        md = preprocess((ROOT / rel).read_text(encoding="utf-8"), figdir, chapter_id(rel),
                        raster_svg=raster_svg, lang=lang)
        if rel == chapters(lang)[0]:
            # Title page and contents are numbered i, ii, ...; Chapter 1 starts at page 1.
            md = "```{=latex}\n\\clearpage\n\\pagenumbering{arabic}\n```\n\n" + md
        if rel.endswith("appendix.md"):
            md = "\n```{=latex}\n\\appendix\n```\n\n" + md
        parts.append(md)
    (out / "book.md").write_text("\n\n".join(parts), encoding="utf-8")

    version, commit = edition()
    date = L["date"](datetime.date.today())
    title = (HERE / L["title_tex"]).read_text().replace("$date$", date).replace("$commit$", commit) \
        .replace("$version$", version)
    (out / "title.tex").write_text(title)
    (out / "preamble.tex").write_text((HERE / "preamble.tex").read_text()
                                      .replace(LANG["en"]["title"], L["title"]))
    before_body = ["--include-before-body=title.tex"]
    if cover:
        # The cover marks the ebook edition (Leanpub): no "A Preprint" label, and
        # the abstract reads as a book's "About this book".
        (out / "title.tex").write_text(title.replace("A Preprint\\quad--\\quad ", "")
                                       .replace("\\textbf{Abstract}", "\\textbf{About this book}"))
        pre = (out / "preamble.tex").read_text()
        (out / "preamble.tex").write_text(pre.replace("\\quad--\\quad A Preprint", ""))
        # The 2:3 cover art, cropped to the page's aspect ratio (its background
        # bleeds sideways), placed full-bleed before the title page. It goes in as a
        # 300 dpi image: cairo's PDF output draws spaces with Type 3 fonts.
        svg = (COVERS / L["cover"]).read_text(encoding="utf-8")
        w = 900 * 8.5 / 11                       # US Letter, the report class default
        svg = svg.replace('viewBox="0 0 600 900" width="1600" height="2400"',
                          f'viewBox="{(600 - w) / 2:.2f} 0 {w:.2f} 900" width="{w:.2f}" height="900"', 1)
        (out / "cover.svg").write_text(svg, encoding="utf-8")
        subprocess.run(["rsvg-convert", "-w", "2550", "-h", "3300", "-o", "cover.png", "cover.svg"],
                       cwd=out, check=True)
        with open(out / "preamble.tex", "a", encoding="utf-8") as fh:
            fh.write("\n\\usepackage{pdfpages}\n")
        (out / "cover.tex").write_text(f"{{\\renewcommand{{\\thepage}}{{{L['cover_label']}}}"
                                       "\\includepdf[pages=1]{cover.png}}\n")
        before_body.insert(0, "--include-before-body=cover.tex")

    subprocess.run([
        "pandoc", "book.md", "-o", "offline-rl-book.tex", "--standalone",
        "--from", "markdown+tex_math_dollars+pipe_tables+lists_without_preceding_blankline-implicit_figures+raw_tex",
        "--top-level-division=chapter", "--number-sections", "--toc", "--toc-depth=1",
        "--include-in-header=preamble.tex", *before_body,
        "--highlight-style=tango",
        "-V", "documentclass=report", "-V", "classoption=11pt",
        "-V", "geometry:margin=1in", "-V", "colorlinks=true",
        "-V", "linkcolor=blue!50!black", "-V", "urlcolor=blue!50!black",
        "-M", f"pagetitle={L['title']}", "-V", f"lang={lang}",
        "-M", "author-meta=Artem Zvyagintsev",
    ], cwd=out, check=True)

    # Long inline code (paths, calls) may break after / _ . = instead of overflowing.
    tex = out / "offline-rl-book.tex"
    def breakable(m):
        body = m.group(1)
        for ch in ("/", r"\_", ".", "="):
            body = body.replace(ch, ch + r"\allowbreak{}")
        return r"\texttt{" + body + "}"
    src = re.sub(r"\\texttt\{([^{}]*)\}", breakable, tex.read_text(encoding="utf-8"))

    # Height of a table in text lines: p{} columns wrap, so a row takes as many
    # lines as its longest cell needs at that column's share of ~95 characters.
    def table_lines(body):
        widths = [float(w) for w in re.findall(r"\\real\{([\d.]+)\}", body)]
        lines = 0
        for row in re.split(r"\\\\\*?\s*\n", body.split("\\endlastfoot")[-1]):
            cells = row.split("&")
            if not row.strip() or row.lstrip().startswith("\\bottomrule") and len(cells) == 1:
                continue
            need = 1
            for i, cell in enumerate(cells):
                if widths and i < len(widths):
                    text = re.sub(r"\\[a-zA-Z]+|[{}$]", "", cell)
                    need = max(need, -(-len(text.strip()) // max(1, int(95 * widths[i]))))
            lines += need
        return lines

    # Short tables stay on one page: forbid breaks between their rows (\\*) and
    # move the whole table to the next page if it does not fit.
    def keep_together(m):
        body = m.group(0)
        rows = len(re.findall(r"\\\\\s*$", body, flags=re.M))
        if rows > 15 or table_lines(body) > 40:   # longer than a page: let it break
            return body
        body = re.sub(r"\\\\(\s*)$", r"\\\\*\1", body, flags=re.M)
        return f"\\needspace{{{table_lines(body) + 3}\\baselineskip}}\n" + body
    src = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", keep_together, src, flags=re.S)

    # A heading followed by a kept-together table, or by further headings, must
    # reserve room for the whole group; otherwise the table's \needspace is met
    # after the heading and leaves it alone at the bottom of the page. Pandoc
    # may wrap the table in a "{\def\LTcaptype{none}" group, which is kept.
    heading = r"\\(?:sub){0,2}section\{[^{}]*?\}\\label\{[^}]*\}\n\s*"   # titles may wrap
    wrapper = r"(?:\{\\def\\LTcaptype\{none\}[^\n]*\n)?"
    def hoist(m):
        heads = m.group("heads")
        n = int(m.group("n")) + 3 * len(re.findall(heading, heads))
        return f"\\needspace{{{n}\\baselineskip}}\n" + heads + m.group("wrap")
    src = re.sub(rf"(?P<heads>(?:{heading})+)(?P<wrap>{wrapper})\\needspace\{{(?P<n>\d+)\\baselineskip\}}\n",
                 hoist, src)
    # Two or more headings in a row (a section opening with a subsection) stay
    # together with a few lines of what follows.
    def group(m):
        k = len(re.findall(heading, m.group(0)))
        return f"\\needspace{{{3 * k + 4}\\baselineskip}}\n" + m.group(0)
    src = re.sub(rf"(?<!baselineskip\}}\n)(?:{heading}){{2,}}", group, src)
    tex.write_text(src, encoding="utf-8")

    if tex_only:
        print(f"LaTeX source: {out / 'offline-rl-book.tex'}")
        return
    need("tectonic")
    subprocess.run(["tectonic", "--keep-logs", "offline-rl-book.tex"], cwd=out, check=True)
    print(f"PDF: {out / 'offline-rl-book.pdf'}")


def about_text(lang="en"):
    """The abstract on the PDF's title page, as plain text: one source for both formats."""
    tex = (HERE / LANG[lang]["title_tex"]).read_text(encoding="utf-8")
    body = re.search(r"\\begin\{quote\}(.*?)\\end\{quote\}", tex, flags=re.S).group(1)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if p.strip()]
    return "\n\n".join(paras).replace("~--", "\u00a0–").replace("--", "–")


def build_epub(out, lang="en"):
    """EPUB 3 of one edition: same chapters, MathML for formulas, SVG figures."""
    L = LANG[lang]
    need("pandoc"); need("rsvg-convert")
    if out.exists():
        shutil.rmtree(out)
    figdir = out / "figures"
    figdir.mkdir(parents=True)
    # Cover in the style of the README banner; the PNG is also what Leanpub asks for.
    subprocess.run(["rsvg-convert", "-w", "1600", "-h", "2400", "-o", str(out / "cover.png"),
                    str(COVERS / L["cover"])], check=True)
    parts = [preprocess((ROOT / rel).read_text(encoding="utf-8"), figdir, chapter_id(rel), epub=True, lang=lang)
             for rel in chapters(lang)]
    # Same text as "About this book" in the Leanpub PDF, right after the title page.
    parts.insert(0, f"# {L['about']} {{.unnumbered .unlisted}}\n\n" + about_text(lang))
    (out / "book.md").write_text("\n\n".join(parts), encoding="utf-8")
    version, commit = edition()
    date = datetime.date.today().isoformat()
    # Title page: where the free, citable version lives, as on the PDF's title page.
    rights = "\\\n  ".join(line.format(version=version, site=L["site"], doi=DOI, commit=commit)
                            for line in L["rights"])
    (out / "meta.yaml").write_text(f"""---
title: "{L['title']}"
subtitle: "{L['subtitle']}"
author: "Artem Zvyagintsev"
lang: {lang}
toc-title: "{L['toc']}"
date: "{date}"
description: "Built from commit {commit}. Free online at {L['site']}"
rights: |
  {rights}
---
""", encoding="utf-8")
    subprocess.run([
        "pandoc", "book.md", "-o", "offline-rl-book.epub", "--to", "epub3", "--metadata-file=meta.yaml",
        "--from", "markdown+tex_math_dollars+pipe_tables+lists_without_preceding_blankline-implicit_figures",
        "--math-method=mathml", "--toc", "--toc-depth=2", "--number-sections", "--split-level=1",
        "--syntax-highlighting=tango", "--css", str(HERE / "epub.css"), "--epub-cover-image", "cover.png",
    ], cwd=out, check=True)
    print(f"EPUB: {out / 'offline-rl-book.epub'}")


if __name__ == "__main__":
    main()
