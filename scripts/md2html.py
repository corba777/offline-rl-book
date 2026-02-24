#!/usr/bin/env python3
"""
Convert chapter .md files to .html in the same style as existing en/chapter1.html.
Reads YAML front matter from each .md, converts body with markdown, wraps in template.
Requires: pip install markdown pyyaml
Usage: python scripts/md2html.py
"""
import re
import os
from pathlib import Path

import yaml
import markdown

# Base path for links (as in existing HTML)
BASE = "/offline-rl-book"

# Same style as en/chapter1.html (minified)
HEAD = '''<!DOCTYPE html>
<html lang="__LANG__">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>__TITLE__</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script>MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:false},options:{skipHtmlTags:['script','noscript','style','textarea','pre']}}</script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <script>document.addEventListener('DOMContentLoaded',()=>hljs.highlightAll())</script>
  <style>
    :root{--text:#1a1a2e;--text-light:#555570;--bg:#fafaf8;--bg-code:#f4f4f0;--accent:#2563eb;--border:#e2e2dc;--serif:'Source Serif 4',Georgia,serif;--sans:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;--max-width:720px}
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    html{font-size:18px;scroll-behavior:smooth}
    body{font-family:var(--serif);background:var(--bg);color:var(--text);line-height:1.75;padding:0 1rem}
    .site-header{border-bottom:1px solid var(--border);padding:1.2rem 0;margin-bottom:3rem}
    .site-header-inner{max-width:var(--max-width);margin:0 auto;display:flex;justify-content:space-between;align-items:center}
    .site-title{font-family:var(--sans);font-weight:600;font-size:0.95rem;color:var(--text);text-decoration:none}
    .lang-switch{font-family:var(--sans);font-size:0.82rem;color:var(--text-light);text-decoration:none;border:1px solid var(--border);padding:0.25rem 0.65rem;border-radius:4px}
    .lang-switch:hover{border-color:var(--accent);color:var(--accent)}
    main{max-width:var(--max-width);margin:0 auto 6rem}
    h1{font-size:2rem;font-weight:600;line-height:1.25;letter-spacing:-0.02em;margin-bottom:1rem}
    h2{font-family:var(--sans);font-size:1.15rem;font-weight:600;margin:3rem 0 0.75rem;padding-bottom:0.4rem;border-bottom:1px solid var(--border)}
    h3{font-family:var(--sans);font-size:1rem;font-weight:600;margin:2rem 0 0.5rem}
    p{margin-bottom:1.2rem}
    strong{font-weight:600}
    em{font-style:italic}
    a{color:var(--accent);text-underline-offset:2px}
    blockquote{border-left:3px solid var(--accent);margin:2rem 0;padding:0.75rem 0 0.75rem 1.5rem;color:var(--text-light);font-style:italic}
    blockquote p{margin:0}
    code{font-family:var(--mono);font-size:0.84em;background:var(--bg-code);padding:0.15em 0.35em;border-radius:3px;border:1px solid var(--border)}
    pre{background:var(--bg-code)!important;border:1px solid var(--border);border-radius:6px;padding:1.25rem 1.5rem;overflow-x:auto;margin:1.5rem 0 2rem;font-size:0.82rem;line-height:1.6}
    pre code{background:none;border:none;padding:0;font-size:inherit}
    table{width:100%;border-collapse:collapse;font-family:var(--sans);font-size:0.88rem;margin:1.5rem 0 2rem}
    th{background:var(--bg-code);font-weight:600;text-align:left;padding:0.6rem 1rem;border-bottom:2px solid var(--border)}
    td{padding:0.55rem 1rem;border-bottom:1px solid var(--border)}
    hr{border:none;border-top:1px solid var(--border);margin:2rem 0}
    .chapter-nav{display:flex;justify-content:space-between;margin-top:4rem;padding-top:2rem;border-top:1px solid var(--border);font-family:var(--sans);font-size:0.85rem}
    .chapter-nav a{color:var(--text-light);text-decoration:none;display:flex;flex-direction:column;gap:0.2rem;max-width:45%}
    .chapter-nav a:hover{color:var(--accent)}
    .nav-label{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em}
    .nav-title{font-weight:500}
    .next{text-align:right;margin-left:auto}
    .site-footer{border-top:1px solid var(--border);padding:2rem 0;text-align:center;font-family:var(--sans);font-size:0.8rem;color:var(--text-light)}
    .site-footer a{color:var(--accent)}
    @media(max-width:600px){html{font-size:16px}h1{font-size:1.6rem}pre{font-size:0.78rem;padding:1rem}}
  </style>
</head>
<body>
<header class="site-header">
  <div class="site-header-inner">
    <a class="site-title" href="__BASE__/">Offline RL</a>
    <a class="lang-switch" href="__LANG_SWITCH_URL__">__LANG_SWITCH_LABEL__</a>
  </div>
</header>
<main>
'''

TAIL = '''</main>
<nav class="chapter-nav">
__NAV__
</nav>
<footer class="site-footer">
  <div style="max-width:720px;margin:0 auto">
    Offline RL: From Theory to Industrial Practice · <a href="https://github.com/corba777/offline-rl-book">GitHub</a>
  </div>
</footer>
</body>
</html>
'''


def url_to_html(u):
    """Turn /en/chapter1/ into /offline-rl-book/en/chapter1.html"""
    if not u:
        return None
    path = u.strip("/")
    return f"{BASE}/{path}.html" if path else f"{BASE}/"


def parse_front_matter(text):
    """Return (fm_dict, body). Front matter is between first --- and second ---."""
    if not text.strip().startswith("---"):
        return {}, text
    parts = text.split("\n", 1)
    rest = parts[1] if len(parts) > 1 else ""
    match = re.match(r"---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text
    fm_str, body = match.group(1), text[match.end() :]
    try:
        fm = yaml.safe_load(fm_str) or {}
    except Exception:
        fm = {}
    return fm, body


def protect_math(body):
    """Replace $$...$$ with placeholder; return (new_body, list of math blocks)."""
    blocks = []
    def repl(m):
        blocks.append(m.group(0))
        return f"\n@@MATH{len(blocks)-1}@@\n"
    new_body = re.sub(r"\$\$[\s\S]*?\$\$", repl, body)
    return new_body, blocks


def restore_math(html, blocks):
    """Replace @@MATHi@@ with <div class="math-display">...</div>"""
    for i, math in enumerate(blocks):
        html = html.replace(f"@@MATH{i}@@", f'<div class="math-display">{math}</div>')
    return html


def md_to_html(body):
    """Convert markdown to HTML; preserve display math."""
    body, math_blocks = protect_math(body)
    md = markdown.Markdown(extensions=["extra", "codehilite"])
    html = md.convert(body)
    html = restore_math(html, math_blocks)
    # Ensure code blocks have language for highlight.js (optional)
    return html


def build_nav(prev_chapter, next_chapter):
    """Build chapter nav HTML."""
    prev_a = ""
    next_a = ""
    if prev_chapter and prev_chapter.get("url") and prev_chapter.get("title"):
        u = url_to_html(prev_chapter["url"])
        prev_a = f'<a href="{u}" class="prev"><span class="nav-label">← Previous</span><span class="nav-title">{prev_chapter["title"]}</span></a>'
    if next_chapter and next_chapter.get("url") and next_chapter.get("title"):
        u = url_to_html(next_chapter["url"])
        next_a = f'<a href="{u}" class="next"><span class="nav-label">Next →</span><span class="nav-title">{next_chapter["title"]}</span></a>'
    if not prev_a and not next_a:
        return "<span></span>\n    <span></span>"
    if not prev_a:
        prev_a = "<span></span>"
    return f"{prev_a}\n    {next_a}"


def convert_file(md_path, out_path):
    """Convert one .md file to .html."""
    text = md_path.read_text(encoding="utf-8")
    fm, body = parse_front_matter(text)
    title = fm.get("title", "Chapter")
    lang = fm.get("lang", "en")
    if lang == "ru":
        lang_switch_url = url_to_html(fm.get("en_url", "/en/chapter1/"))
        lang_switch_label = "English"
    else:
        lang_switch_url = url_to_html(fm.get("ru_url", "/ru/chapter1/"))
        lang_switch_label = "Русский"
    content = md_to_html(body)
    nav = build_nav(fm.get("prev_chapter"), fm.get("next_chapter"))
    html = (
        HEAD.replace("__LANG__", lang)
        .replace("__TITLE__", title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        .replace("__BASE__", BASE)
        .replace("__LANG_SWITCH_URL__", (lang_switch_url or f"{BASE}/ru/"))
        .replace("__LANG_SWITCH_LABEL__", lang_switch_label)
        + content
        + TAIL.replace("__NAV__", nav)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  {md_path} -> {out_path}")


def main():
    root = Path(__file__).resolve().parent.parent
    for folder in ["en", "ru"]:
        md_dir = root / folder
        if not md_dir.is_dir():
            continue
        print(f"{folder}/")
        for md_path in sorted(md_dir.glob("chapter*.md")):
            out_path = md_dir / md_path.name.replace(".md", ".html")
            convert_file(md_path, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
