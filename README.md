# Offline RL: From Theory to Industrial Practice

A practical book on offline reinforcement learning, published on GitHub Pages.

**English:** https://your-username.github.io/offline-rl-book/  
**Russian:**  https://your-username.github.io/offline-rl-book/ru/

## Contents

| Chapter | EN | RU | Status |
|---|---|---|---|
| 1. Behavioral Cloning | `en/chapter1.md` | `ru/chapter1.md` | ✅ Ready |
| 2. The Offline RL Problem | `en/chapter2.md` | `ru/chapter2.md` | ✅ Ready |
| 3. Conservative Q-Learning (CQL) | — | — | 🔜 |
| 4. Implicit Q-Learning (IQL) | — | — | 🔜 |
| 5. Model-Based Offline RL | — | — | 🔜 |
| 6. Physics-Informed Offline RL | — | — | 🔜 |
| 7. Industrial Applications | — | — | 🔜 |

## Deploying to GitHub Pages

### 1. Create repo and push

```bash
git init
git add .
git commit -m "Initial: chapters 1-2 EN+RU"
git remote add origin https://github.com/YOUR_USERNAME/offline-rl-book.git
git push -u origin main
```

### 2. Enable GitHub Pages

Go to **Settings → Pages → Source → Deploy from branch → main / (root)**.

That's it. GitHub automatically builds the Jekyll site. Available at:
`https://YOUR_USERNAME.github.io/offline-rl-book/`

### 3. Update `_config.yml`

Edit these two lines in `_config.yml`:
```yaml
github_username: YOUR_USERNAME
github_repo:     offline-rl-book
```

## Repository Structure

```
├── _config.yml              # Jekyll config (MathJax, theme)
├── _layouts/
│   └── default.html         # Custom layout with math + code highlighting
├── index.md                 # English home page (TOC)
├── en/
│   ├── chapter1.md
│   └── chapter2.md
├── ru/
│   ├── index.md             # Russian home page
│   ├── chapter1.md
│   └── chapter2.md
└── code/
    ├── behavioral_cloning.py
    └── extrapolation_error.py
```

## Writing New Chapters

Copy the front matter from any existing chapter, update the fields:

```yaml
---
layout: default
title: "Chapter 3: Conservative Q-Learning"
lang: en
ru_url: /ru/chapter3/
prev_chapter:
  url: /en/chapter2/
  title: "The Offline RL Problem"
next_chapter:
  url: /en/chapter4/
  title: "Implicit Q-Learning (IQL)"
---
```

Math: use `$...$` for inline, `$$...$$` for display — rendered by MathJax.  
Code: standard fenced code blocks with `python` language tag.  
File references: `> 📄 Full code: [filename.py](../../code/filename.py)`
