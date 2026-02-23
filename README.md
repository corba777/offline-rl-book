# Offline RL: From Theory to Industrial Practice

A practical book on offline reinforcement learning.

**English:** https://corba777.github.io/offline-rl-book/  
**Russian:**  https://corba777.github.io/offline-rl-book/ru/

## Contents

| Chapter | EN | RU | Status |
|---|---|---|---|
| 1. Behavioral Cloning | `en/chapter1.md` | `ru/chapter1.md` | ✅ Ready |
| 2. The Offline RL Problem | `en/chapter2.md` | `ru/chapter2.md` | ✅ Ready |
| 3. Conservative Q-Learning (CQL) | `en/chapter3.md`| `ru/chapter3.md` | ✅ Ready |
| 4. Implicit Q-Learning (IQL) | `en/chapter4.md` | `ru/chapter4.md` | ✅ Ready |
| 5. Model-Based Offline RL | — | — | 🔜 |
| 6. Physics-Informed Offline RL | — | — | 🔜 |
| 7. Industrial Applications | — | — | 🔜 |

## Repository Structure

```
├── _config.yml              # Jekyll config (MathJax, theme)
├── _layouts/
│   └── default.html         # Custom layout with math + code highlighting
├── index.md                 # English home page (TOC)
├── en/
│   ├── chapter1.md
│   ├── chapter2.md
│   ├── chapter3.md
│   └── chapter4.md
├── ru/
│   ├── index.md             # Russian home page
│   ├── chapter1.md
│   ├── chapter2.md
│   ├── chapter3.md
│   └── chapter4.md
└── code/
    ├── behavioral_cloning.py
    ├── extrapolation_error.py
    ├── cql.py
    └── iql.py
```

## Writing New Chapters

Copy the front matter from any existing chapter:

```yaml
---
layout: default
title: "Chapter 5: Model-Based Offline RL"
lang: en
ru_url: /ru/chapter5/
permalink: "/offline-rl-book/en/chapter5/"
prev_chapter:
  url: /en/chapter4/
  title: "Implicit Q-Learning (IQL)"
next_chapter:
  url: /en/chapter6/
  title: "Physics-Informed Offline RL"
---
```

Math: `$...$` inline, `$$...$$` display — rendered by MathJax.  
Code: fenced blocks with `python` tag.  
File refs: `> 📄 Full code: [filename.py](../../code/filename.py)`
