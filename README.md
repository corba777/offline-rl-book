# Offline RL: From Theory to Industrial Practice

A practical book on offline reinforcement learning.

**English:** https://corba777.github.io/offline-rl-book/  
**Russian:**  https://corba777.github.io/offline-rl-book/ru/

## Contents

| Chapter | EN | RU | Status |
|---|---|---|---|
| 1. Behavioral Cloning | `en/chapter1.md` | `ru/chapter1.md` | ✅ Ready |
| 2. The Offline RL Problem | `en/chapter2.md` | `ru/chapter2.md` | ✅ Ready |
| 3. Conservative Q-Learning (CQL) | `en/chapter3.md` | `ru/chapter3.md` | ✅ Ready |
| 4. Implicit Q-Learning (IQL) | `en/chapter4.md` | `ru/chapter4.md` | ✅ Ready |
| 5. Model-Based Offline RL (MOPO, MOReL) | `en/chapter5.md` | `ru/chapter5.md` | ✅ Ready |
| 6. Physics-Informed Offline RL | `en/chapter6.md` | `ru/chapter6.md` | ✅ Ready |
| 7. Industrial Applications | `en/chapter7.md` | `ru/chapter7.md` | ✅ Ready |
| 8. Explainability in Offline RL | `en/chapter8.md` | `ru/chapter8.md` | ✅ Ready |
| 9. Conclusion and Future Directions | `en/chapter9.md` | `ru/chapter9.md` | ✅ Ready |

## Repository Structure

```
├── index.html               # English TOC
├── ru.html                  # Russian TOC
├── en/
│   ├── chapter1.md … chapter9.md
│   └── chapter1.html … chapter9.html
├── ru/
│   ├── chapter1.md … chapter9.md
│   └── chapter1.html … chapter9.html
└── code/
    ├── behavioral_cloning.py
    ├── extrapolation_error.py
    ├── cql.py
    ├── iql.py
    ├── mopo.py
    ├── morel.py
    ├── physics_informed.py
    ├── chapter7.py          # Industrial case study (coating process)
    └── chapter8.py          # SHAP explainability
```

## Writing New Chapters

Copy the front matter from any existing chapter:

```yaml
---
layout: default
title: "Chapter 8: Explainability in Offline RL"
lang: en
ru_url: /offline-rl-book/ru/chapter8/
permalink: "/offline-rl-book/en/chapter8/"
prev_chapter:
  url: /offline-rl-book/en/chapter7/
  title: "Industrial Applications"
next_chapter:
  url: /offline-rl-book/en/chapter9/
  title: "Conclusion and Future Directions"
---
```

Math: `$...$` inline, `$$...$$` display — rendered by MathJax.  
Code: fenced blocks with `python` tag.  
File refs: `> 📄 Full code: [filename.py](https://github.com/corba777/offline-rl-book/blob/main/code/filename.py)` (use full GitHub URL so links work on github.io)
