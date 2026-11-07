<p align="center">
  <img src="figures/readme/banner.svg" alt="Offline RL: From Theory to Industrial Practice" width="920" />
</p>

<p align="center"><em>applied learning notes on offline deep reinforcement learning — from behavioral cloning to physics-informed control and tool-using agents</em></p>

<p align="center">
  <img src="figures/ch4/cql_vs_naive.png" alt="Q-values along one action dimension: naive offline Q-learning is optimistic everywhere, far above the true value; CQL stays close to the truth on the logged actions and falls below it outside them." width="640" />
</p>
<p align="center"><em>The central problem of the book in one picture: trained on a fixed dataset, naive Q-learning overestimates actions it has never seen; CQL keeps those values below the truth. — <a href="https://corba777.github.io/offline-rl-book/en/chapter4.html">Chapter 4</a></em></p>

<p align="center">
  <b>Thirteen chapters in English and Russian, each built as idea → formalization → code → limitations.</b><br/>
  Every chapter script runs in CI; the case-study numbers in Chapters 10–12 are the output of those scripts.
</p>

<p align="center">
  <a href="https://github.com/corba777/offline-rl-book/actions/workflows/ci.yml"><img src="https://github.com/corba777/offline-rl-book/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-3fb950?labelColor=24292f" alt="code: MIT" /></a>
  <a href="LICENSE-text"><img src="https://img.shields.io/badge/text-CC%20BY%204.0-3fb950?labelColor=24292f" alt="text: CC BY 4.0" /></a>
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-3572A5?labelColor=24292f" alt="python 3.12 | 3.13" />
  <img src="https://img.shields.io/badge/PyTorch-CPU%20is%20enough-ee4c2c?labelColor=24292f" alt="PyTorch, CPU is enough" />
  <img src="https://img.shields.io/badge/languages-EN%20%7C%20RU-0969da?labelColor=24292f" alt="EN | RU" />
  <a href="https://corba777.github.io/offline-rl-book/"><img src="https://img.shields.io/badge/read-GitHub%20Pages-1f6feb?labelColor=24292f" alt="GitHub Pages" /></a>
</p>

<p align="center">
  <a href="https://corba777.github.io/offline-rl-book/">Read (EN)</a> ·
  <a href="https://corba777.github.io/offline-rl-book/ru/">Читать (RU)</a> ·
  <a href="#contents">Contents</a> ·
  <a href="#run-the-examples">Run the examples</a> ·
  <a href="#repository-layout">Layout</a> ·
  <a href="#writing-and-building">Writing and building</a> ·
  <a href="#license">License</a>
</p>

---

Created and maintained by **Artem Zvyagintsev**. These are applied learning notes, not an authoritative textbook: the chapters summarize algorithms, implementation patterns and practical caveats, and the code is educational scaffolding rather than benchmark-grade implementations.

## Contents

| # | Chapter | EN | RU | Code |
|---|---|---|---|---|
| 1 | Behavioral Cloning | [read](https://corba777.github.io/offline-rl-book/en/chapter1.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter1.html) | [`behavioral_cloning.py`](code/behavioral_cloning.py) |
| 2 | The Offline RL Problem | [read](https://corba777.github.io/offline-rl-book/en/chapter2.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter2.html) | [`extrapolation_error.py`](code/extrapolation_error.py) |
| 3 | Off-Policy Evaluation (OPE) | [read](https://corba777.github.io/offline-rl-book/en/chapter3.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter3.html) | [`fqe.py`](code/fqe.py) |
| 4 | Conservative Q-Learning (CQL) | [read](https://corba777.github.io/offline-rl-book/en/chapter4.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter4.html) | [`cql.py`](code/cql.py) |
| 5 | Implicit Q-Learning (IQL) | [read](https://corba777.github.io/offline-rl-book/en/chapter5.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter5.html) | [`iql.py`](code/iql.py) |
| 6 | Policy-Constraint and Actor-Critic (TD3+BC, AWAC) | [read](https://corba777.github.io/offline-rl-book/en/chapter6.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter6.html) | [`td3bc.py`](code/td3bc.py) |
| 7 | Decision Transformers | [read](https://corba777.github.io/offline-rl-book/en/chapter7.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter7.html) | [`decision_transformer.py`](code/decision_transformer.py) |
| 8 | Model-Based Offline RL (MOPO, MOReL) | [read](https://corba777.github.io/offline-rl-book/en/chapter8.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter8.html) | [`mopo.py`](code/mopo.py), [`morel.py`](code/morel.py) |
| 9 | Physics-Informed Offline RL | [read](https://corba777.github.io/offline-rl-book/en/chapter9.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter9.html) | [`physics_informed.py`](code/physics_informed.py) |
| 10 | Industrial Applications | [read](https://corba777.github.io/offline-rl-book/en/chapter10.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter10.html) | [`chapter10.py`](code/chapter10.py) |
| 11 | Explainability in Offline RL | [read](https://corba777.github.io/offline-rl-book/en/chapter11.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter11.html) | [`chapter11.py`](code/chapter11.py) |
| 12 | Offline RL for Tool-Using LLM Agents | [read](https://corba777.github.io/offline-rl-book/en/chapter12.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter12.html) | [`agentic_offline_rl_toy.py`](code/agentic_offline_rl_toy.py) |
| 13 | Conclusion and Future Directions | [read](https://corba777.github.io/offline-rl-book/en/chapter13.html) | [читать](https://corba777.github.io/offline-rl-book/ru/chapter13.html) | — |
| App. | Algorithm Selection Guide | [read](https://corba777.github.io/offline-rl-book/en/appendix.html) | [читать](https://corba777.github.io/offline-rl-book/ru/appendix.html) | — |

Off-policy evaluation has its own chapter (3); value-based methods (CQL, IQL), policy constraints and actor-critic methods (TD3+BC, AWAC), Decision Transformers and model-based methods (MOPO, MOReL) each have one too. Chapter 2 maps the whole landscape with references. Chapters 9–12 move from algorithms to practice: physics priors, an industrial case study with multi-seed results, explainability, and tool-using LLM agents.

## Run the examples

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CPU is enough. Each script reproduces the experiment of its chapter; most accept `--quick` for tiny settings. Three that finish in seconds:

```bash
python code/chapter11_causal_toy.py     # numpy only
python code/agentic_offline_rl_toy.py   # stdlib only
python code/extrapolation_error.py      # PyTorch: out-of-data overestimation (Chapter 2)
```

The full Chapter 10 benchmark (five seeds, about 30 minutes on a laptop) writes [`code/results/ch10_benchmark.json`](code/results/ch10_benchmark.json), which the chapter quotes:

```bash
python code/chapter10.py --seeds 0 1 2 3 4
```

To check that every script still runs (about two minutes on a laptop):

```bash
python scripts/run_examples.py                       # all chapter scripts, quick settings, in a temp copy
python -m unittest discover -s code -p 'test_*.py'   # numerical checks for CQL, TD3+BC and the Chapter 12 toy
```

GitHub Actions runs both on every push and also checks that the chapter HTML was rebuilt after the Markdown changed.

## Repository layout

```
├── index.md, index.html, ru.html   # English and Russian tables of contents
├── en/, ru/                   # chapter1.md … chapter13.md, appendix.md and the built .html
├── figures/                   # figures per chapter (ch1/, ch2/, …), README banner in readme/
├── code/
│   ├── behavioral_cloning.py, extrapolation_error.py, fqe.py
│   ├── cql.py, iql.py, td3bc.py, decision_transformer.py
│   ├── mopo.py, morel.py, physics_informed.py
│   ├── chapter10.py, chapter11.py, chapter11_toy_figures.py, chapter11_causal_toy.py
│   ├── agentic_offline_rl_toy.py
│   ├── chapter1_5_8_figures.py, chapter2_4_figures.py   # figure generators
│   ├── test_*.py                                         # unit tests
│   └── results/                                          # saved benchmark results
├── scripts/
│   ├── md2html.py             # Markdown → chapter HTML
│   └── run_examples.py        # smoke-runs every chapter script
├── .github/workflows/ci.yml   # tests, quick runs, stale-HTML check
└── requirements.txt           # dependencies and the chapter → script map
```

## Writing and building

Chapters are Markdown with front matter; copy it from a neighbouring chapter and adjust the numbers:

```yaml
---
layout: default
title: "Chapter 11: Explainability in Offline RL"
lang: en
ru_url: /ru/chapter11/
prev_chapter:
  url: /en/chapter10/
  title: "Industrial Applications"
next_chapter:
  url: /en/chapter12/
  title: "Offline RL for Tool-Using LLM Agents"
permalink: "/offline-rl-book/en/chapter11/"
---
```

- Math: `$...$` inline, `$$...$$` display, rendered by MathJax.
- Code: fenced blocks with the `python` tag.
- Links to code: `> 📄 Full code: [filename.py](https://github.com/corba777/offline-rl-book/blob/main/code/filename.py)`; use the full GitHub URL so the link also works on github.io.

After editing, rebuild the HTML of every chapter, so that all pages share the same markup:

```bash
python scripts/md2html.py
```

The HTML was built with markdown 3.11.3, Pygments 2.21.0 and PyYAML 6.0.3; other versions change the markup, not the content, and CI checks with exactly these.

## License

- Code in `code/` and `scripts/`: [MIT](LICENSE).
- Text and figures (`en/`, `ru/`, `figures/`, the tables of contents): [CC BY 4.0](LICENSE-text) — free to share and adapt with attribution.
