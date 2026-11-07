# Paper: uncertainty-based termination in model-based offline RL

A short research paper built from the case study in Chapter 10 of the book.

- `main.tex`, `refs.bib` — the paper.
- `make_artifacts.py` — builds every table (`tables/*.tex`) and the figure (`figures/rollout_error.pdf`) from the saved results in `code/results/`; nothing is typed in by hand.
- `build.sh` — runs `make_artifacts.py`, compiles the PDF with tectonic and packs `arxiv.tar.gz` (source, `.bbl`, tables and figure) for upload.

```bash
sh paper/build.sh        # needs the repo's Python environment and tectonic
```

To regenerate the results themselves (about an hour per ten-seed run on a laptop CPU):

```bash
python code/chapter10.py                    # benchmark, five logged variables
python code/chapter10.py --lagged-flow      # benchmark, Markov state
python code/chapter10.py --oracle-ablation  # 2x2: learned vs true dynamics, HALT on/off
python code/chapter10.py --halt-study --lagged-flow
python code/chapter10.py --halt-study
python code/chapter10.py --lagged-flow --seeds 10 11 12 13 14 15 16 17 18 19 --out code/results/ch10_benchmark_lagged_fresh.json
python code/chapter10.py --halt-study --lagged-flow --seeds 10 11 12 13 14 15 16 17 18 19 --out code/results/ch10_halt_lagged_fresh.json
```
