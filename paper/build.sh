#!/bin/sh
# Build the paper from the saved results and package the source for arXiv.
# Usage (from the repo root):  sh paper/build.sh
set -e
cd "$(dirname "$0")"
python3 make_artifacts.py
tectonic --keep-intermediates main.tex          # keeps main.bbl, which arXiv needs
rm -f arxiv.tar.gz
tar -czf arxiv.tar.gz main.tex main.bbl tables figures/rollout_error.pdf
echo "PDF: paper/main.pdf   arXiv source: paper/arxiv.tar.gz"
