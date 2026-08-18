#!/usr/bin/env bash
# Regenerate every figure and table from stored matrices, in one script.
#
# Reproducibility checklist item: "Every figure and table regenerable from stored matrices by
# one script." This is that script. It runs the tests first, because a figure drawn from
# broken code is worse than no figure.
set -euo pipefail

cd "$(dirname "$0")"

echo "== tests"
python -m pytest -q

echo
echo "== analysis"
shopt -s nullglob
matrices=(data/matrices/*.npz)
if [ ${#matrices[@]} -eq 0 ]; then
  echo "no matrices in data/matrices; falling back to a synthetic matrix"
  echo "(the analysis is identical; only the data is simulated)"
  python scripts/run_analysis.py --synthetic
else
  python scripts/run_analysis.py --matrices "${matrices[@]}"
fi

echo
echo "== figures and tables"
python scripts/make_figures.py

echo
echo "done: figures/ tables/ results/"
