# The Winner's Curse in LLM Pipeline Selection

Measuring how much of a reported prompt-optimization gain is selection bias rather than
improvement, why the textbook corrections misfire on correlated candidate pools, and how to
split a fixed evaluation budget between more candidates and more dev items.

Companion code for the preprint. Two halves, deliberately separated:

| Half | Needs a GPU | What it does |
| --- | --- | --- |
| `scripts/generate_candidates.py`, `scripts/build_matrix.py` | yes | builds the per-item correctness matrix, once |
| `src/wcurse/`, `scripts/run_analysis.py`, `scripts/make_figures.py` | no | every result in the paper, by resampling that matrix |

## The instrument

Everything rests on one artifact: a per-item correctness matrix `C` of shape `K x M`, where
`C[k, i] = 1` iff candidate prompt `k` answers evaluation item `i` correctly. `K = 100`
candidates, `M = 600` items, split into a 300-item dev pool that selection is allowed to see
and a disjoint 300-item truth pool that it never is.

Once that matrix exists, every experiment is a subsample of it and costs milliseconds:
random search at any `N` is a row subsample, any dev-set size `m` is a column subsample, and
the true value of whatever candidate wins is a lookup in the truth pool.

## Install

```bash
pip install -e ".[analysis,dev]"     # analysis half; no torch, no vLLM
pytest -q                            # 97 tests, about 20 seconds
```

## Reproduce the analysis

```bash
./run_all.sh                                     # tests, analysis, figures, tables
python scripts/run_analysis.py --synthetic        # analysis on a synthetic matrix
python scripts/run_analysis.py --matrices data/matrices/*.npz
python scripts/make_figures.py
```

`run_analysis.py` refuses to write figures if any sanity check fails, because a violated
check means a bug rather than a finding.

## Build a matrix (Kaggle T4)

See [`docs/KAGGLE.md`](docs/KAGGLE.md). Pilot before committing the quota:

```bash
python scripts/generate_candidates.py --task sst2 --model Qwen/Qwen2.5-1.5B-Instruct
python scripts/build_matrix.py --task sst2 --limit-candidates 10 --pilot   # extrapolate
python scripts/build_matrix.py --task sst2                                 # resumable
python scripts/build_matrix.py --task sst2 --finalize
```

Scoring is one constrained token per candidate-item pair: `allowed_token_ids` restricts
sampling to the first token of each verbalizer, so the model cannot dodge the question and
greedy decoding makes the prediction an exact argmax over the classes.

## Using the correction on your own results

If you are reporting a best-of-N result and have the per-item correctness of every candidate
on your dev set, this is the whole interface:

```python
import numpy as np
from wcurse import union_bound_neff, spectral_neff

dev = np.load("my_candidates_by_items.npy")      # (N, m) binary
print(spectral_neff(dev)["n_eff"])               # how many candidates you effectively had
est = union_bound_neff(dev, alpha=0.10)
print(f"selected candidate {est.selected}: {est.point:.3f} [{est.lo:.3f}, {est.hi:.3f}]")
```

`union_bound_neff` is the recommended method. It estimates the effective candidate count
from the correlation structure of your own dev matrix, subtracts the extreme-value bias term
implied by that count, and returns a Bonferroni interval. No extra data, no extra model
calls.

The other five methods in `wcurse.corrections` exist so the comparison in the paper is
honest, not because you should use them. In particular `union_bound_k` corrects over the
nominal candidate count and systematically overshoots on correlated pools, which is one of
the results.

## What the package contains

| Module | Purpose |
| --- | --- |
| `matrix.py` | the `CorrectnessMatrix` container, its splits, and its self-validation |
| `resample.py` | selection simulation over the `(N, m)` grid, plus the bias decomposition |
| `neff.py` | both effective-candidate-count estimators |
| `corrections.py` | the six reporting methods |
| `evaluate.py` | grading methods against truth on bias, MSE, and coverage |
| `allocation.py` | the budget-split curve and its interior optimum |
| `stats.py` | Wilcoxon signed-rank, Holm-Bonferroni, bootstrap intervals, effect sizes |
| `sanity.py` | the bug detectors, run before any figure is drawn |
| `synth.py` | synthetic pools with a tunable correlation regime |
| `tasks.py` | dataset plumbing and verbalizers for reproduction |

## Two subtleties worth knowing before reading the numbers

**The bias splits in two.** `reported - true` is the sum of an item-level term, from
selecting on only `m` of the dev-pool items, and a pool-level term, from the dev pool itself
being a finite sample of 300 items. Only the first is visible to any correction computed
inside the dev set, which is why even a perfect split does not drive the measured bias to
zero. `SelectionDraws.item_bias` and `.pool_bias` report the two separately.

**The split has a constant offset.** The dev and truth pools are fixed and published, so one
is very likely a shade easier than the other. That offset shifts every measured bias without
having anything to do with selection, and at `N = 1` it is the entire measured bias.
`split_offset()` reports it and `bias_adj` in the results subtracts it.

## Status

Day-one scaffolding. The analysis half is complete and validated end to end on synthetic
matrices with known ground truth; no real correctness matrix has been built yet. Two things
block the real numbers:

1. The novelty check in section 11 of the PRD, which is the go/pivot gate. See
   [`docs/NOVELTY_CHECK.md`](docs/NOVELTY_CHECK.md).
2. GPU time on Kaggle for the matrix build.

Table 2, which places measured bias next to published prompt-optimizer gains, needs those
published numbers entered by hand with citations in `data/published_gains.csv`. Nothing is
guessed there.

## License

MIT.
