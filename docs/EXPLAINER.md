# Explainer: the selection-bias harness

What this change adds, why each piece is shaped the way it is, and how to convince yourself it
works. Written to be readable by someone who has never seen the project; the deep background
can be skipped if you already know why best-of-N reporting is biased.

---

## Background

### For someone new to the problem

Suppose you want a language model to classify movie reviews. You write an instruction, measure
its accuracy on a hundred labelled examples, and get 82%. A colleague suggests trying more
instructions, so you write a hundred of them, measure all hundred on the same hundred
examples, and report the best: 91%. You have improved the prompt by nine points.

Except you probably have not. Some of that nine points is real, and some of it is the
instruction that got luckiest on those particular hundred examples. You used the same data to
*choose* and to *report*, and the maximum of a hundred noisy numbers is systematically higher
than the true value of whichever candidate produced it. Statisticians call this the winner's
curse; machine learning calls it model-selection overfitting when it remembers to call it
anything at all.

Almost every modern LLM improvement has this shape. Prompt optimizers generate candidates and
report the argmax on a dev set. DSPy optimizers search instruction-and-demonstration
combinations. Agent scaffold choices, judge rubrics, and routing configurations are all
best-of-N selections wearing different clothes. The field treats the bias as a footnote, and
nobody has published the number: how large is it at the candidate counts and dev-set sizes
that appear in real papers?

There is a textbook answer for how large it should be. If the candidates are independent and
each one's dev-set estimate is noisy with standard error `sigma / sqrt(m)`, extreme-value
theory says the expected inflation grows like

```
B  ~  (sigma / sqrt(m)) * sqrt(2 * ln(K))
```

so it grows with the number of candidates `K` and shrinks with the number of dev items `m`.

> [!IMPORTANT]
> The independence assumption is what breaks. A hundred paraphrases of one instruction succeed
> and fail on very nearly the same items. Their dev-set estimates rise and fall together, so
> the pool behaves like far fewer than `K` independent draws. Applying no correction understates
> the bias; applying a union bound over `K` overstates it. The right correction depends on an
> **effective candidate count** `N_eff`, which nobody has measured for LLM prompt pools.

### For someone joining this repository

The repository was empty before this change. The project plan lays out an eleven-day timeline whose single
compute-bound step is building a per-item correctness matrix on a free Kaggle T4, after which
every experiment is resampling on stored data.

That structure is the whole reason the timeline is credible, and it dictates the architecture:
a hard wall between the GPU half, which runs once and produces one artifact, and the analysis
half, which must run in seconds on a laptop with no model and no CUDA. This change builds the
analysis half completely, builds the GPU half ready to run, and validates the analysis half
end to end against synthetic data whose correlation structure is known by construction.

---

## Intuition

### The instrument

Everything is a function of one object: a binary matrix `C` of shape `K x M`, where
`C[k, i] = 1` if candidate prompt `k` gets evaluation item `i` right.

```
              item0  item1  item2  item3  item4  item5
candidate0      1      1      0      1      0      1
candidate1      1      1      0      1      1      0
candidate2      0      1      0      0      1      1
```

Split the columns in two: a **dev pool** that selection is allowed to see, and a disjoint
**truth pool** that it never is. Now every experiment in the paper is a subsample:

- random search over `N` candidates: pick `N` rows,
- a dev set of `m` items: pick `m` dev columns,
- the reported score: the row mean of the argmax over those columns,
- the honest score: that same row's mean over the *truth* columns.

The gap between the last two is the bias. It costs one array slice to measure, which is why
the grid over `N` and `m`, six correction methods, and thousands of resamples all fit in
eleven seconds.

### Why `N_eff` is not `N`

Take three candidates and four items. In the first pool the candidates disagree about which
items are hard:

```
        i0 i1 i2 i3        row means
c0       1  1  0  0           0.50
c1       0  1  1  0           0.50
c2       1  0  0  1           0.50
```

In the second pool they agree completely about item difficulty, and differ only in how much
they gain from it:

```
        i0 i1 i2 i3        row means
c0       1  1  0  0           0.50
c1       1  1  0  0           0.50
c2       1  0  1  0           0.50
```

Sample two of the four items. In the first pool, whichever two you draw, some candidate got
lucky, so the maximum is well above 0.5 — there are effectively three independent chances at a
lucky draw. In the second pool, `c0` and `c1` are the same candidate as far as the data can
tell; there are effectively two chances, not three. Correcting as though there were three
subtracts too much.

The spectral estimator reads that structure off the eigenvalues of the candidate correlation
matrix. If the candidates are independent the correlation matrix is nearly the identity, its
`K` eigenvalues are all near 1, and the participation ratio `(sum lambda)^2 / sum lambda^2`
comes out near `K`. If one shared factor dominates, one eigenvalue holds most of the variance
and the ratio collapses toward 1. On the synthetic pool used for validation, `K = 100` gives
`N_eff = 19`.

### Why an interior optimum exists

Fix a total budget of `N * m = 1000` candidate-item evaluations and slide along it. More
candidates raise the ceiling of the pool; more dev items make the choice of which candidate to
ship less noisy. Measured on the synthetic matrix:

| N | m | true score of the selected candidate | reported bias |
| --- | --- | --- | --- |
| 5 | 200 | 0.706 | 0.013 |
| 10 | 100 | 0.710 | 0.033 |
| **25** | **40** | **0.711** | 0.097 |
| 50 | 20 | 0.701 | 0.173 |
| 100 | 10 | 0.689 | 0.271 |

The peak is in the middle. Because `N_eff` saturates well below `N`, the ceiling gain has sharp
diminishing returns, while the noise penalty from starving `m` does not. Note also the second
column: spending the whole budget on candidates makes the number you *report* look 27 points
better while making the model you *ship* four points worse. That contrast is the paper's most
quotable result, and it is Contribution 4.

---

## Code

### The container and its splits — `src/wcurse/matrix.py`

`CorrectnessMatrix` holds the matrix, the two column index lists, and a provenance label per
candidate. It validates itself on construction, because every bug this project can have shows
up here first:

```python
def __post_init__(self) -> None:
    if not np.isin(self.C, (0, 1)).all():
        raise ValueError("C must contain only 0/1 entries")
    overlap = np.intersect1d(self.dev_idx, self.truth_idx)
    if overlap.size:
        raise ValueError(f"dev and truth pools overlap on {overlap.size} items")
```

Splits are drawn once by `make_splits(n_items, n_dev, n_truth, seed)` and stored inside the
matrix file, so they are published as index lists rather than regenerated at analysis time.

### The harness — `src/wcurse/resample.py`

`simulate_cell` is the core loop. Two details in it are not incidental.

First, tie-breaking. Dev scores over `m` binary items take only `m + 1` distinct values, so
ties are the common case at small `m`, and breaking them by lowest index would bias selection
toward whichever candidates happen to sit early in the pool:

```python
def argmax_random_tie(scores, rng):
    winners = np.flatnonzero(scores == scores.max())
    return int(winners[rng.integers(winners.size)]) if winners.size > 1 else int(winners[0])
```

Second, each grid cell derives its own seed, so adding a cell never perturbs the draws in cells
already computed — which makes the grid safe to extend without re-verifying old numbers.

The class also records the selected candidate's score on the *whole* dev pool, which buys a
decomposition that turned out to matter more than expected:

```python
@property
def item_bias(self):   # removable by a within-dev correction
    return self.reported - self.dev_pool

@property
def pool_bias(self):   # invisible to any correction computed inside the dev pool
    return self.dev_pool - self.true
```

At `N = 100`, watch which term dominates as `m` grows:

| m | total bias | item term | pool term |
| --- | --- | --- | --- |
| 20 | 0.190 | 0.179 | 0.012 |
| 50 | 0.105 | 0.086 | 0.019 |
| 100 | 0.071 | 0.037 | 0.034 |
| 200 | 0.063 | 0.002 | 0.061 |

By `m = 200` essentially all of the remaining bias comes from overfitting the 300-item dev pool
as a whole, not from the `m` items drawn out of it. No correction computed inside the dev pool
can see that term, which is exactly why the naive split does not reach zero bias, and it is a
limitation the paper should state rather than a defect to hide.

`split_offset` handles a smaller artifact of the same family. The dev and truth pools are fixed
and published, so one is a shade easier than the other; on the validation matrix the offset is
−0.6 points. That constant shifts every measured bias without having anything to do with
selection, and at `N = 1` it *is* the entire measured bias.

### Two estimators for `N_eff` — `src/wcurse/neff.py`

Estimator A is the participation ratio of the candidate correlation eigenvalues. The one real
decision is what to remove before correlating:

```python
X = X - X.mean(axis=1, keepdims=True)     # remove each candidate's own accuracy
if center == "candidate":
    return X                              # keep shared item difficulty
if center == "double":
    return X - X.mean(axis=0, keepdims=True)
```

Candidate-centering is the default and it is the substantive choice. Shared item difficulty is
not a nuisance to be removed here — it is precisely the channel through which dev-set noise
becomes correlated across candidates, and therefore the thing that drives `N_eff` below `K`.
Double-centering answers the different question of how much candidates disagree *beyond* item
difficulty, and is available for that.

Estimator B inverts the extreme-value relation. Two corrections were needed to make it well
posed, and both were found by watching the estimator misbehave.

The reference point must be the dev-pool mean, not the truth-pool score:

```python
reference = dev.mean(axis=1)
vals[s] = (dev[:, cols].mean(axis=1) - reference).max()
```

Measuring against the truth pool folds in the truth pool's own sampling noise, which does not
shrink as `m` grows, so the inferred `N_eff` drifts upward with `m` — and `N_eff` is supposed to
be a property of the pool alone. In an early run the estimator reported `N_eff = 100` at
`m = 200` on a pool whose spectral estimate was 18, purely from this effect.

Sampling `m` items without replacement from a pool of 300 also needs a finite-population
correction:

```python
se = sigma * np.sqrt(max(1.0 / m - (1.0 / n_pool if n_pool else 0.0), 1e-12))
```

Finally, the inversion uses the exact Gaussian expected maximum rather than `sqrt(2 ln n)`,
because the asymptotic form overshoots by roughly a third at the candidate counts prompt search
actually uses. Both variants are reported, and the difference is stark:

| stratum | K | `N_eff` spectral | `N_eff` moment (exact) | `N_eff` moment (asymptotic) |
| --- | --- | --- | --- | --- |
| all | 100 | 19.3 | 43.7 | 11.2 |
| seed | 10 | 7.1 | 6.8 | 2.5 |
| ape | 45 | 16.3 | 23.5 | 6.6 |
| mutation | 45 | 14.6 | 23.8 | 6.7 |

The two estimators agree on the ten hand-written seeds and disagree by roughly a factor of two
on the generated strata. That disagreement was anticipated going in: the participation ratio
measures average pairwise correlation across the whole pool, while the moment estimator only
sees the upper tail, and the most extreme candidates are less correlated with each other than
the pool average. Report both.

### Six ways to report a winner — `src/wcurse/corrections.py`

Every method receives exactly what an honest practitioner has — the `(N, m)` binary dev matrix
— and returns a point estimate, an interval, and which candidate it is talking about. On the
validation matrix at `N = 100`, `m = 100`, nominal 90% intervals:

| method | bias | MSE | coverage | width |
| --- | --- | --- | --- | --- |
| No correction | +0.072 | 0.0064 | **0.38** | 0.129 |
| Naive split | +0.014 | 0.0043 | 0.87 | 0.200 |
| Union bound over K | −0.047 | 0.0034 | 0.99 | 0.258 |
| **Union bound over `N_eff`** | **−0.013** | **0.0014** | 1.00 | 0.216 |
| Cross-fitting | −0.004 | 0.0021 | 0.96 | 0.205 |
| Bootstrap plug-in | +0.025 | 0.0025 | 0.80 | 0.127 |

Current practice puts a nominal 90% interval on the truth 38% of the time. Correcting over `N`
overshoots; correcting over `N_eff` is the only method that is both nearly unbiased and best on
MSE, and it needs no extra data:

```python
def union_bound_neff(dev, alpha=0.10, rng=None):
    n_eff = float(np.clip(spectral_neff(dev)["n_eff"], 1.0, dev.shape[0]))
    return _evt_correction(dev, alpha, rng, n_eff, {"n_used": n_eff})
```

One deliberate choice inside `_evt_correction`: the `K` variant also uses the exact Gaussian
expected maximum. Using `sqrt(2 ln K)` would have made the straw man easier to knock over and
the result less believable.

### The bug detectors — `src/wcurse/sanity.py`

Six properties must hold here, on the principle that a violation
means a bug rather than a finding. Making them actually work took three passes, and the failures
were instructive.

**Fixed tolerances do not survive contact with the grid.** At `N = 1, m = 20` the per-cell
standard error is an order of magnitude larger than at `N = 100, m = 200`, so any single
constant either misses real violations in the quiet corner or fires constantly in the noisy one.
The monotonicity checks now allow crossings within a few combined standard errors:

```python
def _tolerance(a, b, n_se, floor):
    return max(floor, n_se * float(np.hypot(_cell_se(a), _cell_se(b))))
```

**The `N = 1` check needed the split offset.** "Bias is approximately zero at `N = 1`" is false
as literally written, because of the constant dev-versus-truth difficulty gap. The check now
tests the selection-attributable bias, against a tolerance in standard errors of that cell.

**One check contradicted the paper's own thesis.** "Corrected estimates are closer to truth than
uncorrected ones in every grid cell" fails for the union bound over `K` — and demonstrating that
failure is one of the contributions. A method whose failure is a result cannot simultaneously be
a check on the code, so `union_K` sits outside the required set and its overshoot is reported as
expected behaviour:

```python
required: tuple[str, ...] = ("split", "union_neff", "cross_fit", "bootstrap")
```

The check also stands down where the uncorrected bias is immaterial, since any extreme-value
correction must overshoot when there is nothing to correct.

### The statistical protocol — `src/wcurse/stats.py`

Wilcoxon signed-rank on paired observations, Holm-Bonferroni across the whole family of tests,
Cliff's delta and median differences alongside every p-value, and percentile bootstrap intervals
on every bias estimate. Keeping these in one module means no analysis script can quietly skip a
step; the Holm correction directly answers the uncorrected-multiplicity objection a reviewer
would otherwise raise.

### The scripts

`generate_candidates.py` builds the pool: ten hand-written seeds committed verbatim, 45
APE-style instructions written by the model from labelled demonstrations, and 45 paraphrases of
the seeds, each labelled with its provenance and its parent. Demonstrations are drawn strictly
from items *outside* the 600 that get scored, because a candidate written from an item that is
later scored leaks the truth pool into the pool of candidates and the leak looks like a real
improvement.

`build_matrix.py` is the compute-bound step. Scoring is one constrained token per pair, with
`allowed_token_ids` restricting sampling to the first token of each verbalizer, so the model
cannot dodge the question and greedy decoding makes the prediction an exact argmax over the
classes. A collision check fails loudly if two class words share a first token, which would
silently reduce those two classes to a coin flip. Checkpoints are per candidate and the item
sample and split are written once and verified on resume, so a killed Kaggle session costs one
candidate and can never silently score a different item sample than the rows already on disk.

`run_analysis.py` produces every number, refuses to write figures if a sanity check fails, and
grades the corrections in the corner of the grid where the curse bites hardest — the most
candidates and the fewest dev items — rather than in whichever cell happened to be evaluated
last.

`make_figures.py` draws all six figures and writes all four tables. Table 2 is the exception it
cannot complete: it needs the gains published prompt optimizers claim, and those must be read
off the papers by hand into `data/published_gains.csv` with a citation each. The script prints a
reminder and leaves the column empty rather than inventing numbers.

---

## Verification

### What was run

- **97 tests pass** in about 20 seconds (`pytest -q`). They cover the container's validation
  rules, tie-breaking uniformity, the exactness of the bias decomposition, both `N_eff`
  estimators against pools built to be independent or single-factor, all six corrections for
  determinism and interval validity, Holm against a hand-computed example, bootstrap coverage at
  the nominal rate, and the end-to-end pipeline through both scripts.
- **The sanity checks are themselves tested.** `tests/test_sanity.py` injects the bugs each
  check exists to catch — a reversed grid axis, a leaky truth pool, a duplicated candidate, a
  correction that makes things worse — and asserts a `FAIL`. A check that cannot fail is not a
  check.
- **The full pipeline runs clean** via `./run_all.sh`: tests, analysis, six figures, four tables.
  All seven sanity checks pass, and 39 of 60 paired method comparisons survive Holm correction
  at `alpha = 0.05`.
- **Every figure was opened and read**, not merely generated without an exception.

The qualitative behaviour matches the theory on all four axes: bias rises monotonically in `N`
(from 0.09 at `N = 5` to 0.19 at `N = 100`, at `m = 20`), falls monotonically in `m`, sits at
zero within noise at `N = 1`, and `N_eff = 19` for `K = 100`.

> [!NOTE]
> Every number quoted in this document comes from a synthetic matrix, and none of it is
> evidence about real prompt pools. The synthetic generator was written to make the analysis
> testable, and its correlation structure is an assumption, not a measurement. What has been
> verified is that the machinery is correct and self-consistent; whether real LLM candidate
> pools show `N_eff` far below `K` is exactly the empirical question the Kaggle run answers.
>
> **Update, post-Kaggle-run:** it does, more starkly than this synthetic pool suggested. The
> four real correctness matrices give spectral `N_eff` of 1.5 to 3.5 for `K` between 105 and
> 111 — roughly 1-3% of the nominal pool size, well below this synthetic pool's `N_eff = 19`
> at `K = 100`. See `tables/table4_neff.md` and the updated verdict in
> [`docs/NOVELTY_CHECK.md`](NOVELTY_CHECK.md) for the real numbers and what they change about
> the paper's framing.

### Manual QA

```bash
git clone https://github.com/juniorcharlie/llm-pipeline-selection.git
cd llm-pipeline-selection
pip install -e ".[analysis,dev]"
pytest -q                      # expect 97 passed
./run_all.sh                   # expect seven PASS lines, then figures and tables
```

1. Open `figures/figure1_bias_vs_N.png`. The bias curve should rise with `N` and start at zero.
2. Open `figures/figure5_calibration.png`. "No correction" should sit far below the 90% line;
   that gap is the paper's headline calibration failure.
3. Open `figures/figure6_allocation.png`. Each budget curve should have a red circle at an
   interior point, not at either end.
4. Check `results/*/sanity.txt` — seven lines, all `PASS`.
5. Confirm the checks can fail. Corrupt a matrix and watch the pipeline stop:
   ```python
   from wcurse import make_synthetic_matrix
   cm = make_synthetic_matrix(seed=0)
   cm.C[1] = cm.C[0]          # duplicate a candidate
   cm.save("data/matrices/broken.npz")
   ```
   `python scripts/run_analysis.py --matrices data/matrices/broken.npz` should exit non-zero
   with a duplicate-candidate message. Delete the file afterwards.
6. Try the shipped correction on an arbitrary dev matrix, following the README snippet, and
   confirm `est.point < dev.mean(axis=1).max()`: the correction always moves the reported number
   down.

### Not verified

The GPU half has never been executed — there is no GPU and no Hugging Face access in the
environment where this was written. `generate_candidates.py` and `build_matrix.py` are import-
clean and their model-free paths are tested, but the vLLM calls, the tokenizer collision check
against real tokenizers, and the throughput projection are all unexercised. Treat the first
Kaggle pilot as their first real test, and read the three diagnostics `--finalize` prints before
trusting a matrix.

**Update, post-Kaggle-run:** it has been, since. Four correctness matrices are built and
committed under `data/matrices/`: AG News and Subj on Qwen2.5-1.5B-Instruct, SST-2 on both
Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct. All three `--finalize` diagnostics came back
clean on every matrix — sane accuracy range, zero unparsed generations, a dev/truth offset of a
few tenths of a point — and all seven `sanity.py` checks pass against the real data, not just
the synthetic validation pool above. TREC and Qwen2.5-7B-Instruct remain unbuilt; extending to
either is the same procedure in `docs/KAGGLE.md`, unchanged.

---

## Alternatives

### Storing correctness only, versus storing full class logprobs

The matrix stores one bit per candidate-item pair. Storing the full class-probability vector was
the alternative.

| Pros of storing logprobs | Cons |
| --- | --- |
| Enables margin analysis: how *nearly* a candidate got each item right | Multiplies matrix size by the class count, and the artifact stops being a few kilobytes |
| Would let the paper study confidence-weighted selection as a bonus result | Introduces model-specific calibration into an object that is currently model-agnostic |
| Recovers correctness at any decision threshold without rescoring | Every downstream function needs a thresholding step, and the sanity checks lose their binary crispness |

Correctness-only won because the paper's object of study is the *estimator*, not the model, and
because a binary matrix makes both the resampling and the checks trivially auditable. The
prediction rows are checkpointed rather than discarded, so a future paper can recover per-item
predictions without a rerun.

### Resampling the dev/truth split, versus one fixed published split

The split is drawn once, stored in the matrix, and never redrawn. The alternative is a fresh
300/300 split per resample seed.

| Pros of resampling the split | Cons |
| --- | --- |
| Averages away the constant dev-versus-truth difficulty offset, so `N = 1` bias is exactly zero | Violates the reproducibility requirement that item splits be published as index lists |
| Removes the pool-level bias term, making the headline bias number cleaner | Hides a real effect: practitioners *do* have one fixed dev set and *do* overfit it as a whole |
| No need for `split_offset` or the two-term decomposition | Truth would no longer be a single fixed quantity per candidate, complicating every comparison |

The fixed split won because it mirrors what practitioners actually face, and because the
pool-level term it exposes is a finding rather than a nuisance. The offset is measured and
reported instead of being averaged away, and `bias_adj` gives the offset-free number for anyone
who wants it.

---

## Suggested people to talk to

The repository had no commits before this change, so there is no authorship history to mine and
no colleague with prior context on these files. Two things follow.

First, the reviewer of this pull request is the only human who will have read the code before
the paper is submitted, which makes the review load unusually high. The three places where an
independent pair of eyes is worth the most, in order:

1. **`neff.py`, the reference point in `observed_max_deviation`.** Whether `N_eff` should be
   measured against the dev-pool mean or the truth-pool score is a modelling decision, not a
   coding one, and the paper's mechanism section rests on it.
2. **`corrections.py`, the interval constructions.** The point estimates are well grounded; the
   intervals involve judgement calls, particularly cross-fitting's fold-sized standard error and
   the bootstrap's neglect of uncertainty in its own bias estimate. Both are documented at the
   call site and both are visible in the coverage column.
3. **`sanity.py`, the exclusions.** Two checks were relaxed from their original wording. Both
   relaxations are argued in the docstrings, and both deserve a sceptical read, because a
   weakened bug detector is exactly how a wrong result gets published.

Second, for the statistical framing — especially the decision to report both `N_eff` estimators
side by side rather than picking one — the natural readers are whoever last worked on
multiplicity control in paper 2, since the Holm protocol here is deliberately the answer to the
weakness that paper left open.

---
