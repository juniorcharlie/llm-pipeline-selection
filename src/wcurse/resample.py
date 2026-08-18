"""Simulating best-of-N selection by resampling a stored correctness matrix.

Every experiment in the paper is a call into this module: pick ``N`` rows, pick
``m`` dev columns, take the argmax, then look up what that candidate was really
worth on the disjoint truth pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matrix import CorrectnessMatrix


@dataclass(frozen=True)
class SelectionDraws:
    """One (N, m) grid cell, resampled ``n_seeds`` times.

    Attributes:
        reported: dev score of the selected candidate (the number papers print).
        true: truth-pool score of the selected candidate.
        dev_pool: score of the selected candidate on the *entire* dev pool, which is what
            makes the two-part bias decomposition possible.
        pool_max_true: best truth-pool score available among the N sampled candidates.
        selected: row index (into the full pool) of the selected candidate.
        N: number of candidates in each draw.
        m: dev-set size in each draw.
    """

    reported: np.ndarray
    true: np.ndarray
    dev_pool: np.ndarray
    pool_max_true: np.ndarray
    selected: np.ndarray
    N: int
    m: int

    @property
    def bias(self) -> np.ndarray:
        """``B``: how inflated the reported number is, per draw."""
        return self.reported - self.true

    @property
    def regret(self) -> np.ndarray:
        """``R``: true performance lost to noisy selection, per draw."""
        return self.pool_max_true - self.true

    @property
    def item_bias(self) -> np.ndarray:
        """The part of the bias caused by selecting on only ``m`` of the dev-pool items.

        This is the component that a within-dev-set correction can hope to remove.
        """
        return self.reported - self.dev_pool

    @property
    def pool_bias(self) -> np.ndarray:
        """The part of the bias caused by the dev pool itself being a finite sample.

        Even a perfect split of the ``m`` items leaves this term: choosing the argmax over
        the dev pool overfits the 300 dev items as a whole, and the truth pool is a
        disjoint sample. No correction computed inside the dev pool can see it, which is
        why the naive split does not reach exactly zero bias.
        """
        return self.dev_pool - self.true

    def summary(self) -> dict:
        return {
            "N": self.N,
            "m": self.m,
            "n_seeds": self.reported.size,
            "mean_reported": float(self.reported.mean()),
            "mean_true": float(self.true.mean()),
            "bias": float(self.bias.mean()),
            "bias_se": float(self.bias.std(ddof=1) / np.sqrt(self.bias.size)),
            "item_bias": float(self.item_bias.mean()),
            "pool_bias": float(self.pool_bias.mean()),
            "regret": float(self.regret.mean()),
            "regret_se": float(self.regret.std(ddof=1) / np.sqrt(self.regret.size)),
        }


def argmax_random_tie(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with uniform tie-breaking.

    Dev scores over ``m`` binary items take only ``m + 1`` distinct values, so ties are
    the common case at small ``m``. Breaking them by lowest index would quietly bias
    selection toward whichever candidates happen to sit early in the pool.
    """
    winners = np.flatnonzero(scores == scores.max())
    return int(winners[rng.integers(winners.size)]) if winners.size > 1 else int(winners[0])


def simulate_cell(
    cm: CorrectnessMatrix,
    N: int,
    m: int,
    n_seeds: int,
    seed: int,
) -> SelectionDraws:
    """Resample one ``(N, m)`` cell of the grid.

    Rows are drawn without replacement from the pool of ``K`` candidates, columns
    without replacement from the dev pool. Truth is always the full truth pool, which
    the selection step never sees.
    """
    if N > cm.K:
        raise ValueError(f"N={N} exceeds pool size K={cm.K}")
    dev = cm.dev()
    if m > dev.shape[1]:
        raise ValueError(f"m={m} exceeds dev pool size {dev.shape[1]}")

    truth = cm.truth()
    dev_pool_mean = dev.mean(axis=1)
    rng = np.random.default_rng(seed)
    reported = np.empty(n_seeds)
    true = np.empty(n_seeds)
    dev_pool = np.empty(n_seeds)
    pool_max = np.empty(n_seeds)
    selected = np.empty(n_seeds, dtype=np.int32)

    for s in range(n_seeds):
        rows = rng.choice(cm.K, size=N, replace=False)
        cols = rng.choice(dev.shape[1], size=m, replace=False)
        dev_scores = dev[np.ix_(rows, cols)].mean(axis=1)
        j = argmax_random_tie(dev_scores, rng)
        k_star = rows[j]
        reported[s] = dev_scores[j]
        true[s] = truth[k_star]
        dev_pool[s] = dev_pool_mean[k_star]
        pool_max[s] = truth[rows].max()
        selected[s] = k_star

    return SelectionDraws(reported, true, dev_pool, pool_max, selected, N=N, m=m)


def simulate_grid(
    cm: CorrectnessMatrix,
    n_values: list[int],
    m_values: list[int],
    n_seeds: int,
    seed: int,
) -> dict[tuple[int, int], SelectionDraws]:
    """Resample the whole ``(N, m)`` grid.

    Each cell gets its own derived seed so that adding a cell never perturbs the draws
    in cells already computed.
    """
    out: dict[tuple[int, int], SelectionDraws] = {}
    for i, N in enumerate(n_values):
        for j, m in enumerate(m_values):
            cell_seed = seed + 1000 * (i + 1) + (j + 1)
            out[(N, m)] = simulate_cell(cm, N=N, m=m, n_seeds=n_seeds, seed=cell_seed)
    return out


def split_offset(cm: CorrectnessMatrix) -> float:
    """Mean gap between the dev pool and the truth pool, averaged over all candidates.

    The dev/truth split is fixed and published, which means one pool is very likely a
    little easier than the other. That constant offset shifts every measured bias without
    having anything to do with selection: at ``N = 1``, where no selection happens, the
    measured bias equals exactly this number. Subtract it to get the selection-attributable
    bias, and report the raw value too.
    """
    return float(cm.dev().mean(axis=1).mean() - cm.truth().mean())


def draw_dev_view(
    cm: CorrectnessMatrix, N: int, m: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the ``(N, m)`` dev-only view that a correction method is allowed to see.

    Returns the binary sub-matrix and the pool row indices it came from, so the caller
    can look up the truth of whichever candidate the method ends up reporting.
    """
    dev = cm.dev()
    rows = rng.choice(cm.K, size=N, replace=False)
    cols = rng.choice(dev.shape[1], size=m, replace=False)
    return dev[np.ix_(rows, cols)], rows
