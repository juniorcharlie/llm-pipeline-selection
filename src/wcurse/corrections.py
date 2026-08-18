"""Six ways to report a best-of-N winner, five of which try to undo the curse.

Every method sees exactly what an honest practitioner sees: the ``(N, m)`` binary dev
matrix, and nothing else. The truth pool is reserved for grading the methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

from .neff import expected_max_gaussian, spectral_neff


@dataclass(frozen=True)
class Estimate:
    """What a correction method returns.

    Attributes:
        point: the estimate of the selected candidate's true accuracy.
        lo: lower end of the interval.
        hi: upper end of the interval.
        selected: index into the rows of the dev matrix it was handed.
        extra: method-specific diagnostics (e.g. the ``N_eff`` it inferred).
    """

    point: float
    lo: float
    hi: float
    selected: int
    extra: dict | None = None


def _binom_se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1.0 - p), 1e-12) / max(n, 1)))


def _argmax_tie(scores: np.ndarray, rng: np.random.Generator) -> int:
    winners = np.flatnonzero(scores == scores.max())
    return int(winners[rng.integers(winners.size)]) if winners.size > 1 else int(winners[0])


def _item_sigma(dev: np.ndarray) -> float:
    mu = dev.mean(axis=1)
    return float(np.sqrt(max(np.mean(mu * (1.0 - mu)), 1e-12)))


def no_correction(dev: np.ndarray, alpha: float = 0.10, rng=None) -> Estimate:
    """Method 1: report the dev score of the argmax. Current published practice."""
    rng = rng or np.random.default_rng(0)
    N, m = dev.shape
    scores = dev.mean(axis=1)
    k = _argmax_tie(scores, rng)
    p = float(scores[k])
    z = norm.ppf(1 - alpha / 2)
    se = _binom_se(p, m)
    return Estimate(p, p - z * se, p + z * se, k)


def naive_split(dev: np.ndarray, alpha: float = 0.10, rng=None) -> Estimate:
    """Method 2: select on half the dev items, report on the other half.

    Unbiased by construction, and the baseline every other method has to beat. It pays
    for that unbiasedness twice: selection is noisier on ``m/2`` items, and the reported
    interval is wider.
    """
    rng = rng or np.random.default_rng(0)
    N, m = dev.shape
    perm = rng.permutation(m)
    a, b = perm[: m // 2], perm[m // 2 :]
    k = _argmax_tie(dev[:, a].mean(axis=1), rng)
    p = float(dev[k, b].mean())
    z = norm.ppf(1 - alpha / 2)
    se = _binom_se(p, b.size)
    return Estimate(p, p - z * se, p + z * se, k)


def _evt_correction(dev, alpha, rng, n_effective: float, extra: dict) -> Estimate:
    """Shared machinery for methods 3 and 4.

    The point estimate subtracts the expected maximum of ``n_effective`` Gaussian dev-noise
    draws. We use the exact Gaussian expected maximum rather than ``sqrt(2 ln n)`` so that
    the ``K`` variant is not straw-manned: at ``N = 25`` the asymptotic form overstates the
    expected maximum by roughly a third. The interval is Bonferroni over ``n_effective``,
    which is the union-bound part proper.
    """
    N, m = dev.shape
    scores = dev.mean(axis=1)
    k = _argmax_tie(scores, rng)
    p = float(scores[k])
    se = _item_sigma(dev) / np.sqrt(m)
    point = p - se * expected_max_gaussian(n_effective)
    z = norm.ppf(1 - alpha / (2 * max(n_effective, 1.0)))
    half = z * _binom_se(p, m)
    return Estimate(point, point - half, point + half, k, extra)


def union_bound_k(dev: np.ndarray, alpha: float = 0.10, rng=None) -> Estimate:
    """Method 3: correct as if all ``N`` candidates were independent."""
    rng = rng or np.random.default_rng(0)
    N = dev.shape[0]
    return _evt_correction(dev, alpha, rng, float(N), {"n_used": float(N)})


def union_bound_neff(dev: np.ndarray, alpha: float = 0.10, rng=None) -> Estimate:
    """Method 4: correct over the effective candidate count, estimated in-sample.

    ``N_eff`` comes from the spectral participation ratio of the very same dev matrix, so
    this method needs no extra data and can be run by anyone reporting a best-of-N result.
    """
    rng = rng or np.random.default_rng(0)
    n_eff = spectral_neff(dev)["n_eff"]
    n_eff = float(np.clip(n_eff, 1.0, dev.shape[0]))
    return _evt_correction(dev, alpha, rng, n_eff, {"n_used": n_eff})


def cross_fitting(
    dev: np.ndarray, alpha: float = 0.10, rng=None, n_splits: int = 50
) -> Estimate:
    """Method 5: average the held-out score of the winner over repeated random splits.

    Averaging kills the split-to-split noise that makes :func:`naive_split` erratic, but
    it cannot manufacture items, so the interval is sized by one fold rather than by the
    full dev set. The candidate actually reported is the full-dev argmax, because that is
    the candidate a practitioner would ship.
    """
    rng = rng or np.random.default_rng(0)
    N, m = dev.shape
    vals = np.empty(n_splits)
    for r in range(n_splits):
        perm = rng.permutation(m)
        a, b = perm[: m // 2], perm[m // 2 :]
        k = _argmax_tie(dev[:, a].mean(axis=1), rng)
        vals[r] = dev[k, b].mean()
    point = float(vals.mean())
    k_report = _argmax_tie(dev.mean(axis=1), rng)
    z = norm.ppf(1 - alpha / 2)
    se = _binom_se(point, m // 2)
    return Estimate(point, point - z * se, point + z * se, k_report, {"split_sd": float(vals.std())})


def bootstrap_plugin(
    dev: np.ndarray, alpha: float = 0.10, rng=None, n_boot: int = 200
) -> Estimate:
    """Method 6: estimate the selection bias by bootstrapping the dev items.

    On each resample we re-run the selection, then ask how much better the winner looked
    on the resample than it does on the full dev set. That gap is a plug-in estimate of
    the curse, and subtracting its mean gives the point estimate. The interval is the
    point estimate plus or minus the bootstrap standard error of the selected candidate's
    score; note that this deliberately ignores the uncertainty in the bias estimate
    itself, which is the standard practice and which the coverage column will judge.
    """
    rng = rng or np.random.default_rng(0)
    N, m = dev.shape
    full = dev.mean(axis=1)
    k_star = _argmax_tie(full, rng)
    boot_star = np.empty(n_boot)
    boot_gap = np.empty(n_boot)
    for b in range(n_boot):
        cols = rng.integers(0, m, size=m)
        scores = dev[:, cols].mean(axis=1)
        k = _argmax_tie(scores, rng)
        boot_star[b] = scores[k_star]
        boot_gap[b] = scores[k] - full[k]
    bias_hat = float(boot_gap.mean())
    point = float(full[k_star] - bias_hat)
    se = float(boot_star.std(ddof=1))
    z = norm.ppf(1 - alpha / 2)
    return Estimate(
        point,
        point - z * se,
        point + z * se,
        k_star,
        {"bias_hat": bias_hat, "boot_se": se},
    )


METHODS: dict[str, Callable[..., Estimate]] = {
    "none": no_correction,
    "split": naive_split,
    "union_K": union_bound_k,
    "union_neff": union_bound_neff,
    "cross_fit": cross_fitting,
    "bootstrap": bootstrap_plugin,
}

METHOD_LABELS = {
    "none": "No correction",
    "split": "Naive split",
    "union_K": "Union bound over K",
    "union_neff": "Union bound over N_eff",
    "cross_fit": "Cross-fitting",
    "bootstrap": "Bootstrap plug-in",
}
