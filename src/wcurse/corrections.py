from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

from .neff import expected_max_gaussian, spectral_neff


@dataclass(frozen=True)
class Estimate:
    
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
    rng = rng or np.random.default_rng(0)
    N, m = dev.shape
    scores = dev.mean(axis=1)
    k = _argmax_tie(scores, rng)
    p = float(scores[k])
    z = norm.ppf(1 - alpha / 2)
    se = _binom_se(p, m)
    return Estimate(p, p - z * se, p + z * se, k)


def naive_split(dev: np.ndarray, alpha: float = 0.10, rng=None) -> Estimate:
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
    rng = rng or np.random.default_rng(0)
    N = dev.shape[0]
    return _evt_correction(dev, alpha, rng, float(N), {"n_used": float(N)})


def union_bound_neff(dev: np.ndarray, alpha: float = 0.10, rng=None) -> Estimate:
    rng = rng or np.random.default_rng(0)
    n_eff = spectral_neff(dev)["n_eff"]
    n_eff = float(np.clip(n_eff, 1.0, dev.shape[0]))
    return _evt_correction(dev, alpha, rng, n_eff, {"n_used": n_eff})


def cross_fitting(
    dev: np.ndarray, alpha: float = 0.10, rng=None, n_splits: int = 50
) -> Estimate:
    
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
