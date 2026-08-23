from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class PairedTest:

    label: str
    n: int
    statistic: float
    p_value: float
    median_diff: float
    cliffs_delta: float


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x) - np.asarray(y)
    return float((d > 0).mean() - (d < 0).mean())


def paired_comparison(x: np.ndarray, y: np.ndarray, label: str) -> PairedTest:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired samples must match in shape: {x.shape} vs {y.shape}")
    diff = x - y
    if np.allclose(diff, 0):
        # Wilcoxon is undefined when every difference is zero; report it honestly.
        return PairedTest(label, diff.size, 0.0, 1.0, 0.0, 0.0)
    stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return PairedTest(
        label=label,
        n=int(diff.size),
        statistic=float(stat),
        p_value=float(p),
        median_diff=float(np.median(diff)),
        cliffs_delta=cliffs_delta(x, y),
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> dict:
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return {
        "adjusted": adjusted,
        "reject": adjusted <= alpha,
        "alpha": alpha,
        "n_tests": n,
        "n_reject": int((adjusted <= alpha).sum()),
    }


def bootstrap_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    statistic=np.mean,
) -> tuple[float, float, float]:
    
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    reps = statistic(values[idx], axis=1)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return float(statistic(values)), float(lo), float(hi)
