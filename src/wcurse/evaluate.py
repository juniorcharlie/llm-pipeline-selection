from __future__ import annotations

import numpy as np

from .corrections import METHODS, Estimate
from .matrix import CorrectnessMatrix
from .resample import draw_dev_view


def evaluate_methods(
    cm: CorrectnessMatrix,
    N: int,
    m: int,
    n_seeds: int,
    seed: int,
    alpha: float = 0.10,
    methods: dict | None = None,
) -> dict[str, dict]:
    
    methods = methods or METHODS
    truth = cm.truth()
    errors = {name: np.empty(n_seeds) for name in methods}
    covered = {name: np.zeros(n_seeds, dtype=bool) for name in methods}
    widths = {name: np.empty(n_seeds) for name in methods}
    extras: dict[str, list] = {name: [] for name in methods}

    for s in range(n_seeds):
        view_rng = np.random.default_rng(seed + s)
        dev_view, rows = draw_dev_view(cm, N=N, m=m, rng=view_rng)
        for name, fn in methods.items():
            est: Estimate = fn(dev_view, alpha=alpha, rng=np.random.default_rng(seed + s))
            true_val = truth[rows[est.selected]]
            errors[name][s] = est.point - true_val
            covered[name][s] = est.lo <= true_val <= est.hi
            widths[name][s] = est.hi - est.lo
            if est.extra:
                extras[name].append(est.extra)

    out = {}
    for name in methods:
        e = errors[name]
        out[name] = {
            "N": N,
            "m": m,
            "bias": float(e.mean()),
            "abs_bias": float(abs(e.mean())),
            "mse": float(np.mean(e**2)),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "coverage": float(covered[name].mean()),
            "mean_width": float(widths[name].mean()),
            "errors": e,
            "extra": extras[name],
        }
    return out
