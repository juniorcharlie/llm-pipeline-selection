from __future__ import annotations

import numpy as np

from .matrix import CorrectnessMatrix
from .resample import simulate_cell


def allocation_curve(
    cm: CorrectnessMatrix,
    budget: int,
    n_values: list[int],
    n_seeds: int = 200,
    seed: int = 0,
) -> list[dict]:

    dev_size = cm.dev().shape[1]
    rows = []
    for i, N in enumerate(sorted(n_values)):
        m = budget // N
        if N > cm.K or m < 1 or m > dev_size:
            rows.append({"N": N, "m": m, "feasible": False})
            continue
        draws = simulate_cell(cm, N=N, m=m, n_seeds=n_seeds, seed=seed + 7919 * (i + 1))
        se = draws.true.std(ddof=1) / np.sqrt(draws.true.size)
        rows.append(
            {
                "N": N,
                "m": m,
                "feasible": True,
                "budget": budget,
                "true_mean": float(draws.true.mean()),
                "true_se": float(se),
                "reported_mean": float(draws.reported.mean()),
                "bias": float(draws.bias.mean()),
                "regret": float(draws.regret.mean()),
            }
        )
    return rows


def optimum(curve: list[dict]) -> dict | None:
    feasible = [r for r in curve if r.get("feasible")]
    if not feasible:
        return None
    return max(feasible, key=lambda r: r["true_mean"])
