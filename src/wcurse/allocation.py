"""Budget allocation: how to spend ``N * m`` candidate-item evaluations.

Raising ``N`` raises the ceiling of the pool; raising ``m`` makes the choice of which
candidate to ship less noisy. Because ``N_eff`` saturates well below ``N``, the ceiling
gain has sharp diminishing returns while the noise penalty does not, so somewhere in the
middle there is an interior optimum.
"""

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
    """Trace expected true score of the selected candidate along ``m = budget / N``.

    Points whose implied ``m`` falls outside ``[1, dev pool size]`` or whose ``N`` exceeds
    the pool are skipped rather than extrapolated, and the skipping is reported so the
    figure can show where the budget line leaves the measurable region.
    """
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
    """The feasible point on an allocation curve with the highest expected true score."""
    feasible = [r for r in curve if r.get("feasible")]
    if not feasible:
        return None
    return max(feasible, key=lambda r: r["true_mean"])
