"""Run the full analysis on stored correctness matrices.

Everything the paper reports, from one command and no GPU::

    python scripts/run_analysis.py --matrices data/matrices/*.npz
    python scripts/run_analysis.py --synthetic          # validation on known ground truth

Outputs land in ``results/`` as CSV and JSON: the ``(N, m)`` bias grid, both effective
candidate-count estimators, the six-method correction comparison with Holm-corrected
paired tests, the allocation curves, and the sanity-check report. Nothing here touches a
model, so the whole thing reruns in minutes whenever an analysis choice changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wcurse import (
    CorrectnessMatrix,
    METHOD_LABELS,
    allocation_curve,
    bootstrap_ci,
    evaluate_methods,
    holm_bonferroni,
    make_synthetic_matrix,
    optimum,
    paired_comparison,
    simulate_grid,
    spectral_neff,
    split_offset,
)
from wcurse.matrix import PROVENANCES
from wcurse.neff import item_noise_scale, moment_matching_neff, observed_max_deviation
from wcurse.sanity import run_all

REPO = Path(__file__).resolve().parents[1]

N_VALUES = [1, 5, 10, 25, 50, 100]
M_VALUES = [20, 50, 100, 200]
BUDGETS = [1000, 2000, 5000]


def feasible_axes(cm: CorrectnessMatrix) -> tuple[list[int], list[int]]:
    """Trim the grid to what the matrix can actually support.

    The published matrices are 100 x 600 and support the whole grid, but tests and pilot
    matrices are smaller. Silently clipping is better than crashing halfway through, and
    the trimmed axes are printed so a short run is never mistaken for a full one.
    """
    n_values = [n for n in N_VALUES if n <= cm.K]
    m_values = [m for m in M_VALUES if m <= cm.dev().shape[1]]
    if not m_values:
        raise SystemExit(f"dev pool of {cm.dev().shape[1]} items is too small to analyse")
    return n_values, m_values


def bias_grid(
    cm: CorrectnessMatrix, n_values: list[int], m_values: list[int], n_seeds: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    """The (N, m) grid of selective bias and regret, with bootstrap intervals."""
    grid = simulate_grid(cm, n_values, m_values, n_seeds=n_seeds, seed=seed)
    offset = split_offset(cm)
    rows = []
    for (N, m), draws in grid.items():
        s = draws.summary()
        point, lo, hi = bootstrap_ci(draws.bias, alpha=0.10, seed=seed)
        r_point, r_lo, r_hi = bootstrap_ci(draws.regret, alpha=0.10, seed=seed)
        rows.append(
            {
                "task": cm.task,
                "model": cm.model,
                **s,
                "bias_lo": lo,
                "bias_hi": hi,
                "bias_adj": s["bias"] - offset,
                "regret_lo": r_lo,
                "regret_hi": r_hi,
                "split_offset": offset,
            }
        )
    return pd.DataFrame(rows).sort_values(["N", "m"]), grid


def neff_table(
    cm: CorrectnessMatrix, m_values: list[int], n_seeds: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Both estimators, for the whole pool and for each provenance stratum."""
    dev = cm.dev()
    n_pool = dev.shape[1]
    rows = []
    spectra = {}

    strata = [("all", np.arange(cm.K))] + [
        (p, cm.rows_with_provenance(p)) for p in PROVENANCES
    ]
    for label, idx in strata:
        if idx.size < 2:
            continue
        sub = dev[idx]
        spec = spectral_neff(sub)
        spectra[label] = spec["eigenvalues"]
        sigma = item_noise_scale(sub)
        for m in m_values:
            obs = observed_max_deviation(sub, m, n_seeds=n_seeds, seed=seed + m)
            mm = moment_matching_neff(obs, sigma, m, idx.size, n_pool=n_pool)
            mm_asym = moment_matching_neff(
                obs, sigma, m, idx.size, n_pool=n_pool, asymptotic=True
            )
            rows.append(
                {
                    "task": cm.task,
                    "model": cm.model,
                    "provenance": label,
                    "K": int(idx.size),
                    "m": m,
                    "sigma": sigma,
                    "neff_spectral": spec["n_eff"],
                    "neff_spectral_ratio": spec["n_eff"] / idx.size,
                    "observed_max_dev": obs,
                    "neff_moment": mm["n_eff"],
                    "neff_moment_asymptotic": mm_asym["n_eff"],
                    "neff_moment_saturated": mm["saturated"],
                }
            )
    return pd.DataFrame(rows), spectra


def method_comparison(
    cm: CorrectnessMatrix,
    n_values: list[int],
    m_values: list[int],
    n_seeds: int,
    seed: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Six correction methods on bias, MSE, and coverage, plus paired tests.

    Tests are paired by ``(task, resample seed)`` because every method sees the identical
    dev view on a given seed. The p-values are pooled across grid cells here and Holm
    corrected across the whole family by the caller.
    """
    cells = [(N, m) for N in n_values if N >= 10 for m in m_values if m >= 50]
    rows, tests = [], []
    per_cell = {}
    for N, m in cells:
        res = evaluate_methods(cm, N=N, m=m, n_seeds=n_seeds, seed=seed, alpha=alpha)
        per_cell[(N, m)] = res
        for name, r in res.items():
            rows.append(
                {
                    "task": cm.task,
                    "model": cm.model,
                    "method": name,
                    "method_label": METHOD_LABELS[name],
                    **{k: v for k, v in r.items() if k not in ("errors", "extra")},
                }
            )
        base = res["none"]["errors"] ** 2
        for name, r in res.items():
            if name == "none":
                continue
            t = paired_comparison(r["errors"] ** 2, base, f"{name} vs none (squared error)")
            tests.append(
                {
                    "task": cm.task,
                    "model": cm.model,
                    "N": N,
                    "m": m,
                    "method": name,
                    "comparison": t.label,
                    "n": t.n,
                    "statistic": t.statistic,
                    "p_raw": t.p_value,
                    "median_diff": t.median_diff,
                    "cliffs_delta": t.cliffs_delta,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(tests), per_cell


def allocation_table(
    cm: CorrectnessMatrix, n_values: list[int], n_seeds: int, seed: int
) -> pd.DataFrame:
    rows = []
    for budget in BUDGETS:
        curve = allocation_curve(
            cm, budget=budget, n_values=n_values, n_seeds=n_seeds, seed=seed
        )
        best = optimum(curve)
        for r in curve:
            rows.append(
                {
                    "task": cm.task,
                    "model": cm.model,
                    "budget": budget,
                    "is_optimum": bool(best and r.get("feasible") and r["N"] == best["N"]),
                    **r,
                }
            )
    return pd.DataFrame(rows)


def analyse(cm: CorrectnessMatrix, args, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_values, m_values = feasible_axes(cm)
    print(
        f"\n=== {cm.task} / {cm.model} (K={cm.K}, M={cm.M}) "
        f"grid N={n_values} m={m_values}"
    )

    grid_df, grid = bias_grid(cm, n_values, m_values, args.n_seeds, args.seed)
    neff_df, spectra = neff_table(cm, m_values, args.neff_seeds, args.seed)
    methods_df, tests_df, per_cell = method_comparison(
        cm, n_values, m_values, args.method_seeds, args.seed, args.alpha
    )
    # Grade the corrections in the corner where the curse bites hardest: the most
    # candidates and the fewest dev items. Using whichever cell happened to be evaluated
    # last would make the verdict depend on dictionary iteration order.
    worst_cell = max(per_cell, key=lambda c: (c[0], -c[1]))
    alloc_df = allocation_table(cm, n_values, args.alloc_seeds, args.seed)

    print(f"  correction check evaluated at N={worst_cell[0]}, m={worst_cell[1]}")
    grid_df.to_csv(out_dir / "bias_grid.csv", index=False)
    neff_df.to_csv(out_dir / "neff.csv", index=False)
    methods_df.to_csv(out_dir / "methods.csv", index=False)
    tests_df.to_csv(out_dir / "method_tests.csv", index=False)
    alloc_df.to_csv(out_dir / "allocation.csv", index=False)
    np.savez_compressed(out_dir / "spectra.npz", **spectra)

    spec_all = spectral_neff(cm.dev())["n_eff"]
    checks = run_all(
        cm,
        grid,
        n_eff=spec_all,
        method_results=per_cell[worst_cell],
        n_se=args.sanity_n_se,
    )
    (out_dir / "sanity.txt").write_text("\n".join(str(c) for c in checks) + "\n")
    for c in checks:
        print(" ", c)
    if any(not c.passed for c in checks) and not args.allow_sanity_failure:
        raise SystemExit(
            "sanity check failed: a violation means a bug, not a finding. "
            "Fix it before drafting figures, or pass --allow-sanity-failure to inspect."
        )

    summary = {
        "task": cm.task,
        "model": cm.model,
        "K": cm.K,
        "M": cm.M,
        "n_dev": int(cm.dev_idx.size),
        "n_truth": int(cm.truth_idx.size),
        "split_offset": split_offset(cm),
        "accuracy_min": float(cm.truth().min()),
        "accuracy_median": float(np.median(cm.truth())),
        "accuracy_max": float(cm.truth().max()),
        "neff_spectral_all": spec_all,
        "provenance_counts": {
            p: int((cm.provenance == p).sum()) for p in PROVENANCES
        },
        "headline_bias": {
            f"N={N},m={m}": float(
                grid_df.query("N == @N and m == @m")["bias"].iloc[0]
            )
            for N, m in [(25, 50), (50, 100), (100, 200)]
            if N in n_values and m in m_values
        },
        "sanity_all_passed": all(c.passed for c in checks),
        "matrix_meta": cm.meta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"summary": summary, "tests": tests_df, "methods": methods_df}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrices", nargs="*", default=[])
    ap.add_argument("--synthetic", action="store_true", help="analyse a synthetic matrix")
    ap.add_argument("--out-root", default=str(REPO / "results"))
    ap.add_argument("--n-seeds", type=int, default=500, help="resamples per grid cell")
    ap.add_argument("--method-seeds", type=int, default=200)
    ap.add_argument("--neff-seeds", type=int, default=400)
    ap.add_argument("--alloc-seeds", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.10, help="1 - nominal coverage")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument(
        "--sanity-n-se",
        type=float,
        default=3.0,
        help="how many standard errors of slack the monotonicity checks allow",
    )
    ap.add_argument("--allow-sanity-failure", action="store_true")
    args = ap.parse_args()

    matrices: list[CorrectnessMatrix] = [CorrectnessMatrix.load(p) for p in args.matrices]
    if args.synthetic or not matrices:
        matrices.append(make_synthetic_matrix(seed=args.seed % 2**31))

    out_root = Path(args.out_root)
    all_tests, all_summaries, all_methods = [], [], []
    for cm in matrices:
        out = analyse(cm, args, out_root / f"{cm.task}__{cm.model.replace('/', '__')}")
        all_summaries.append(out["summary"])
        all_tests.append(out["tests"])
        all_methods.append(out["methods"])

    # Holm-Bonferroni across the whole family: every method, every cell, every task.
    tests = pd.concat(all_tests, ignore_index=True)
    holm = holm_bonferroni(tests["p_raw"].tolist(), alpha=0.05)
    tests["p_holm"] = holm["adjusted"]
    tests["significant"] = holm["reject"]
    out_root.mkdir(parents=True, exist_ok=True)
    tests.to_csv(out_root / "method_tests_holm.csv", index=False)
    pd.concat(all_methods, ignore_index=True).to_csv(out_root / "methods_all.csv", index=False)
    (out_root / "summaries.json").write_text(json.dumps(all_summaries, indent=2, default=str))

    print(
        f"\nHolm-Bonferroni over {holm['n_tests']} paired tests at alpha=0.05: "
        f"{holm['n_reject']} significant"
    )
    print(f"results written to {out_root}")


if __name__ == "__main__":
    main()
