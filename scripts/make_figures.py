from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from wcurse import METHOD_LABELS

REPO = Path(__file__).resolve().parents[1]
PALETTE = plt.get_cmap("viridis")
PRIMARY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_all(results_root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {
        "bias_grid": [],
        "neff": [],
        "methods": [],
        "allocation": [],
    }
    runs = sorted(p for p in results_root.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"no run directories under {results_root}; run run_analysis.py first")
    for run in runs:
        for name in frames:
            path = run / f"{name}.csv"
            if path.exists():
                frames[name].append(pd.read_csv(path))
    return {k: pd.concat(v, ignore_index=True) for k, v in frames.items() if v}


def _panels(tasks: list[str], width: float = 4.0, height: float = 3.2):
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(width * len(tasks), height), squeeze=False, sharey=True
    )
    return fig, axes[0]


def pick(available: pd.Series, requested: int, label: str) -> int | None:
    values = sorted(int(v) for v in available.unique())
    if not values:
        print(f"NOTE: no data available to pick {label}; skipping the figure that needs it")
        return None
    if requested in values:
        return requested
    nearest = min(values, key=lambda v: abs(v - requested))
    print(f"NOTE: {label}={requested} is not in the grid {values}; using {nearest}")
    return nearest


def figure1_bias_vs_n(grid: pd.DataFrame, out: Path, m_fixed: int) -> None:
    tasks = sorted(grid["task"].unique())
    fig, axes = _panels(tasks)
    for ax, task in zip(axes, tasks):
        d = grid.query("task == @task and m == @m_fixed").sort_values("N")
        ax.errorbar(
            d["N"],
            d["bias"],
            yerr=[d["bias"] - d["bias_lo"], d["bias_hi"] - d["bias"]],
            marker="o",
            capsize=3,
            color=PALETTE(0.25),
            label="selective bias $B$",
        )
        ax.plot(d["N"], d["regret"], marker="s", ls="--", color=PALETTE(0.7), label="regret $R$")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(d["N"].unique())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("candidates $N$")
        ax.set_title(f"{task} ($m={m_fixed}$)")
    axes[0].set_ylabel("accuracy points")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure1_bias_vs_N.png", dpi=200)
    plt.close(fig)


def figure2_bias_vs_m(grid: pd.DataFrame, out: Path, n_fixed: int) -> None:
    tasks = sorted(grid["task"].unique())
    fig, axes = _panels(tasks)
    for ax, task in zip(axes, tasks):
        d = grid.query("task == @task and N == @n_fixed").sort_values("m")
        ax.errorbar(
            d["m"],
            d["bias"],
            yerr=[d["bias"] - d["bias_lo"], d["bias_hi"] - d["bias"]],
            marker="o",
            capsize=3,
            color=PALETTE(0.25),
            label="measured",
        )
        # The theory says bias falls like 1/sqrt(m); anchor the reference at the first point.
        anchor = d.iloc[0]
        ax.plot(
            d["m"],
            anchor["bias"] * np.sqrt(anchor["m"] / d["m"]),
            ls=":",
            color="0.4",
            label=r"$\propto 1/\sqrt{m}$",
        )
        ax.set_xscale("log")
        ax.minorticks_off()
        m_vals = sorted(d["m"].unique())
        ax.set_xticks(m_vals)
        ax.set_xticklabels([str(v) for v in m_vals])
        ax.set_xlabel("dev items $m$")
        ax.set_title(f"{task} ($N={n_fixed}$)")
    axes[0].set_ylabel("selective bias $B$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure2_bias_vs_m.png", dpi=200)
    plt.close(fig)


def figure3_heatmap(grid: pd.DataFrame, out: Path) -> None:
    tasks = sorted(grid["task"].unique())
    fig, axes = _panels(tasks, width=4.2, height=3.4)
    vmax = float(grid["bias"].max())
    for ax, task in zip(axes, tasks):
        piv = grid.query("task == @task").pivot(index="N", columns="m", values="bias")
        im = ax.imshow(piv.values, cmap="magma", vmin=0, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("dev items $m$")
        ax.set_title(task)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v < vmax * 0.6 else "black",
                )
    axes[0].set_ylabel("candidates $N$")
    fig.colorbar(im, ax=axes[-1], label="selective bias $B$")
    fig.tight_layout()
    fig.savefig(out / "figure3_bias_heatmap.png", dpi=200)
    plt.close(fig)


def figure4_spectrum(results_root: Path, neff: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for run in sorted(p for p in results_root.iterdir() if p.is_dir()):
        path = run / "spectra.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            lam = z["all"]
        task, model = run.name.split("__", 1)
        label = task if model.replace("__", "/") == PRIMARY_MODEL else f"{task} ({model.split('__')[-1]})"
        axes[0].plot(np.arange(1, lam.size + 1), lam / lam.sum(), marker=".", label=label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("eigenvalue rank")
    axes[0].set_ylabel("share of variance")
    axes[0].set_title("candidate correlation spectrum")
    axes[0].legend(frameon=False, fontsize=8)

    neff = neff[neff["model"] == PRIMARY_MODEL] if "model" in neff.columns else neff
    sub = neff.query("provenance != 'all'").groupby(["task", "provenance"], as_index=False).agg(
        neff_spectral_ratio=("neff_spectral_ratio", "first")
    )
    order = ["seed", "ape", "mutation"]
    width = 0.8 / max(len(sub["task"].unique()), 1)
    for i, task in enumerate(sorted(sub["task"].unique())):
        d = sub[sub["task"] == task].set_index("provenance").reindex(order)
        axes[1].bar(
            np.arange(len(order)) + i * width,
            d["neff_spectral_ratio"],
            width=width,
            label=task,
            color=PALETTE(0.15 + 0.6 * i / max(len(sub['task'].unique()), 1)),
        )
    axes[1].set_xticks(np.arange(len(order)) + 0.4 - width / 2, order)
    axes[1].set_ylabel("$N_{eff} / K$")
    axes[1].set_title("effective candidate count by provenance")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure4_neff.png", dpi=200)
    plt.close(fig)


def figure5_calibration(methods: pd.DataFrame, out: Path, alpha: float) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    nominal = 1 - alpha
    order = list(METHOD_LABELS)
    for i, name in enumerate(order):
        d = methods[methods["method"] == name].sort_values("m")
        by_m = d.groupby("m", as_index=False)["coverage"].mean()
        ax.plot(
            by_m["m"],
            by_m["coverage"],
            marker="o",
            color=PALETTE(i / len(order)),
            label=METHOD_LABELS[name],
        )
    ax.axhline(nominal, color="k", ls="--", lw=1, label=f"nominal {nominal:.0%}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("dev items $m$")
    ax.set_ylabel("interval coverage")
    ax.set_title("calibration by correction method")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figure5_calibration.png", dpi=200)
    plt.close(fig)


def figure6_allocation(alloc: pd.DataFrame, out: Path) -> None:
    tasks = sorted(alloc["task"].unique())
    fig, axes = _panels(tasks, height=3.4)
    for ax, task in zip(axes, tasks):
        d = alloc.query("task == @task and feasible")
        for i, (budget, g) in enumerate(d.groupby("budget")):
            g = g.sort_values("N")
            ax.errorbar(
                g["N"],
                g["true_mean"],
                yerr=g["true_se"],
                marker="o",
                capsize=3,
                color=PALETTE(0.15 + 0.7 * i / max(d["budget"].nunique() - 1, 1)),
                label=f"$B={int(budget)}$",
            )
            best = g.loc[g["true_mean"].idxmax()]
            ax.scatter([best["N"]], [best["true_mean"]], s=140, facecolors="none", edgecolors="red", zorder=5)
        ax.set_xscale("log")
        ax.set_xticks(d["N"].unique())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("candidates $N$  ($m = B/N$)")
        ax.set_title(task)
    axes[0].set_ylabel("true score of selected candidate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure6_allocation.png", dpi=200)
    plt.close(fig)


def write_tables(results_root: Path, data: dict[str, pd.DataFrame], out: Path, alpha: float) -> None:
    summaries = json.loads((results_root / "summaries.json").read_text())
    t1 = pd.DataFrame(
        [
            {
                "task": s["task"],
                "model": s["model"],
                "K": s["K"],
                "M": s["M"],
                "dev items": s["n_dev"],
                "truth items": s["n_truth"],
                "acc min": round(s["accuracy_min"], 3),
                "acc median": round(s["accuracy_median"], 3),
                "acc max": round(s["accuracy_max"], 3),
                "split offset": round(s["split_offset"], 4),
                "N_eff (spectral)": round(s["neff_spectral_all"], 1),
            }
            for s in summaries
        ]
    )

    grid = data["bias_grid"]
    published = REPO / "data" / "published_gains.csv"
    gains = pd.read_csv(published) if published.exists() else pd.DataFrame()
    t2 = grid.query("N in [10, 25, 50] and m in [50, 100, 200]")[
        ["task", "model", "N", "m", "bias", "bias_lo", "bias_hi"]
    ].round(4)
    if not gains.empty and gains["reported_gain"].notna().any():
        t2 = t2.merge(gains, on="task", how="left")
        t2["gain_explained_by_bias"] = (t2["bias"] / t2["reported_gain"]).round(2)
    else:
        t2["reported_gain"] = np.nan
        t2["gain_explained_by_bias"] = np.nan

    methods = data["methods"]
    t3 = (
        methods.groupby(["method_label", "N", "m"], as_index=False)
        .agg(bias=("bias", "mean"), mse=("mse", "mean"), coverage=("coverage", "mean"), width=("mean_width", "mean"))
        .round(4)
    )
    t3["coverage_error"] = (t3["coverage"] - (1 - alpha)).round(4)

    t4 = data["neff"][
        [
            "task",
            "model",
            "provenance",
            "K",
            "m",
            "neff_spectral",
            "neff_spectral_ratio",
            "neff_moment",
            "neff_moment_asymptotic",
        ]
    ].round(3)

    out.mkdir(parents=True, exist_ok=True)
    for name, df in [
        ("table1_summary", t1),
        ("table2_bias_vs_published", t2),
        ("table3_methods", t3),
        ("table4_neff", t4),
    ]:
        df.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(df.to_markdown(index=False))

    if gains.empty or not gains["reported_gain"].notna().any():
        print(
            "NOTE: table 2 has no published gains to compare against. "
            f"Fill in {published.relative_to(REPO)} (task, optimizer, reported_gain, citation) "
            "with numbers read off the papers; nothing is invented here."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=str(REPO / "results"))
    ap.add_argument("--figures", default=str(REPO / "figures"))
    ap.add_argument("--tables", default=str(REPO / "tables"))
    ap.add_argument("--m-fixed", type=int, default=100, help="dev size held fixed in figure 1")
    ap.add_argument("--n-fixed", type=int, default=50, help="candidate count held fixed in figure 2")
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    data = load_all(results_root)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    grid = data["bias_grid"]
    primary_grid = grid[grid["model"] == PRIMARY_MODEL]
    if primary_grid.empty:
        fallback_model = sorted(grid["model"].unique())[0]
        print(
            f"NOTE: {PRIMARY_MODEL!r} not present in results; "
            f"using {fallback_model!r} for figures 1, 2, 3, 6 instead"
        )
        primary_grid = grid[grid["model"] == fallback_model]
    alloc = data["allocation"]
    primary_alloc = alloc[alloc["model"] == PRIMARY_MODEL] if "model" in alloc.columns else alloc
    if "model" in alloc.columns and primary_alloc.empty and not primary_grid.empty:
        primary_alloc = alloc[alloc["model"] == primary_grid["model"].iloc[0]]

    m_fixed = pick(primary_grid["m"], args.m_fixed, "m")
    n_fixed = pick(primary_grid["N"], args.n_fixed, "N")
    if m_fixed is not None:
        figure1_bias_vs_n(primary_grid, fig_dir, m_fixed)
    if n_fixed is not None:
        figure2_bias_vs_m(primary_grid, fig_dir, n_fixed)
    figure3_heatmap(primary_grid, fig_dir)
    figure4_spectrum(results_root, data["neff"], fig_dir)
    figure5_calibration(data["methods"], fig_dir, args.alpha)
    figure6_allocation(primary_alloc, fig_dir)
    write_tables(results_root, data, Path(args.tables), args.alpha)

    print(f"figures written to {fig_dir}")
    print(f"tables written to {Path(args.tables)}")


if __name__ == "__main__":
    main()
