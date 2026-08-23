from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wcurse.matrix import CorrectnessMatrix
from wcurse.resample import simulate_cell

REPO = Path(__file__).resolve().parents[1]


def slug(model: str) -> str:
    return model.replace("/", "__")


def compare_one(task: str, model: str, n_seeds: int, seed: int) -> dict:
    replay_path = REPO / "results" / "evoprompt_replay" / f"{task}.json"
    if not replay_path.exists():
        raise SystemExit(f"no {replay_path}; run scripts/evoprompt_replay.py --task {task} first")
    replay = json.loads(replay_path.read_text())
    if "final_best" not in replay:
        raise SystemExit(f"{replay_path} is an unfinished trajectory; resume it first")

    matrix_path = REPO / "data" / "matrices" / f"{task}__{slug(model)}.npz"
    cm = CorrectnessMatrix.load(matrix_path)

    draws = simulate_cell(cm, N=replay["n_live"], m=replay["m"], n_seeds=n_seeds, seed=seed)
    bias = np.asarray(draws.bias)
    bias_replay_mean = float(bias.mean())
    bias_replay_sd = float(bias.std(ddof=1))  # spread of a SINGLE draw -- what a live run should be judged against
    lo, hi = (float(v) for v in np.percentile(bias, [5, 95]))  # empirical, no normality assumed

    live = replay["bias_live"]
    return {
        "task": task,
        "model": model,
        "N_live": replay["n_live"],
        "m": replay["m"],
        "bias_live": live,
        "dev_live": replay["final_best"]["fitness"],
        "true_live": replay["true_score"],
        "bias_replay_mean": bias_replay_mean,
        "bias_replay_sd": bias_replay_sd,
        "bias_replay_lo": lo,
        "bias_replay_hi": hi,
        "z_single_draw": (live - bias_replay_mean) / bias_replay_sd,
        "within_90pct_band": lo <= live <= hi,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tasks", nargs="*", default=["sst2", "subj", "agnews"])
    ap.add_argument("--n-seeds", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out", default=str(REPO / "results" / "evoprompt_replay" / "comparison.md"))
    args = ap.parse_args()

    rows = [compare_one(t, args.model, args.n_seeds, args.seed) for t in args.tasks]

    header = (
        "| task | N_live | m | bias_live | z (single-draw) | bias_replay: mean (sd) | "
        "90% predictive band | consistent? |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['N_live']} | {r['m']} | {r['bias_live']:+.4f} | "
            f"{r['z_single_draw']:+.2f} | "
            f"{r['bias_replay_mean']:+.4f} ({r['bias_replay_sd']:.4f}) | "
            f"[{r['bias_replay_lo']:+.4f}, {r['bias_replay_hi']:+.4f}] | "
            f"{'yes' if r['within_90pct_band'] else 'NO'} |\n"
        )
        print(
            f"{r['task']:8s} N_live={r['N_live']:3d} m={r['m']:3d}  "
            f"bias_live={r['bias_live']:+.4f}  z={r['z_single_draw']:+.2f}  "
            f"replay_mean={r['bias_replay_mean']:+.4f}  replay_sd={r['bias_replay_sd']:.4f}  "
            f"90% band=[{r['bias_replay_lo']:+.4f}, {r['bias_replay_hi']:+.4f}]  "
            f"{'OK' if r['within_90pct_band'] else 'MISMATCH'}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    (out_path.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
