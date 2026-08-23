import subprocess
import sys
from pathlib import Path

import numpy as np

from wcurse.synth import make_synthetic_matrix

REPO = Path(__file__).resolve().parents[1]


def test_analysis_and_figures_regenerate_from_a_stored_matrix(tmp_path):
    cm = make_synthetic_matrix(n_seed=5, n_ape=15, n_mutation=15, M=400, n_dev=200, n_truth=200, seed=4)
    matrix_path = cm.save(tmp_path / "matrices" / "toy.npz")

    results = tmp_path / "results"
    run = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_analysis.py"),
            "--matrices", str(matrix_path),
            "--out-root", str(results),
            "--n-seeds", "120",
            "--method-seeds", "40",
            "--neff-seeds", "60",
            "--alloc-seeds", "60",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[FAIL]" not in run.stdout, run.stdout

    run_dir = next(p for p in results.iterdir() if p.is_dir())
    for name in ("bias_grid.csv", "neff.csv", "methods.csv", "allocation.csv", "sanity.txt", "summary.json"):
        assert (run_dir / name).exists(), name

    figures, tables = tmp_path / "figures", tmp_path / "tables"
    fig = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "make_figures.py"),
            "--results-root", str(results),
            "--figures", str(figures),
            "--tables", str(tables),
        ],
        capture_output=True,
        text=True,
    )
    assert fig.returncode == 0, fig.stdout + fig.stderr
    assert len(list(figures.glob("figure*.png"))) == 6
    assert len(list(tables.glob("table*.csv"))) == 4


def test_analysis_refuses_to_proceed_when_a_check_fails(tmp_path):
    cm = make_synthetic_matrix(n_seed=5, n_ape=10, n_mutation=10, M=200, n_dev=100, n_truth=100, seed=6)
    C = cm.C.copy()
    C[1] = C[0]  # duplicate candidate: trips the integrity check
    duped = type(cm)(
        C=C,
        dev_idx=cm.dev_idx,
        truth_idx=cm.truth_idx,
        provenance=cm.provenance,
        task="corrupt",
        model=cm.model,
        meta=cm.meta,
    )
    path = duped.save(tmp_path / "corrupt.npz")
    run = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_analysis.py"),
            "--matrices", str(path),
            "--out-root", str(tmp_path / "results"),
            "--n-seeds", "60",
            "--method-seeds", "20",
            "--neff-seeds", "40",
            "--alloc-seeds", "40",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "sanity check failed" in run.stdout + run.stderr


def test_figures_fail_loudly_without_results(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "make_figures.py"),
            "--results-root", str(tmp_path / "empty"),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0


def test_matrix_stores_enough_to_reproduce_itself(tmp_path):
    cm = make_synthetic_matrix(M=100, n_dev=50, n_truth=50, seed=9)
    back = type(cm).load(cm.save(tmp_path / "m.npz"))
    assert back.meta["seed"] == 9
    assert np.array_equal(np.sort(np.concatenate([back.dev_idx, back.truth_idx])), np.arange(100))
