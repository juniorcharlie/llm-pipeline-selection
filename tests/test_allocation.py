import numpy as np

from wcurse.allocation import allocation_curve, optimum
from wcurse.synth import make_synthetic_matrix

CM = make_synthetic_matrix(seed=0)


def test_curve_marks_infeasible_points_instead_of_extrapolating():
    curve = allocation_curve(CM, budget=100000, n_values=[5, 100], n_seeds=20, seed=0)
    assert any(not r["feasible"] for r in curve)


def test_curve_respects_the_budget():
    curve = allocation_curve(CM, budget=2000, n_values=[10, 25, 50], n_seeds=20, seed=0)
    for r in curve:
        if r["feasible"]:
            assert r["N"] * r["m"] <= 2000


def test_optimum_is_interior_at_a_tight_budget():
    """The whole point of contribution 4: neither extreme wins."""
    n_values = [5, 10, 25, 50, 100]
    curve = allocation_curve(CM, budget=1000, n_values=n_values, n_seeds=400, seed=1)
    best = optimum(curve)
    feasible = [r["N"] for r in curve if r["feasible"]]
    assert best["N"] not in (min(feasible), max(feasible)), curve


def test_spending_everything_on_candidates_inflates_the_reported_number():
    curve = allocation_curve(CM, budget=1000, n_values=[5, 100], n_seeds=300, seed=2)
    by_n = {r["N"]: r for r in curve if r["feasible"]}
    assert by_n[100]["bias"] > by_n[5]["bias"]


def test_a_bigger_budget_never_hurts():
    small = optimum(allocation_curve(CM, 1000, [5, 10, 25, 50], n_seeds=400, seed=3))
    large = optimum(allocation_curve(CM, 4000, [5, 10, 25, 50], n_seeds=400, seed=3))
    assert large["true_mean"] >= small["true_mean"] - 3 * large["true_se"]


def test_optimum_returns_none_when_nothing_is_feasible():
    assert optimum([{"N": 5, "m": 0, "feasible": False}]) is None
