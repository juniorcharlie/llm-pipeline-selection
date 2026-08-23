import numpy as np

from wcurse.evaluate import evaluate_methods
from wcurse.resample import simulate_grid, split_offset
from wcurse.sanity import (
    check_bias_monotone_in_m,
    check_bias_monotone_in_n,
    check_bias_zero_at_n1,
    check_corrected_beats_uncorrected,
    check_matrix_integrity,
    check_neff_below_k,
    check_selected_within_pool,
    run_all,
)
from wcurse.synth import make_synthetic_matrix

CM = make_synthetic_matrix(seed=0)
GRID = simulate_grid(CM, [1, 10, 50, 100], [20, 50, 200], n_seeds=300, seed=1)


def test_clean_pipeline_passes_every_check():
    results = evaluate_methods(CM, N=50, m=100, n_seeds=120, seed=2)
    checks = run_all(CM, GRID, n_eff=19.0, method_results=results)
    assert all(c.passed for c in checks), [str(c) for c in checks if not c.passed]


def test_monotone_in_m_catches_a_reversed_grid():
    broken = dict(GRID)
    broken[(50, 20)], broken[(50, 200)] = GRID[(50, 200)], GRID[(50, 20)]
    assert not check_bias_monotone_in_m(broken).passed


def test_monotone_in_n_catches_a_reversed_grid():
    broken = dict(GRID)
    broken[(10, 50)], broken[(100, 50)] = GRID[(100, 50)], GRID[(10, 50)]
    assert not check_bias_monotone_in_n(broken).passed


def test_monotone_checks_tolerate_pure_noise():
    flat = {
        (N, m): simulate_grid(CM, [1], [m], n_seeds=200, seed=100 + m)[(1, m)]
        for N in (1,)
        for m in (20, 50, 200)
    }
    assert check_bias_monotone_in_m(flat).passed
    assert check_bias_monotone_in_n(flat).passed


def test_n1_check_catches_a_leaky_truth_pool():
    assert check_bias_zero_at_n1(GRID, offset=split_offset(CM)).passed
    assert not check_bias_zero_at_n1(GRID, offset=0.25).passed


def test_pool_maximum_check_catches_swapped_columns():
    broken = dict(GRID)
    draws = GRID[(50, 50)]
    broken[(50, 50)] = type(draws)(
        reported=draws.reported,
        true=draws.pool_max_true + 0.01,
        dev_pool=draws.dev_pool,
        pool_max_true=draws.pool_max_true,
        selected=draws.selected,
        N=draws.N,
        m=draws.m,
    )
    assert not check_selected_within_pool(broken).passed


def test_neff_ceiling_check():
    assert check_neff_below_k(50.0, 100).passed
    assert not check_neff_below_k(101.0, 100).passed


def test_correction_check_catches_a_method_that_makes_things_worse():
    results = {"none": {"abs_bias": 0.05}, "split": {"abs_bias": 0.09}}
    out = check_corrected_beats_uncorrected(results)
    assert not out.passed
    assert "split" in out.detail


def test_correction_check_treats_union_over_K_overshoot_as_a_finding():
    results = {
        "none": {"abs_bias": 0.05},
        "split": {"abs_bias": 0.01},
        "union_neff": {"abs_bias": 0.01},
        "cross_fit": {"abs_bias": 0.02},
        "bootstrap": {"abs_bias": 0.03},
        "union_K": {"abs_bias": 0.08},
    }
    out = check_corrected_beats_uncorrected(results)
    assert out.passed
    assert "expected overcorrection" in out.detail


def test_correction_check_stands_down_when_bias_is_immaterial():
    results = {"none": {"abs_bias": 0.005}, "split": {"abs_bias": 0.04}}
    out = check_corrected_beats_uncorrected(results)
    assert out.passed
    assert "not applicable" in out.detail


def test_matrix_integrity_catches_duplicate_candidates():
    cm = make_synthetic_matrix(n_seed=4, n_ape=3, n_mutation=3, M=60, n_dev=30, n_truth=30, seed=3)
    assert check_matrix_integrity(cm).passed
    C = cm.C.copy()
    C[1] = C[0]
    duped = type(cm)(
        C=C,
        dev_idx=cm.dev_idx,
        truth_idx=cm.truth_idx,
        provenance=cm.provenance,
        task=cm.task,
        model=cm.model,
        meta=cm.meta,
    )
    out = check_matrix_integrity(duped)
    assert not out.passed and "duplicate" in out.detail


def test_check_result_renders_readably():
    c = check_neff_below_k(10.0, 100)
    assert str(c).startswith("[PASS]")
    assert "N_eff" in str(c)
