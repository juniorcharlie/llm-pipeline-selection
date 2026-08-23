import numpy as np
import pytest

from wcurse.stats import bootstrap_ci, cliffs_delta, holm_bonferroni, paired_comparison


def test_holm_matches_worked_example():
    out = holm_bonferroni([0.01, 0.02, 0.03, 0.04], alpha=0.05)
    np.testing.assert_allclose(out["adjusted"], [0.04, 0.06, 0.06, 0.06])
    assert out["reject"].tolist() == [True, False, False, False]


def test_holm_is_monotone_in_the_original_ordering():
    p = [0.001, 0.5, 0.02, 0.9, 0.04]
    adj = holm_bonferroni(p)["adjusted"]
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_holm_never_reports_below_the_raw_p_value():
    p = [0.001, 0.01, 0.2]
    adj = holm_bonferroni(p)["adjusted"]
    assert np.all(adj >= np.asarray(p) - 1e-12)


def test_holm_is_more_conservative_than_no_correction():
    p = [0.03] * 10
    assert holm_bonferroni(p, alpha=0.05)["n_reject"] == 0


def test_paired_comparison_detects_a_real_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0.2, 1.0, size=200)
    y = rng.normal(0.0, 1.0, size=200)
    t = paired_comparison(x, y, "shifted")
    assert t.p_value < 0.05
    assert t.median_diff > 0
    assert t.cliffs_delta > 0


def test_paired_comparison_on_identical_samples_is_defined():
    x = np.arange(10.0)
    t = paired_comparison(x, x.copy(), "identical")
    assert t.p_value == 1.0
    assert t.median_diff == 0.0


def test_paired_comparison_requires_matching_shapes():
    with pytest.raises(ValueError, match="must match in shape"):
        paired_comparison(np.zeros(5), np.zeros(6), "mismatched")


def test_cliffs_delta_is_bounded_and_signed():
    assert cliffs_delta(np.ones(10), np.zeros(10)) == 1.0
    assert cliffs_delta(np.zeros(10), np.ones(10)) == -1.0
    assert cliffs_delta(np.zeros(10), np.zeros(10)) == 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    v = rng.normal(0.05, 0.1, size=500)
    point, lo, hi = bootstrap_ci(v, alpha=0.10, seed=0)
    assert lo < point < hi
    assert abs(point - v.mean()) < 1e-12


def test_bootstrap_ci_narrows_with_sample_size():
    rng = np.random.default_rng(2)
    wide = bootstrap_ci(rng.normal(0, 1, 50), seed=0)
    narrow = bootstrap_ci(rng.normal(0, 1, 5000), seed=0)
    assert (narrow[2] - narrow[1]) < (wide[2] - wide[1])


def test_bootstrap_ci_covers_the_truth_at_the_nominal_rate():
    covered = 0
    trials = 300
    for t in range(trials):
        v = np.random.default_rng(t).normal(0.1, 0.5, size=200)
        _, lo, hi = bootstrap_ci(v, alpha=0.10, n_boot=400, seed=t)
        covered += lo <= 0.1 <= hi
    assert 0.84 <= covered / trials <= 0.96, covered / trials
