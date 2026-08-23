import numpy as np

from wcurse.neff import (
    expected_max_gaussian,
    item_noise_scale,
    moment_matching_neff,
    observed_max_deviation,
    spectral_neff,
)
from wcurse.synth import make_synthetic_matrix


def independent_pool(K=60, M=400, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((K, M)) < 0.7).astype(np.uint8)


def single_factor_pool(K=60, M=400, seed=0):
    rng = np.random.default_rng(seed)
    difficulty = rng.random(M)
    return (rng.random((K, M)) < np.clip(0.3 + 0.55 * (difficulty > 0.5), 0, 1)).astype(np.uint8)


def test_spectral_neff_recovers_K_for_independent_pool():
    C = independent_pool()
    n_eff = spectral_neff(C)["n_eff"]
    assert n_eff > 0.85 * C.shape[0], n_eff


def test_spectral_neff_collapses_for_correlated_pool():
    C = single_factor_pool()
    assert spectral_neff(C)["n_eff"] < 0.35 * C.shape[0]


def test_spectral_neff_never_exceeds_K():
    for seed in range(5):
        C = independent_pool(seed=seed)
        assert spectral_neff(C)["n_eff"] <= C.shape[0] + 1e-9


def test_spectral_neff_survives_a_constant_candidate():
    C = independent_pool(K=20, M=200)
    C[0] = 1
    out = spectral_neff(C)
    assert out["n_live"] == 19
    assert np.isfinite(out["n_eff"])


def test_double_centering_removes_item_difficulty():
    C = single_factor_pool()
    assert spectral_neff(C, center="double")["n_eff"] > spectral_neff(C, center="candidate")["n_eff"]


def test_expected_max_gaussian_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for n in (2, 5, 25, 100):
        brute = rng.normal(size=(40000, n)).max(axis=1).mean()
        assert abs(expected_max_gaussian(n) - brute) < 0.03, n


def test_expected_max_gaussian_is_below_asymptotic_form():
    for n in (5, 10, 25, 50):
        assert expected_max_gaussian(n) < np.sqrt(2 * np.log(n))


def test_expected_max_gaussian_is_zero_for_single_candidate():
    assert expected_max_gaussian(1) == 0.0


def test_moment_matching_recovers_K_for_independent_pool():
    C = independent_pool(K=40, M=600)
    sigma = item_noise_scale(C)
    obs = observed_max_deviation(C, m=100, n_seeds=400, seed=1)
    out = moment_matching_neff(obs, sigma, m=100, K=40, n_pool=600)
    assert out["n_eff"] > 20, out


def test_moment_matching_is_capped_at_K():
    out = moment_matching_neff(0.5, sigma=0.5, m=100, K=30, n_pool=300)
    assert out["n_eff"] == 30.0
    assert out["saturated"]


def test_moment_matching_handles_degenerate_input():
    assert moment_matching_neff(0.0, sigma=0.5, m=50, K=10)["n_eff"] == 1.0


def test_observed_max_deviation_shrinks_with_m():
    C = independent_pool(K=40, M=600)
    devs = [observed_max_deviation(C, m=m, n_seeds=200, seed=2) for m in (20, 100, 400)]
    assert devs[0] > devs[1] > devs[2]


def test_mutation_pool_is_more_correlated_than_seed_pool():
    cm = make_synthetic_matrix(seed=0)
    dev = cm.dev()
    ratios = {}
    for label in ("seed", "ape", "mutation"):
        rows = cm.rows_with_provenance(label)
        ratios[label] = spectral_neff(dev[rows])["n_eff"] / rows.size
    assert ratios["mutation"] < ratios["seed"], ratios
