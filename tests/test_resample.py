import numpy as np

from wcurse.resample import argmax_random_tie, simulate_cell, simulate_grid, split_offset
from wcurse.synth import make_synthetic_matrix

CM = make_synthetic_matrix(seed=0)


def test_bias_grows_with_candidate_count():
    grid = simulate_grid(CM, [5, 25, 100], [50], n_seeds=400, seed=1)
    biases = [grid[(n, 50)].bias.mean() for n in (5, 25, 100)]
    assert biases[0] < biases[1] < biases[2]


def test_bias_shrinks_with_dev_size():
    grid = simulate_grid(CM, [50], [20, 50, 200], n_seeds=400, seed=2)
    biases = [grid[(50, m)].bias.mean() for m in (20, 50, 200)]
    assert biases[0] > biases[1] > biases[2]


def test_no_selection_no_bias_beyond_split_offset():
    draws = simulate_cell(CM, N=1, m=100, n_seeds=800, seed=3)
    se = draws.bias.std(ddof=1) / np.sqrt(draws.bias.size)
    assert abs(draws.bias.mean() - split_offset(CM)) < 4 * se


def test_bias_decomposition_is_exact():
    draws = simulate_cell(CM, N=25, m=50, n_seeds=100, seed=4)
    np.testing.assert_allclose(draws.bias, draws.item_bias + draws.pool_bias, atol=1e-12)


def test_selected_never_beats_pool_maximum():
    grid = simulate_grid(CM, [5, 50], [20, 100], n_seeds=200, seed=5)
    for draws in grid.values():
        assert np.all(draws.true <= draws.pool_max_true + 1e-12)


def test_regret_is_non_negative_and_zero_at_n1():
    grid = simulate_grid(CM, [1, 50], [50], n_seeds=200, seed=6)
    assert np.all(grid[(1, 50)].regret == 0)
    assert grid[(50, 50)].regret.mean() > 0


def test_grid_cells_are_independent_of_grid_shape():
    small = simulate_grid(CM, [10], [50], n_seeds=50, seed=7)
    big = simulate_grid(CM, [10, 25], [50, 100], n_seeds=50, seed=7)
    np.testing.assert_array_equal(small[(10, 50)].reported, big[(10, 50)].reported)


def test_simulation_is_seed_deterministic():
    a = simulate_cell(CM, N=25, m=50, n_seeds=50, seed=8)
    b = simulate_cell(CM, N=25, m=50, n_seeds=50, seed=8)
    np.testing.assert_array_equal(a.selected, b.selected)


def test_ties_are_broken_uniformly():
    rng = np.random.default_rng(0)
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    picks = np.array([argmax_random_tie(scores, rng) for _ in range(4000)])
    counts = np.bincount(picks, minlength=4)
    assert counts.min() > 850, counts


def test_rejects_oversized_requests():
    import pytest

    with pytest.raises(ValueError, match="exceeds pool size"):
        simulate_cell(CM, N=CM.K + 1, m=20, n_seeds=2, seed=0)
    with pytest.raises(ValueError, match="exceeds dev pool"):
        simulate_cell(CM, N=5, m=CM.dev().shape[1] + 1, n_seeds=2, seed=0)
