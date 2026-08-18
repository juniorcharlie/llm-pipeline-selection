import numpy as np
import pytest

from wcurse.corrections import METHOD_LABELS, METHODS
from wcurse.evaluate import evaluate_methods
from wcurse.synth import make_synthetic_matrix

CM = make_synthetic_matrix(seed=0)


@pytest.fixture(scope="module")
def results():
    return evaluate_methods(CM, N=50, m=100, n_seeds=200, seed=42)


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_method_returns_a_usable_estimate(name):
    dev = CM.dev()[:25, :80]
    est = METHODS[name](dev, alpha=0.10, rng=np.random.default_rng(0))
    assert est.lo <= est.point <= est.hi
    assert 0 <= est.selected < dev.shape[0]
    assert -0.5 <= est.point <= 1.5


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_method_is_seed_deterministic(name):
    dev = CM.dev()[:25, :80]
    a = METHODS[name](dev, rng=np.random.default_rng(7))
    b = METHODS[name](dev, rng=np.random.default_rng(7))
    assert (a.point, a.lo, a.hi, a.selected) == (b.point, b.lo, b.hi, b.selected)


def test_every_method_has_a_label():
    assert set(METHODS) == set(METHOD_LABELS)


def test_uncorrected_reporting_is_badly_biased_upward(results):
    assert results["none"]["bias"] > 0.03


def test_uncorrected_intervals_undercover_badly(results):
    """The headline calibration failure: nominal 90%, actual far below."""
    assert results["none"]["coverage"] < 0.75


def test_every_correction_reduces_absolute_bias(results):
    base = results["none"]["abs_bias"]
    for name, r in results.items():
        if name != "none":
            assert r["abs_bias"] < base, name


def test_union_bound_over_K_overcorrects(results):
    """Correcting over N rather than N_eff pushes the estimate below the truth."""
    assert results["union_K"]["bias"] < 0


def test_neff_correction_beats_union_over_K_on_mse(results):
    assert results["union_neff"]["mse"] < results["union_K"]["mse"]


def test_neff_correction_has_the_best_mse(results):
    best = min(results, key=lambda n: results[n]["mse"])
    assert best == "union_neff", {k: round(v["mse"], 4) for k, v in results.items()}


def test_naive_split_is_nearly_unbiased(results):
    assert abs(results["split"]["bias"]) < 0.5 * results["none"]["bias"]


def test_naive_split_pays_for_unbiasedness_with_width(results):
    assert results["split"]["mean_width"] > results["none"]["mean_width"]


def test_all_methods_see_the_same_draws(results):
    """Errors must be paired across methods for the Wilcoxon protocol to be valid."""
    sizes = {name: r["errors"].shape for name, r in results.items()}
    assert len(set(sizes.values())) == 1


def test_single_candidate_selection_is_unbiased_for_no_correction():
    """With N=1 the argmax is not a choice, so the raw dev score is honest."""
    res = evaluate_methods(CM, N=1, m=200, n_seeds=300, seed=11)
    assert abs(res["none"]["bias"]) < 0.03
