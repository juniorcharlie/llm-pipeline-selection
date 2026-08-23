import numpy as np
import pytest

from wcurse.matrix import CorrectnessMatrix, make_splits
from wcurse.synth import make_synthetic_matrix


def test_splits_are_disjoint_and_deterministic():
    a1, b1 = make_splits(600, 300, 300, seed=1)
    a2, b2 = make_splits(600, 300, 300, seed=1)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert np.intersect1d(a1, b1).size == 0
    assert a1.size == b1.size == 300


def test_splits_reject_impossible_request():
    with pytest.raises(ValueError, match="only 100 available"):
        make_splits(100, 60, 60, seed=0)


def test_rejects_overlapping_pools():
    with pytest.raises(ValueError, match="overlap"):
        CorrectnessMatrix(
            C=np.zeros((3, 10), dtype=np.uint8),
            dev_idx=np.arange(6),
            truth_idx=np.arange(4, 10),
            provenance=np.array(["seed"] * 3),
            task="t",
            model="m",
            meta={},
        )


def test_rejects_non_binary_entries():
    with pytest.raises(ValueError, match="0/1"):
        CorrectnessMatrix(
            C=np.array([[0, 2]], dtype=np.uint8),
            dev_idx=np.array([0]),
            truth_idx=np.array([1]),
            provenance=np.array(["seed"]),
            task="t",
            model="m",
            meta={},
        )


def test_rejects_unknown_provenance():
    with pytest.raises(ValueError, match="unknown provenance"):
        CorrectnessMatrix(
            C=np.zeros((1, 2), dtype=np.uint8),
            dev_idx=np.array([0]),
            truth_idx=np.array([1]),
            provenance=np.array(["handwritten"]),
            task="t",
            model="m",
            meta={},
        )


def test_save_load_roundtrip(tmp_path):
    cm = make_synthetic_matrix(n_seed=2, n_ape=3, n_mutation=3, M=40, n_dev=20, n_truth=20, seed=5)
    path = cm.save(tmp_path / "m.npz")
    back = CorrectnessMatrix.load(path)
    assert np.array_equal(back.C, cm.C)
    assert np.array_equal(back.dev_idx, cm.dev_idx)
    assert np.array_equal(back.provenance, cm.provenance)
    assert back.task == cm.task and back.model == cm.model
    assert back.meta["rho_mutation"] == cm.meta["rho_mutation"]


def test_truth_and_dev_use_disjoint_columns():
    cm = make_synthetic_matrix(M=100, n_dev=50, n_truth=50, seed=2)
    assert cm.truth().shape == (cm.K,)
    assert cm.dev().shape == (cm.K, 50)
    assert np.intersect1d(cm.dev_idx, cm.truth_idx).size == 0


def test_provenance_counts_match_request():
    cm = make_synthetic_matrix(n_seed=10, n_ape=45, n_mutation=45, seed=0)
    assert cm.K == 100
    assert cm.rows_with_provenance("seed").size == 10
    assert cm.rows_with_provenance("ape").size == 45
    assert cm.rows_with_provenance("mutation").size == 45


def test_subset_candidates_keeps_alignment():
    cm = make_synthetic_matrix(seed=0)
    rows = cm.rows_with_provenance("mutation")
    sub = cm.subset_candidates(rows)
    assert sub.K == rows.size
    assert set(np.unique(sub.provenance)) == {"mutation"}
    assert np.array_equal(sub.truth(), cm.truth()[rows])
