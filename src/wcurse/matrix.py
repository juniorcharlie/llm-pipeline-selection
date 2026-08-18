"""The core instrument: a per-item correctness matrix plus its item splits.

Everything downstream of the GPU is a function of this object, so it carries its
own provenance and validates itself on load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROVENANCES = ("seed", "ape", "mutation")


@dataclass(frozen=True)
class CorrectnessMatrix:
    """A ``K x M`` binary matrix of candidate-by-item correctness.

    Attributes:
        C: ``(K, M)`` uint8 array, ``C[k, i] == 1`` iff candidate ``k`` answered item ``i``
            correctly.
        dev_idx: column indices forming the dev pool (used for selection).
        truth_idx: column indices forming the held-out truth pool, disjoint from ``dev_idx``.
        provenance: length-``K`` array of labels drawn from :data:`PROVENANCES`.
        task: dataset name, e.g. ``"sst2"``.
        model: model identifier, e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``.
        meta: free-form dictionary stored alongside the arrays (revision hashes, seeds).
    """

    C: np.ndarray
    dev_idx: np.ndarray
    truth_idx: np.ndarray
    provenance: np.ndarray
    task: str
    model: str
    meta: dict

    def __post_init__(self) -> None:
        if self.C.ndim != 2:
            raise ValueError(f"C must be 2-D, got shape {self.C.shape}")
        if not np.isin(self.C, (0, 1)).all():
            raise ValueError("C must contain only 0/1 entries")
        overlap = np.intersect1d(self.dev_idx, self.truth_idx)
        if overlap.size:
            raise ValueError(f"dev and truth pools overlap on {overlap.size} items")
        for name, idx in (("dev_idx", self.dev_idx), ("truth_idx", self.truth_idx)):
            if idx.size and (idx.min() < 0 or idx.max() >= self.M):
                raise ValueError(f"{name} out of range for M={self.M}")
        if self.provenance.shape != (self.K,):
            raise ValueError(f"provenance must have length K={self.K}")
        unknown = set(np.unique(self.provenance)) - set(PROVENANCES)
        if unknown:
            raise ValueError(f"unknown provenance labels: {sorted(unknown)}")

    @property
    def K(self) -> int:
        return self.C.shape[0]

    @property
    def M(self) -> int:
        return self.C.shape[1]

    def truth(self) -> np.ndarray:
        """True accuracy ``mu_k`` of every candidate, measured on the truth pool."""
        return self.C[:, self.truth_idx].mean(axis=1)

    def dev(self) -> np.ndarray:
        """The ``(K, len(dev_idx))`` sub-matrix that selection is allowed to see."""
        return self.C[:, self.dev_idx]

    def rows_with_provenance(self, label: str) -> np.ndarray:
        return np.flatnonzero(self.provenance == label)

    def subset_candidates(self, rows: np.ndarray) -> "CorrectnessMatrix":
        rows = np.asarray(rows)
        return CorrectnessMatrix(
            C=self.C[rows],
            dev_idx=self.dev_idx,
            truth_idx=self.truth_idx,
            provenance=self.provenance[rows],
            task=self.task,
            model=self.model,
            meta=self.meta,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            C=self.C.astype(np.uint8),
            dev_idx=self.dev_idx.astype(np.int32),
            truth_idx=self.truth_idx.astype(np.int32),
            provenance=self.provenance.astype("U16"),
            task=np.array(self.task),
            model=np.array(self.model),
            meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CorrectnessMatrix":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                C=z["C"].astype(np.uint8),
                dev_idx=z["dev_idx"],
                truth_idx=z["truth_idx"],
                provenance=z["provenance"],
                task=str(z["task"]),
                model=str(z["model"]),
                meta=json.loads(str(z["meta"])),
            )


def make_splits(
    n_items: int, n_dev: int, n_truth: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a disjoint dev/truth column split.

    Published as index lists rather than regenerated at analysis time, per the
    reproducibility checklist; this helper is the single place the split is born.
    """
    if n_dev + n_truth > n_items:
        raise ValueError(f"need {n_dev + n_truth} items, only {n_items} available")
    perm = np.random.default_rng(seed).permutation(n_items)
    return np.sort(perm[:n_dev]), np.sort(perm[n_dev : n_dev + n_truth])
