"""A synthetic correctness matrix with a tunable correlation regime.

The GPU step is the one compute-bound part of the plan, so every downstream component is
developed and tested against synthetic pools whose correlation structure is known by
construction. Mutation-derived candidates are generated in tight clusters, exactly the
structure the paper predicts real paraphrase pools will show.
"""

from __future__ import annotations

import numpy as np

from .matrix import CorrectnessMatrix, make_splits


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_matrix(
    n_seed: int = 10,
    n_ape: int = 45,
    n_mutation: int = 45,
    M: int = 600,
    n_dev: int = 300,
    n_truth: int = 300,
    ape_group_size: int = 3,
    mutation_group_size: int = 9,
    rho_ape: float = 0.25,
    rho_mutation: float = 0.80,
    skill_sd: float = 0.35,
    skill_mean: float = 0.9,
    difficulty_sd: float = 1.4,
    seed: int = 0,
) -> CorrectnessMatrix:
    """Generate a ``K x M`` correctness matrix from a latent factor model.

    Item ``i`` has difficulty ``b_i`` shared by every candidate, which is the channel that
    correlates dev-set noise across candidates. Candidates additionally share a
    group-level latent ``u_{g,i}`` with weight ``rho_g``: mutation groups are large and
    tightly coupled, APE groups small and loose, manual seeds independent.

    Args:
        rho_ape: within-group coupling for APE-generated candidates.
        rho_mutation: within-group coupling for paraphrase/mutation candidates; larger
            values mean a mutation family that succeeds and fails on the same items.

    Returns:
        A :class:`~wcurse.matrix.CorrectnessMatrix` with provenance labels and a disjoint
        dev/truth column split.
    """
    rng = np.random.default_rng(seed)

    groups: list[tuple[str, int, float]] = []  # (provenance, group id, within-group rho)
    gid = 0
    for _ in range(n_seed):
        groups.append(("seed", gid, 0.0))
        gid += 1
    for j in range(n_ape):
        if j % ape_group_size == 0:
            gid += 1
        groups.append(("ape", gid, rho_ape))
    for j in range(n_mutation):
        if j % mutation_group_size == 0:
            gid += 1
        groups.append(("mutation", gid, rho_mutation))

    K = len(groups)
    provenance = np.array([g[0] for g in groups])
    group_ids = np.array([g[1] for g in groups])
    rhos = np.array([g[2] for g in groups])

    b = rng.normal(0.0, difficulty_sd, size=M)
    a = rng.normal(skill_mean, skill_sd, size=K)
    u = rng.normal(0.0, 1.0, size=(group_ids.max() + 1, M))
    e = rng.normal(0.0, 1.0, size=(K, M))

    w_shared = np.sqrt(rhos)[:, None]
    w_own = np.sqrt(1.0 - rhos)[:, None]
    logit = a[:, None] - b[None, :] + w_shared * u[group_ids] + w_own * e
    C = (rng.random((K, M)) < _sigmoid(logit)).astype(np.uint8)

    dev_idx, truth_idx = make_splits(M, n_dev, n_truth, seed=seed + 1)
    return CorrectnessMatrix(
        C=C,
        dev_idx=dev_idx,
        truth_idx=truth_idx,
        provenance=provenance,
        task="synthetic",
        model="latent-factor-simulator",
        meta={
            "synthetic": True,
            "seed": seed,
            "rho_ape": rho_ape,
            "rho_mutation": rho_mutation,
            "ape_group_size": ape_group_size,
            "mutation_group_size": mutation_group_size,
        },
    )
