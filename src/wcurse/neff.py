from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

MAX_LOG_ARG = 700.0  # keeps exp() away from overflow in the moment-matching inversion


def _residuals(C: np.ndarray, center: str) -> np.ndarray:
    X = C.astype(float)
    X = X - X.mean(axis=1, keepdims=True)
    if center == "candidate":
        return X
    if center == "double":
        return X - X.mean(axis=0, keepdims=True)
    raise ValueError(f"center must be 'candidate' or 'double', got {center!r}")


def spectral_neff(C: np.ndarray, center: str = "candidate") -> dict:
    R = _residuals(C, center)
    sd = R.std(axis=1)
    live = sd > 0  # a candidate that is right (or wrong) on every item carries no signal
    R, sd = R[live], sd[live]
    K = R.shape[0]
    if K < 2:
        return {"n_eff": float(K), "eigenvalues": np.ones(K), "n_live": K}
    corr = (R / sd[:, None]) @ (R / sd[:, None]).T / R.shape[1]
    lam = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    n_eff = float(lam.sum() ** 2 / np.square(lam).sum())
    return {"n_eff": n_eff, "eigenvalues": lam[::-1], "n_live": K}


def expected_max_gaussian(n: float, n_mc: int = 20000, seed: int = 0) -> float:
    if n <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    u = rng.random((n_mc,))
    # Max of n iid uniforms is Beta(n, 1); invert through the normal CDF.
    return float(norm.ppf(u ** (1.0 / n)).mean())


def moment_matching_neff(
    observed_max_deviation: float,
    sigma: float,
    m: int,
    K: int,
    n_pool: int | None = None,
    asymptotic: bool = False,
) -> dict:

    se = sigma * np.sqrt(max(1.0 / m - (1.0 / n_pool if n_pool else 0.0), 1e-12))
    if se <= 0 or observed_max_deviation <= 0:
        return {"n_eff": 1.0, "z": 0.0, "saturated": False}
    z = observed_max_deviation / se

    saturated = False
    if asymptotic:
        raw = float(np.exp(min(z * z / 2.0, MAX_LOG_ARG)))
    elif expected_max_gaussian(K) <= z:
        raw = float(K)  # the pool is at least as spread out as K independent draws
        saturated = True
    else:
        raw = float(brentq(lambda n: expected_max_gaussian(n) - z, 1.0 + 1e-9, K))

    # N_eff is a count of candidates in the pool, so K is a hard ceiling (sanity check 5).
    return {"n_eff": min(raw, float(K)), "z": float(z), "saturated": saturated or raw > K}


def item_noise_scale(C: np.ndarray) -> float:
    mu = C.astype(float).mean(axis=1)
    return float(np.sqrt(np.mean(mu * (1.0 - mu))))


def observed_max_deviation(
    dev: np.ndarray,
    m: int,
    n_seeds: int = 500,
    seed: int = 0,
) -> float:

    rng = np.random.default_rng(seed)
    reference = dev.mean(axis=1)
    vals = np.empty(n_seeds)
    for s in range(n_seeds):
        cols = rng.choice(dev.shape[1], size=m, replace=False)
        vals[s] = (dev[:, cols].mean(axis=1) - reference).max()
    return float(vals.mean())
