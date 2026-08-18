"""Estimating the effective number of independent candidates.

A hundred paraphrases of one instruction succeed and fail on nearly the same items, so
the pool behaves like far fewer than ``K`` independent draws. Two estimators, reported
side by side: if they disagree sharply, the disagreement is the finding.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

MAX_LOG_ARG = 700.0  # keeps exp() away from overflow in the moment-matching inversion


def _residuals(C: np.ndarray, center: str) -> np.ndarray:
    """Center the correctness matrix before correlating candidates.

    ``center="candidate"`` (the default) removes each candidate's own mean accuracy and
    leaves shared item difficulty in place. That is deliberate: shared difficulty is
    precisely the channel through which dev-set noise becomes correlated across
    candidates, and it is what drives ``N_eff`` below ``K``.

    ``center="double"`` additionally removes item difficulty, which answers the
    different question of how much candidates disagree *beyond* item difficulty.
    """
    X = C.astype(float)
    X = X - X.mean(axis=1, keepdims=True)
    if center == "candidate":
        return X
    if center == "double":
        return X - X.mean(axis=0, keepdims=True)
    raise ValueError(f"center must be 'candidate' or 'double', got {center!r}")


def spectral_neff(C: np.ndarray, center: str = "candidate") -> dict:
    """Estimator A: participation ratio of the candidate correlation eigenvalues.

    ``N_eff = (sum lambda_i)^2 / sum lambda_i^2``. For a correlation matrix the
    eigenvalues sum to ``K``, so a perfectly independent pool gives ``N_eff = K`` and a
    pool driven by one shared factor gives ``N_eff -> 1``.
    """
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
    """``E[max of n iid standard normals]``, by Monte Carlo for small ``n``.

    The textbook ``sqrt(2 ln n)`` is only asymptotic and overshoots badly in the range
    that matters here (``n`` of order 5 to 100), so the numeric inversion uses this.
    """
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
    """Estimator B: invert the extreme-value relation for ``N_eff``.

    Given the empirically observed ``E[max_k (mu_hat_k - mu_k)]`` at dev size ``m`` and the
    per-item noise scale ``sigma``, solve for the candidate count that would have produced
    it. With ``asymptotic=True`` this is the closed form ``N_eff = exp(z^2 / 2)`` with
    ``z = observed * sqrt(m) / sigma``; otherwise ``z`` is matched numerically against
    :func:`expected_max_gaussian`, which is the honest version at realistic ``N``, where
    ``sqrt(2 ln n)`` overshoots badly.

    Args:
        n_pool: size of the pool the ``m`` items were drawn from. Sampling without
            replacement from a finite dev pool has standard error
            ``sigma * sqrt(1/m - 1/n_pool)``, and using ``sigma / sqrt(m)`` instead makes
            the estimate drift upward as ``m`` approaches the pool size.
    """
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
    """Per-item Bernoulli noise scale ``sigma``, averaged over candidates.

    Each candidate's per-item correctness is Bernoulli with variance
    ``mu_k (1 - mu_k)``; ``sigma`` is the root-mean of those variances.
    """
    mu = C.astype(float).mean(axis=1)
    return float(np.sqrt(np.mean(mu * (1.0 - mu))))


def observed_max_deviation(
    dev: np.ndarray,
    m: int,
    n_seeds: int = 500,
    seed: int = 0,
) -> float:
    """Empirical ``E[max_k (mu_hat_k - mu_k)]`` at dev size ``m``.

    This is the quantity estimator B inverts, and the reference ``mu_k`` is each candidate's
    mean over the *whole dev pool* rather than its truth-pool score. That choice is what
    makes the estimator well posed. Measuring against the truth pool would fold in two
    quantities that have nothing to do with candidate correlation: the truth pool's own
    sampling noise, which does not shrink as ``m`` grows, and the fixed difficulty gap
    between the two pools. Either one makes the inferred ``N_eff`` drift upward with ``m``,
    when ``N_eff`` is supposed to be a property of the pool alone.
    """
    rng = np.random.default_rng(seed)
    reference = dev.mean(axis=1)
    vals = np.empty(n_seeds)
    for s in range(n_seeds):
        cols = rng.choice(dev.shape[1], size=m, replace=False)
        vals[s] = (dev[:, cols].mean(axis=1) - reference).max()
    return float(vals.mean())
