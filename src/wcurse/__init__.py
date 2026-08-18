"""Bias-corrected reporting for best-of-N selection over correlated candidate pools.

The short version, if you are reporting a best-of-N result::

    from wcurse import spectral_neff, union_bound_neff

    print(spectral_neff(dev)["n_eff"])       # how many candidates you effectively had
    est = union_bound_neff(dev, alpha=0.10)  # dev is your (N, m) binary correctness matrix

The rest of the public surface:

- :class:`~wcurse.matrix.CorrectnessMatrix` -- the stored instrument every result derives from.
- :func:`~wcurse.resample.simulate_grid` -- selection bias across the ``(N, m)`` grid.
- :func:`~wcurse.neff.spectral_neff`, :func:`~wcurse.neff.moment_matching_neff` -- the two
  effective-candidate-count estimators.
- :mod:`wcurse.corrections` -- six ways to report a winner, one of which you should use.
- :func:`~wcurse.allocation.allocation_curve` -- how to split a fixed budget between
  candidates and dev items.
"""

from .allocation import allocation_curve, optimum
from .corrections import (
    METHOD_LABELS,
    METHODS,
    Estimate,
    bootstrap_plugin,
    cross_fitting,
    naive_split,
    no_correction,
    union_bound_k,
    union_bound_neff,
)
from .evaluate import evaluate_methods
from .matrix import CorrectnessMatrix, make_splits
from .neff import item_noise_scale, moment_matching_neff, observed_max_deviation, spectral_neff
from .resample import SelectionDraws, simulate_cell, simulate_grid, split_offset
from .stats import bootstrap_ci, holm_bonferroni, paired_comparison
from .synth import make_synthetic_matrix

__version__ = "0.1.0"

__all__ = [
    "CorrectnessMatrix",
    "Estimate",
    "METHODS",
    "METHOD_LABELS",
    "SelectionDraws",
    "allocation_curve",
    "bootstrap_ci",
    "bootstrap_plugin",
    "cross_fitting",
    "evaluate_methods",
    "holm_bonferroni",
    "item_noise_scale",
    "make_splits",
    "make_synthetic_matrix",
    "moment_matching_neff",
    "naive_split",
    "no_correction",
    "observed_max_deviation",
    "optimum",
    "paired_comparison",
    "simulate_cell",
    "simulate_grid",
    "spectral_neff",
    "split_offset",
    "union_bound_k",
    "union_bound_neff",
]
