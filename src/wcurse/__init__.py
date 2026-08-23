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
