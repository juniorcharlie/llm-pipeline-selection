from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matrix import CorrectnessMatrix
from .resample import SelectionDraws, split_offset


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def check_selected_within_pool(grid: dict[tuple[int, int], SelectionDraws]) -> CheckResult:
    worst = max(
        (float((d.true - d.pool_max_true).max()) for d in grid.values()), default=0.0
    )
    return CheckResult(
        "selected_true <= pool_max_true",
        worst <= 1e-12,
        f"max excess = {worst:.3e}",
    )


def _cell_se(draws: SelectionDraws) -> float:
    return float(draws.bias.std(ddof=1) / np.sqrt(draws.bias.size))


def _tolerance(a: SelectionDraws, b: SelectionDraws, n_se: float, floor: float) -> float:
    return max(floor, n_se * float(np.hypot(_cell_se(a), _cell_se(b))))


def check_bias_monotone_in_m(grid, n_se: float = 3.0, floor: float = 0.002) -> CheckResult:
    violations = []
    n_values = sorted({N for N, _ in grid})
    m_values = sorted({m for _, m in grid})
    for N in n_values:
        cells = [grid[(N, m)] for m in m_values if (N, m) in grid]
        for i in range(len(cells) - 1):
            a, b = cells[i], cells[i + 1]
            if b.bias.mean() > a.bias.mean() + _tolerance(a, b, n_se, floor):
                violations.append((N, a.m, b.m, a.bias.mean(), b.bias.mean()))
    return CheckResult(
        "bias non-increasing in m",
        not violations,
        f"{len(violations)} violation(s)" + (f": {violations[:3]}" if violations else ""),
    )


def check_bias_monotone_in_n(grid, n_se: float = 3.0, floor: float = 0.002) -> CheckResult:
    violations = []
    n_values = sorted({N for N, _ in grid})
    m_values = sorted({m for _, m in grid})
    for m in m_values:
        cells = [grid[(N, m)] for N in n_values if (N, m) in grid]
        for i in range(len(cells) - 1):
            a, b = cells[i], cells[i + 1]
            if b.bias.mean() < a.bias.mean() - _tolerance(a, b, n_se, floor):
                violations.append((m, a.N, b.N, a.bias.mean(), b.bias.mean()))
    return CheckResult(
        "bias non-decreasing in N",
        not violations,
        f"{len(violations)} violation(s)" + (f": {violations[:3]}" if violations else ""),
    )


def check_bias_zero_at_n1(grid, offset: float = 0.0, n_se: float = 3.0) -> CheckResult:
    cells = [d for (N, _), d in grid.items() if N == 1]
    if not cells:
        return CheckResult("bias ~ 0 at N=1", True, "no N=1 cells present (skipped)")
    ratios = [
        (abs(d.bias.mean() - offset) / max(_cell_se(d), 1e-12), d) for d in cells
    ]
    ratio, worst_cell = max(ratios, key=lambda t: t[0])
    return CheckResult(
        "bias ~ 0 at N=1 (offset-adjusted)",
        ratio <= n_se,
        f"worst cell m={worst_cell.m}: |bias - offset| = "
        f"{abs(worst_cell.bias.mean() - offset):.4f} = {ratio:.1f} SE "
        f"(limit {n_se:g} SE, offset = {offset:+.4f})",
    )


def check_neff_below_k(n_eff: float, K: int) -> CheckResult:
    return CheckResult("N_eff <= K", n_eff <= K + 1e-9, f"N_eff = {n_eff:.2f}, K = {K}")


def check_corrected_beats_uncorrected(
    method_results: dict[str, dict],
    baseline: str = "none",
    required: tuple[str, ...] = ("split", "union_neff", "cross_fit", "bootstrap"),
    min_baseline_bias: float = 0.02,
    tolerance: float = 0.003,
) -> CheckResult:
    
    base = method_results[baseline]["abs_bias"]
    if base < min_baseline_bias:
        return CheckResult(
            "recommended corrections closer to truth than uncorrected",
            True,
            f"not applicable: uncorrected |bias| = {base:.4f} < {min_baseline_bias:.4f}",
        )
    worse = {
        name: round(method_results[name]["abs_bias"], 4)
        for name in required
        if name in method_results and method_results[name]["abs_bias"] > base + tolerance
    }
    informational = {
        name: round(r["abs_bias"], 4)
        for name, r in method_results.items()
        if name not in required and name != baseline and r["abs_bias"] > base
    }
    detail = f"baseline |bias| = {base:.4f}"
    detail += f"; not improved by {worse}" if worse else "; all recommended methods improve"
    if informational:
        detail += f" (expected overcorrection, not a failure: {informational})"
    return CheckResult(
        "recommended corrections closer to truth than uncorrected", not worse, detail
    )


def check_matrix_integrity(cm: CorrectnessMatrix) -> CheckResult:
    problems = []
    if np.intersect1d(cm.dev_idx, cm.truth_idx).size:
        problems.append("dev/truth overlap")
    if cm.dev_idx.size == 0 or cm.truth_idx.size == 0:
        problems.append("empty split")
    if not np.isin(cm.C, (0, 1)).all():
        problems.append("non-binary entries")
    if np.unique(cm.C, axis=0).shape[0] < cm.K:
        problems.append("duplicate candidate rows (suspect prompt de-duplication)")
    return CheckResult(
        "matrix integrity",
        not problems,
        "; ".join(problems) if problems else f"K={cm.K}, M={cm.M}, splits disjoint",
    )


def run_all(
    cm: CorrectnessMatrix,
    grid: dict[tuple[int, int], SelectionDraws],
    n_eff: float | None = None,
    method_results: dict[str, dict] | None = None,
    n_se: float = 3.0,
) -> list[CheckResult]:

    checks = [
        check_matrix_integrity(cm),
        check_selected_within_pool(grid),
        check_bias_monotone_in_m(grid, n_se=n_se),
        check_bias_monotone_in_n(grid, n_se=n_se),
        check_bias_zero_at_n1(grid, offset=split_offset(cm), n_se=n_se),
    ]
    if n_eff is not None:
        checks.append(check_neff_below_k(n_eff, cm.K))
    if method_results is not None:
        checks.append(check_corrected_beats_uncorrected(method_results))
    return checks
