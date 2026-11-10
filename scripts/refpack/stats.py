"""
Section 2.3 of the work plan: multi-seed aggregation and paired significance
testing.

The differences between fusion operators in this study are in the third and
fourth decimal place. Nothing in the manuscript should claim an ordering that
this module cannot support.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np


# --------------------------------------------------------------------------
# effect size and paired tests
# --------------------------------------------------------------------------
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]. Computed by rank arithmetic so it
    stays O(n log n) rather than the O(n^2) pairwise definition."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    order = np.argsort(pooled, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, pooled.size + 1)
    # average ranks over ties
    _, inv, counts = np.unique(pooled, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_a = ranks[:n].sum()
    u_a = rank_a - n * (n + 1) / 2.0
    return float(2.0 * u_a / (n * m) - 1.0)


def interpret_delta(d: float) -> str:
    ad = abs(d)
    if np.isnan(ad):
        return "undefined"
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired test on per-patch scores. Uses scipy when available, otherwise a
    normal approximation with tie and zero correction."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[np.isfinite(d)]
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p), "n": int(d.size)}
    except Exception:
        nz = d[d != 0]
        n = nz.size
        if n < 10:
            return {"statistic": float("nan"), "p_value": float("nan"), "n": int(n)}
        absd = np.abs(nz)
        order = np.argsort(absd, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        w_plus = ranks[nz > 0].sum()
        mu = n * (n + 1) / 4.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w_plus - mu) / sigma
        from math import erfc, sqrt
        p = erfc(abs(z) / sqrt(2.0))
        return {"statistic": float(w_plus), "p_value": float(p), "n": int(n)}


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float],
                        n_boot: int = 10_000, alpha: float = 0.05,
                        seed: int = 0) -> Dict[str, float]:
    """Bootstrap confidence interval on the mean paired difference a - b.

    Resampling is over patches, paired, which is the unit of independence.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = a.size
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_diff": float((a - b).mean()), "ci_low": float(lo),
            "ci_high": float(hi), "n": int(n),
            "crosses_zero": bool(lo <= 0.0 <= hi)}


# --------------------------------------------------------------------------
# multi-seed reporting
# --------------------------------------------------------------------------
@dataclass
class SeedSummary:
    metric: str
    variant: str
    mean: float
    std: float
    n_seeds: int
    values: List[float]

    def cell(self, dec: int = 4) -> str:
        """Table cell in the form used by the manuscript."""
        return f"{self.mean:.{dec}f} $\\pm$ {self.std:.{dec}f}"


def summarise_seeds(per_seed: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, SeedSummary]]:
    """
    per_seed[variant][metric] is a list of values, one per seed.
    Returns summaries keyed by metric then variant.
    """
    out: Dict[str, Dict[str, SeedSummary]] = {}
    for variant, metrics in per_seed.items():
        for metric, values in metrics.items():
            vals = [float(v) for v in np.atleast_1d(values) if np.isfinite(v)]
            if not vals:
                continue
            out.setdefault(metric, {})[variant] = SeedSummary(
                metric=metric, variant=variant,
                mean=float(np.mean(vals)),
                # sample standard deviation; with three seeds this is noisy and
                # should be presented as an indication of spread, not a CI
                std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                n_seeds=len(vals), values=vals,
            )
    return out


def compare_variants(per_patch: Dict[str, Sequence[float]],
                     reference: str, seed: int = 0,
                     min_effect: float = 0.0) -> Dict[str, dict]:
    """
    Full paired comparison of every variant against a reference variant.

    min_effect is the smallest difference worth claiming. It matters here more
    than usual. With several thousand test patches the paired test has enough
    power to return a vanishingly small p-value for a difference in the fourth
    decimal place, which is statistically real and practically meaningless. A
    reviewer who notices that a significant result has a negligible effect size
    will say so. Set min_effect to the smallest difference that would change a
    decision downstream and report both verdicts.
    """
    report = {}
    ref = per_patch[reference]
    for variant, scores in per_patch.items():
        if variant == reference:
            continue
        w = wilcoxon_signed_rank(scores, ref)
        ci = paired_bootstrap_ci(scores, ref, seed=seed)
        d = cliffs_delta(scores, ref)
        report[variant] = {
            "vs": reference,
            **w,
            **ci,
            "cliffs_delta": d,
            "effect": interpret_delta(d),
            "verdict": (
                "no significant difference"
                if (not np.isfinite(w["p_value"])) or w["p_value"] >= 0.05 or ci["crosses_zero"]
                else ("better" if ci["mean_diff"] > 0 else "worse")
            ),
            "min_effect": min_effect,
            "exceeds_min_effect": bool(abs(ci["mean_diff"]) >= min_effect),
        }
        stat_sig = report[variant]["verdict"] != "no significant difference"
        practical = report[variant]["exceeds_min_effect"]
        report[variant]["practical_verdict"] = (
            report[variant]["verdict"] if (stat_sig and practical)
            else ("statistically significant but below the stated minimum "
                  "meaningful difference" if stat_sig else "no difference"))
    return report


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """Multiple-comparison correction. With nineteen metrics and three variants
    the family of tests is large enough that uncorrected p-values will be
    challenged in review."""
    items = sorted(((k, v) for k, v in p_values.items() if np.isfinite(v)),
                   key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (key, p) in enumerate(items):
        thresh = alpha / (m - i)
        adj = max(prev, min(1.0, p * (m - i)))
        prev = adj
        out[key] = {"p_raw": p, "p_adjusted": adj, "threshold": thresh,
                    "reject": adj < alpha}
    return out
