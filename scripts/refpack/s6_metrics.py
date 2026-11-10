"""
Section 6 of the work plan: metrics nobody has put on this problem yet.

The existing protocol treats seven ordinal height classes as if they were
seven unrelated labels, reports everything in pixels, and offers no way to
read an absolute score against a reference that is itself only about one
third correct. This module fixes all three.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (BIOMASS_IS_PROVISIONAL, BIOMASS_T_PER_KM, CARBON_FRACTION,
                     CARBON_FRACTION_SD, CLASS_CODE, CLASS_MID_M, N_CLASSES,
                     PIXEL_SIZE_M, REF_EXACT_AGREEMENT,
                     REF_WITHIN_ONE_AGREEMENT)


# ==========================================================================
# 6.1  the classes are ordinal, so treat them that way
# ==========================================================================
def ordinal_metrics(pred_class: np.ndarray, true_class: np.ndarray,
                    ignore: int = 0) -> Dict[str, float]:
    """
    pred_class, true_class: integer class maps, 1..7, with `ignore` for
    background and nodata.

    mae_metres is the number that changes how this paper reads. Deep canopy
    height work on forests reports errors of five to nine metres at ten to
    thirty metre resolution. If this model lands under a metre on a target
    two to four metres wide, that comparison belongs in the abstract, and it
    answers the reviewer who objects to an F1 of 0.43 far better than any
    argument about label noise.
    """
    m = (true_class != ignore) & (pred_class != ignore)
    if not m.any():
        return {k: float("nan") for k in
                ("mae_class", "rmse_class", "qwk", "within_one",
                 "exact", "mae_metres", "rmse_metres", "bias_metres", "n")}

    p = pred_class[m].astype(np.int64)
    t = true_class[m].astype(np.int64)
    d = p - t

    mid = np.asarray(CLASS_MID_M)
    dm = mid[p - 1] - mid[t - 1]

    return {
        "n": int(m.sum()),
        "exact": float(np.mean(d == 0)),
        "within_one": float(np.mean(np.abs(d) <= 1)),
        "within_two": float(np.mean(np.abs(d) <= 2)),
        "mae_class": float(np.mean(np.abs(d))),
        "rmse_class": float(np.sqrt(np.mean(d.astype(float) ** 2))),
        "qwk": quadratic_weighted_kappa(t, p),
        "mae_metres": float(np.mean(np.abs(dm))),
        "rmse_metres": float(np.sqrt(np.mean(dm ** 2))),
        # a signed bias tells you whether the model systematically under or
        # over-estimates height, which a symmetric error hides completely
        "bias_metres": float(np.mean(dm)),
    }


def quadratic_weighted_kappa(t: np.ndarray, p: np.ndarray,
                             n_classes: int = N_CLASSES) -> float:
    """Standard agreement statistic for ordinal grading. Penalises a two-class
    error four times as heavily as a one-class error, which is the right
    behaviour for height bands."""
    t = np.asarray(t).astype(int) - 1
    p = np.asarray(p).astype(int) - 1
    O = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(O, (t, p), 1.0)
    w = np.subtract.outer(np.arange(n_classes), np.arange(n_classes)) ** 2
    w = w / max((n_classes - 1) ** 2, 1)
    ht = np.bincount(t, minlength=n_classes).astype(float)
    hp = np.bincount(p, minlength=n_classes).astype(float)
    E = np.outer(ht, hp)
    E *= O.sum() / max(E.sum(), 1e-12)
    denom = (w * E).sum()
    return float(1.0 - (w * O).sum() / denom) if denom > 0 else float("nan")


# ==========================================================================
# 6.2  confusion in kilometres, not pixels
# ==========================================================================
def confusion_km(pred_class: np.ndarray, true_class: np.ndarray,
                 skeleton: Optional[np.ndarray] = None,
                 pixel_size_m: float = PIXEL_SIZE_M,
                 n_classes: int = N_CLASSES) -> Dict:
    """
    A seven by seven confusion matrix whose entries are kilometres of hedge.

    Restricting to the reference skeleton when one is supplied makes the row
    sums equal the mapped length of each class, which is the quantity a land
    manager recognises. Pixel counts over the buffer would instead be
    dominated by buffer area and would not mean anything outside the model.
    """
    if skeleton is not None:
        m = skeleton > 0
    else:
        m = true_class > 0
    t = true_class[m].astype(int)
    p = pred_class[m].astype(int)

    M = np.zeros((n_classes, n_classes + 1), dtype=float)   # last column: missed
    for ti, pi in zip(t, p):
        if ti < 1 or ti > n_classes:
            continue
        if pi < 1 or pi > n_classes:
            M[ti - 1, n_classes] += 1.0
        else:
            M[ti - 1, pi - 1] += 1.0

    M *= pixel_size_m / 1000.0
    row = M.sum(1)
    return {
        "matrix_km": M.tolist(),
        "row_labels": CLASS_CODE,
        "col_labels": CLASS_CODE + ["not detected"],
        "reference_length_km": row.tolist(),
        "total_reference_km": float(row.sum()),
        "detected_km": float(M[:, :n_classes].sum()),
        "missed_km": float(M[:, n_classes].sum()),
        "exact_km": float(np.trace(M[:, :n_classes])),
        "within_one_km": float(sum(
            M[i, j] for i in range(n_classes) for j in range(n_classes)
            if abs(i - j) <= 1)),
        "note": ("Lengths are raw pixel counts along the skeleton. Apply "
                 "refpack.s7_length.kulpa_length for the reported figure."),
    }


# ==========================================================================
# 6.3  the achievable ceiling
# ==========================================================================
def build_reference_confusion(exact: float = REF_EXACT_AGREEMENT,
                              within_one: float = REF_WITHIN_ONE_AGREEMENT,
                              n_classes: int = N_CLASSES) -> np.ndarray:
    """
    Reconstructs P(label = j | truth = i) for the reference product from the
    only two numbers its own validation publishes: exact agreement with field
    survey and agreement within one adjacent class.

    A one-parameter geometric decay over class offset has exactly the freedom
    needed. The decay rate is solved so that the diagonal reproduces `exact`,
    then checked against `within_one`; the residual is reported so that the
    fit is not silently assumed to be good.
    """
    from scipy.optimize import brentq

    def rows_for(rate: float) -> np.ndarray:
        off = np.abs(np.subtract.outer(np.arange(n_classes), np.arange(n_classes)))
        W = rate ** off
        return W / W.sum(axis=1, keepdims=True)

    def diag_gap(rate: float) -> float:
        return float(np.mean(np.diag(rows_for(rate)))) - exact

    try:
        rate = brentq(diag_gap, 1e-4, 0.999999)
    except Exception:
        rate = 0.5

    C = rows_for(rate)
    off = np.abs(np.subtract.outer(np.arange(n_classes), np.arange(n_classes)))
    achieved_w1 = float((C * (off <= 1)).sum(axis=1).mean())

    return {
        "matrix": C,
        "decay_rate": float(rate),
        "achieved_exact": float(np.mean(np.diag(C))),
        "achieved_within_one": achieved_w1,
        "target_within_one": within_one,
        "within_one_residual": achieved_w1 - within_one,
        "warning": ("A single decay parameter cannot satisfy both published "
                    "figures exactly. Report the residual; if it is large, "
                    "state that the ceiling is an approximation."),
    }


def achievable_ceiling(class_prior: Sequence[float],
                       score_fn,
                       n_pixels: int = 200_000,
                       n_trials: int = 200,
                       exact: float = REF_EXACT_AGREEMENT,
                       within_one: float = REF_WITHIN_ONE_AGREEMENT,
                       seed: int = 0) -> Dict:
    """
    The highest score any model could obtain against this reference.

    Monte Carlo: draw a latent truth from the observed class prior, corrupt it
    through the reference confusion to get the label the model is actually
    scored against, then score a hypothetical perfect model, whose prediction
    equals the truth, against that label.

    The resulting sentence is the one that settles the review argument about
    absolute numbers: no model, not even one that reproduces ground truth
    exactly, can score above X against this reference, and ours reaches Y
    percent of that ceiling.

    score_fn(pred, label) -> float, using the same metric as the paper.
    """
    rng = np.random.default_rng(seed)
    built = build_reference_confusion(exact, within_one)
    C = built["matrix"]
    prior = np.asarray(class_prior, dtype=float)
    prior = prior / prior.sum()

    scores = []
    for _ in range(n_trials):
        truth = rng.choice(len(prior), size=n_pixels, p=prior)
        u = rng.random(n_pixels)
        cdf = np.cumsum(C[truth], axis=1)
        label = (u[:, None] > cdf).sum(axis=1)
        label = np.clip(label, 0, len(prior) - 1)
        scores.append(score_fn(truth + 1, label + 1))

    s = np.asarray(scores, dtype=float)
    return {
        "ceiling_mean": float(s.mean()),
        "ceiling_std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "ceiling_ci95": [float(np.percentile(s, 2.5)),
                         float(np.percentile(s, 97.5))],
        "reference_confusion": {k: v for k, v in built.items() if k != "matrix"},
        "n_trials": n_trials,
        "assumption": ("Reference errors are assumed independent across "
                       "pixels and dependent only on the true class. Real "
                       "errors are spatially correlated, so this is an "
                       "optimistic ceiling and must be described as such."),
    }


def fraction_of_ceiling(our_score: float, ceiling: Dict) -> Dict:
    c = ceiling["ceiling_mean"]
    return {
        "our_score": our_score,
        "ceiling": c,
        "percent_of_ceiling": 100.0 * our_score / c if c else float("nan"),
        "headline": (f"No model can exceed {c:.4f} against this reference; "
                     f"ours reaches {100.0 * our_score / c:.1f} percent of that."),
    }


# ==========================================================================
# 6.4  gap-length distribution
# ==========================================================================
def gap_lengths(skeleton: np.ndarray, min_run: int = 1) -> np.ndarray:
    """
    Lengths, in pixels, of unbroken runs of background along a skeletonised
    network, measured along both axes.

    A hedge network with the right total length but broken at every gateway
    has the same F1 and a completely different ecological function. This is
    the distribution that exposes it.
    """
    runs = []
    for arr in (skeleton, skeleton.T):
        for row in arr:
            idx = np.flatnonzero(row > 0)
            if idx.size < 2:
                continue
            d = np.diff(idx) - 1
            runs.extend(d[d >= min_run].tolist())
    return np.asarray(runs, dtype=float)


def gap_distribution_test(pred_skeleton: np.ndarray,
                          true_skeleton: np.ndarray) -> Dict:
    a = gap_lengths(pred_skeleton)
    b = gap_lengths(true_skeleton)
    if a.size < 10 or b.size < 10:
        return {"n_pred_gaps": int(a.size), "n_true_gaps": int(b.size),
                "note": "too few gaps to test"}
    try:
        from scipy.stats import ks_2samp
        stat, p = ks_2samp(a, b)
    except Exception:
        stat, p = float("nan"), float("nan")
    return {
        "n_pred_gaps": int(a.size), "n_true_gaps": int(b.size),
        "median_pred_gap_m": float(np.median(a) * PIXEL_SIZE_M),
        "median_true_gap_m": float(np.median(b) * PIXEL_SIZE_M),
        "ks_statistic": float(stat), "ks_p_value": float(p),
        "excess_short_gaps": float(np.mean(a <= 3) - np.mean(b <= 3)),
        "reading": ("A positive excess of short gaps means the model is "
                    "fragmenting continuous hedges, which pixel metrics do "
                    "not see."),
    }


# ==========================================================================
# 6.5  landscape connectivity
# ==========================================================================
def network_graph(skeleton: np.ndarray):
    """Connected components of the skeleton, with their pixel areas, used as
    the nodes of a habitat network."""
    from scipy import ndimage as ndi
    lbl, n = ndi.label(skeleton > 0, structure=np.ones((3, 3)))
    if n == 0:
        return lbl, np.zeros(0), np.zeros((0, 2))
    areas = np.bincount(lbl.ravel())[1:].astype(float)
    cents = np.array(ndi.center_of_mass(skeleton > 0, lbl, range(1, n + 1)))
    return lbl, areas, cents


def connectivity_indices(skeleton: np.ndarray,
                         dispersal_m: float = 500.0,
                         pixel_size_m: float = PIXEL_SIZE_M,
                         max_components: int = 2000) -> Dict:
    """
    Integral index of connectivity and probability of connectivity, the two
    standard landscape-ecology measures, computed on the hedgerow network.

    These convert a segmentation result into the quantity a conservation
    planner actually uses. Reporting how much of the reference network's
    connectivity a map retains is a far stronger statement than reporting its
    clDice.
    """
    lbl, areas, cents = network_graph(skeleton)
    n = areas.size
    if n == 0:
        return {"n_components": 0, "IIC": 0.0, "PC": 0.0}
    if n > max_components:
        keep = np.argsort(areas)[::-1][:max_components]
        areas, cents = areas[keep], cents[keep]
        n = areas.size

    length = areas * pixel_size_m          # skeleton pixels approximate length
    total = length.sum()

    d = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1)) * pixel_size_m
    np.fill_diagonal(d, 0.0)

    # topological distance in steps: components closer than the dispersal
    # distance are one step apart, and shortest paths are taken over that graph
    adj = (d <= dispersal_m)
    nl = np.where(adj, 1.0, np.inf)
    np.fill_diagonal(nl, 0.0)
    for k in range(n):                      # Floyd-Warshall, n is bounded above
        nl = np.minimum(nl, nl[:, k][:, None] + nl[k, :][None, :])

    iic = float((np.outer(length, length) / (1.0 + nl)).sum() / (total ** 2))

    theta = 3.0 / max(dispersal_m, 1e-6)    # exponential dispersal kernel
    p = np.exp(-theta * d)
    np.fill_diagonal(p, 1.0)
    pmax = p.copy()
    for k in range(n):                      # maximum-product paths
        pmax = np.maximum(pmax, pmax[:, k][:, None] * pmax[k, :][None, :])
    pc = float((np.outer(length, length) * pmax).sum() / (total ** 2))

    return {"n_components": int(n), "total_length_m": float(total),
            "IIC": iic, "PC": pc, "dispersal_m": dispersal_m}


def connectivity_retention(pred_skeleton: np.ndarray, true_skeleton: np.ndarray,
                           dispersal_m: float = 500.0) -> Dict:
    a = connectivity_indices(pred_skeleton, dispersal_m)
    b = connectivity_indices(true_skeleton, dispersal_m)
    return {
        "predicted": a, "reference": b,
        "IIC_retention": a["IIC"] / b["IIC"] if b["IIC"] else float("nan"),
        "PC_retention": a["PC"] / b["PC"] if b["PC"] else float("nan"),
    }


# ==========================================================================
# 6.6  carbon, with uncertainty propagated
# ==========================================================================
def carbon_stock(length_km_by_class: Dict[str, float],
                 biomass: Dict[str, Tuple[float, float, float]] = None,
                 carbon_fraction: float = CARBON_FRACTION,
                 carbon_fraction_sd: float = CARBON_FRACTION_SD,
                 n_draws: int = 20_000, seed: int = 0) -> Dict:
    """
    Landscape carbon stock with a Monte Carlo uncertainty interval.

    The per-class biomass coefficients shipped with this package are
    PROVISIONAL placeholders so that the machinery can be exercised. They must
    be replaced with sourced values before any number derived from them
    appears in the manuscript, and the source must be recorded per class.
    """
    biomass = biomass or BIOMASS_T_PER_KM
    rng = np.random.default_rng(seed)

    totals = np.zeros(n_draws)
    per_class = {}
    for code, km in length_km_by_class.items():
        if code not in biomass:
            continue
        lo, best, hi = biomass[code]
        # triangular is the honest choice for an expert range with a mode
        draws = rng.triangular(lo, best, hi, n_draws) * km
        per_class[code] = {"length_km": km,
                           "biomass_t_mean": float(draws.mean())}
        totals += draws

    cf = rng.normal(carbon_fraction, carbon_fraction_sd, n_draws).clip(0.35, 0.6)
    carbon = totals * cf

    return {
        "biomass_t": {"mean": float(totals.mean()),
                      "ci95": [float(np.percentile(totals, 2.5)),
                               float(np.percentile(totals, 97.5))]},
        "carbon_t": {"mean": float(carbon.mean()),
                     "ci95": [float(np.percentile(carbon, 2.5)),
                              float(np.percentile(carbon, 97.5))]},
        "co2e_t": {"mean": float(carbon.mean() * 44.0 / 12.0)},
        "per_class": per_class,
        "coefficients_are_provisional": BIOMASS_IS_PROVISIONAL,
    }


def binary_map_carbon_bias(length_km_by_class: Dict[str, float],
                           **kw) -> Dict:
    """
    The sentence that makes this paper matter outside computer vision.

    Compares the carbon stock computed from the height-resolved map against
    the stock a binary hedgerow map would imply, where every hedge is assigned
    the length-weighted mean biomass. The difference is the cost, in carbon
    accounting terms, of not resolving height, and it is exactly the argument
    for doing this work at all.
    """
    resolved = carbon_stock(length_km_by_class, **kw)

    total_km = sum(length_km_by_class.values())
    if total_km <= 0:
        return {"error": "no length"}
    mean_b = float(np.mean([BIOMASS_T_PER_KM[c][1]
                            for c in length_km_by_class if c in BIOMASS_T_PER_KM]))
    naive_t = mean_b * total_km * CARBON_FRACTION

    true_t = resolved["carbon_t"]["mean"]
    return {
        "height_resolved_carbon_t": true_t,
        "binary_map_carbon_t": naive_t,
        "absolute_bias_t": naive_t - true_t,
        "relative_bias_percent": 100.0 * (naive_t - true_t) / true_t if true_t else float("nan"),
        "interpretation": ("Using a binary hedgerow map instead of a "
                           "height-resolved one biases the landscape carbon "
                           "estimate by this amount, because the length "
                           "distribution across height classes is not uniform."),
        "coefficients_are_provisional": BIOMASS_IS_PROVISIONAL,
    }


# ==========================================================================
# 6.7  orientation
# ==========================================================================
def line_orientations(skeleton: np.ndarray, window: int = 15) -> np.ndarray:
    """Local orientation, in radians, of every skeleton pixel, from the
    structure tensor. Axial data, so the range is [0, pi)."""
    from scipy import ndimage as ndi
    s = (skeleton > 0).astype(float)
    gy, gx = np.gradient(ndi.gaussian_filter(s, 1.0))
    jxx = ndi.uniform_filter(gx * gx, window)
    jyy = ndi.uniform_filter(gy * gy, window)
    jxy = ndi.uniform_filter(gx * gy, window)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    theta = (theta + np.pi / 2.0) % np.pi          # gradient to line direction
    return theta[skeleton > 0]


def orientation_test(pred_skeleton: np.ndarray,
                     true_skeleton: np.ndarray) -> Dict:
    """
    Watson's two-sample U-squared test on doubled angles, which is the correct
    treatment for axial data.

    Hedgerow orientation is not arbitrary. It reflects the historical field
    system and it governs shelter function. A model that reproduces length but
    not orientation has not reproduced the landscape.
    """
    a = line_orientations(pred_skeleton)
    b = line_orientations(true_skeleton)
    if a.size < 30 or b.size < 30:
        return {"n_pred": int(a.size), "n_true": int(b.size),
                "note": "too few skeleton pixels"}

    # doubling maps axial data onto the full circle
    a2 = (2 * a) % (2 * np.pi)
    b2 = (2 * b) % (2 * np.pi)

    def mean_dir(x):
        return float(np.arctan2(np.sin(x).mean(), np.cos(x).mean()) / 2.0 % np.pi)

    def resultant(x):
        return float(np.hypot(np.sin(x).mean(), np.cos(x).mean()))

    n, m = a2.size, b2.size
    pooled = np.sort(np.concatenate([a2, b2]))
    Fa = np.searchsorted(np.sort(a2), pooled, side="right") / n
    Fb = np.searchsorted(np.sort(b2), pooled, side="right") / m
    d = Fa - Fb
    N = n + m
    u2 = (n * m / N ** 2) * ((d ** 2).sum() - (d.sum() ** 2) / N)

    return {
        "n_pred": int(n), "n_true": int(m),
        "mean_orientation_pred_deg": float(np.degrees(mean_dir(a2))),
        "mean_orientation_true_deg": float(np.degrees(mean_dir(b2))),
        "concentration_pred": resultant(a2),
        "concentration_true": resultant(b2),
        "watson_u2": float(u2),
        "reject_same_distribution_at_5pct": bool(u2 > 0.187),
        "note": "Critical value 0.187 is the large-sample 5 percent point.",
    }


# ==========================================================================
# 6.8  error stratification
# ==========================================================================
def stratify_errors(records: List[Dict], metric: str,
                    by: str, bins: Sequence[float]) -> Dict:
    """
    Performance as a function of a scene property.

    A paper that says where and why a model fails earns more citations than
    one that reports a single number. Useful strata: hedge width, distance to
    woodland, terrain slope, solar azimuth relative to the line.
    """
    vals = np.array([r.get(by, np.nan) for r in records], dtype=float)
    scores = np.array([r.get(metric, np.nan) for r in records], dtype=float)
    idx = np.digitize(vals, bins)
    out = []
    for b in range(len(bins) + 1):
        m = (idx == b) & np.isfinite(scores)
        if m.sum() < 5:
            continue
        lo = "-inf" if b == 0 else f"{bins[b-1]:g}"
        hi = "inf" if b == len(bins) else f"{bins[b]:g}"
        out.append({"bin": f"[{lo}, {hi})", "n": int(m.sum()),
                    "mean": float(scores[m].mean()),
                    "std": float(scores[m].std(ddof=1)) if m.sum() > 1 else 0.0})
    return {"stratified_by": by, "metric": metric, "bins": out}


# ==========================================================================
# 6.9  label-free quality index
# ==========================================================================
def equivariance_error(model, dtm, ms, device=None) -> Dict[str, float]:
    """
    A quality signal that needs no labels at all.

    The mapping from a scene to a hedgerow map should commute with rotation
    and reflection. Any violation is pure model error, measurable anywhere the
    model is deployed, including the regions where no reference exists. This
    is what makes it usable for the second scene and for national rollout.
    """
    import torch
    model.eval()
    if device is not None:
        dtm, ms = dtm.to(device), ms.to(device)

    with torch.no_grad():
        base = torch.sigmoid(model(dtm, ms)["seg_logits"])

        transforms = {
            "rot90": (lambda x: torch.rot90(x, 1, (-2, -1)),
                      lambda x: torch.rot90(x, -1, (-2, -1))),
            "rot180": (lambda x: torch.rot90(x, 2, (-2, -1)),
                       lambda x: torch.rot90(x, -2, (-2, -1))),
            "rot270": (lambda x: torch.rot90(x, 3, (-2, -1)),
                       lambda x: torch.rot90(x, -3, (-2, -1))),
            "flip_h": (lambda x: torch.flip(x, (-1,)),
                       lambda x: torch.flip(x, (-1,))),
            "flip_v": (lambda x: torch.flip(x, (-2,)),
                       lambda x: torch.flip(x, (-2,))),
        }

        errs = {}
        for name, (fwd, inv) in transforms.items():
            y = torch.sigmoid(model(fwd(dtm), fwd(ms))["seg_logits"])
            errs[name] = float((inv(y) - base).abs().mean())

    errs["mean"] = float(np.mean(list(errs.values())))
    errs["max"] = float(np.max([v for k, v in errs.items() if k != "mean"]))
    return errs
