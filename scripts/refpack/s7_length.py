"""
Section 7 of the work plan: how many kilometres of hedge are there?

Neither the reference nor the model sees the whole network, so the honest
answer is not a count but an estimate with an interval. This module provides
three independent routes to it, in increasing order of ambition.

  7.1  Length straight from the predicted proximity field, with no threshold
       and no skeletonisation, using the coarea formula.
  7.2  Digitisation bias correction for any length that does come from a
       skeleton.
  7.3  Capture-recapture estimation of the true network length, borrowing the
       standard estimators used in ecology for populations that are never
       fully observed.
  7.4  Positive-unlabelled correction of precision and recall.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DIST_CAP_M, PIXEL_SIZE_M


# ==========================================================================
# 7.1  length from the distance field, threshold free
# ==========================================================================
def length_from_field(D: np.ndarray, dist_cap_m: float = DIST_CAP_M,
                      pixel_size_m: float = PIXEL_SIZE_M,
                      intersection_correction: float = 1.0) -> Dict:
    """
    Estimates hedge length directly from the bounded proximity field the
    network outputs.

    The field decays linearly from the centreline, so its gradient magnitude
    is close to constant inside the buffer. By the coarea formula the integral
    of the gradient magnitude over the region equals the integral of the
    lengths of the level sets. For a distance field of buffer radius T around
    a line of length L the level set is two parallels of length L plus two end
    caps, giving

        integral |grad D| dA  =  2 L T + pi T^2

    which inverts for L.

    Why this is worth doing. The usual route is threshold, skeletonise, count
    pixels, and every one of those steps injects a choice and an error. This
    route has no threshold at all, is insensitive to topological breaks, and
    is differentiable, which means a length-calibration term could even be
    added to the training objective.

    Caveat that must be stated in the paper: at junctions the buffers of
    several lines overlap and the formula overestimates. Calibrate
    intersection_correction on synthetic networks with
    `validate_coarea_estimator` before using it on real data.
    """
    T = dist_cap_m / pixel_size_m                      # buffer radius in pixels
    field_m = np.asarray(D, dtype=np.float64) * dist_cap_m
    gy, gx = np.gradient(field_m)
    integral = float(np.hypot(gx, gy).sum())           # metres times pixels

    # units: the integral is in metres per pixel times pixel area, so dividing
    # by pixel_size_m twice is not needed; work in pixel units then convert
    T_px = T
    integral_px = integral / pixel_size_m
    L_px = (integral_px - np.pi * T_px ** 2) / (2.0 * T_px)
    L_px = max(L_px, 0.0) * intersection_correction

    return {
        "length_px": float(L_px),
        "length_m": float(L_px * pixel_size_m),
        "length_km": float(L_px * pixel_size_m / 1000.0),
        "gradient_integral": integral,
        "buffer_radius_px": float(T_px),
        "intersection_correction": intersection_correction,
    }


def validate_coarea_estimator(shape=(512, 512), dist_cap_m: float = DIST_CAP_M,
                              seed: int = 0) -> Dict:
    """
    Calibration on synthetic networks of known length.

    Three cases are generated: isolated straight lines, diagonals, and a grid
    with junctions. The ratio of estimated to true length on the junction case
    gives the correction factor to apply to real scenes. Without this step the
    estimator has no error bar and should not be reported.
    """
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    H, W = shape
    results = {}

    def field_from_mask(mask):
        d = ndi.distance_transform_edt(~mask) * PIXEL_SIZE_M
        return np.clip((dist_cap_m - d) / dist_cap_m, 0.0, 1.0)

    # straight horizontal lines, no junctions
    m = np.zeros(shape, bool)
    true_len = 0.0
    for r in (100, 250, 400):
        m[r, 60:W - 60] = True
        true_len += (W - 120)
    est = length_from_field(field_from_mask(m), dist_cap_m)["length_px"]
    results["straight"] = {"true_px": true_len, "est_px": est,
                           "ratio": est / true_len}

    # diagonals, where pixel counting is most biased
    m = np.zeros(shape, bool)
    true_len = 0.0
    for off in (-120, 0, 120):
        rr = np.arange(60, H - 60)
        cc = np.clip(rr + off, 0, W - 1)
        m[rr, cc] = True
        true_len += len(rr) * np.sqrt(2.0)
    est = length_from_field(field_from_mask(m), dist_cap_m)["length_px"]
    results["diagonal"] = {"true_px": true_len, "est_px": est,
                           "ratio": est / true_len}

    # a grid with junctions, the realistic case
    m = np.zeros(shape, bool)
    true_len = 0.0
    for r in (120, 260, 400):
        m[r, 40:W - 40] = True
        true_len += (W - 80)
    for c in (120, 260, 400):
        m[40:H - 40, c] = True
        true_len += (H - 80)
    est = length_from_field(field_from_mask(m), dist_cap_m)["length_px"]
    results["grid_with_junctions"] = {"true_px": true_len, "est_px": est,
                                      "ratio": est / true_len}

    ratio = results["grid_with_junctions"]["ratio"]
    results["recommended_intersection_correction"] = float(1.0 / ratio) if ratio else 1.0
    results["note"] = ("Apply the recommended correction to real scenes and "
                       "report it in the paper alongside this validation.")
    return results


# ==========================================================================
# 7.2  digitisation bias
# ==========================================================================
def kulpa_length(skeleton: np.ndarray, pixel_size_m: float = PIXEL_SIZE_M) -> Dict:
    """
    Length of a rasterised curve, corrected for digitisation bias.

    Counting skeleton pixels overestimates by roughly five to ten percent
    because diagonal runs are staircased. Kulpa's weights, 0.948 for
    orthogonal steps and 1.343 for diagonal ones, remove most of that bias.
    Almost nobody in remote sensing applies this, and reviewers who know the
    digital geometry literature will notice that it has been done.
    """
    s = (np.asarray(skeleton) > 0)
    if s.sum() == 0:
        return {"length_px_raw": 0.0, "length_px_kulpa": 0.0, "length_km": 0.0}

    # count adjacency steps once each, using only the forward half-neighbourhood
    ortho = 0
    diag = 0
    ortho += int((s[:, :-1] & s[:, 1:]).sum())          # right
    ortho += int((s[:-1, :] & s[1:, :]).sum())          # down
    diag += int((s[:-1, :-1] & s[1:, 1:]).sum())        # down-right
    diag += int((s[:-1, 1:] & s[1:, :-1]).sum())        # down-left

    raw = float(ortho + diag * np.sqrt(2.0))
    kulpa = 0.948 * ortho + 1.343 * diag

    return {
        "n_pixels": int(s.sum()),
        "orthogonal_steps": ortho, "diagonal_steps": diag,
        "length_px_raw": raw,
        "length_px_kulpa": float(kulpa),
        "length_km": float(kulpa * pixel_size_m / 1000.0),
        "bias_removed_percent": float(100.0 * (raw - kulpa) / raw) if raw else 0.0,
    }


# ==========================================================================
# 7.3  capture-recapture
# ==========================================================================
def segment_capture_history(detections: List[np.ndarray],
                            tolerance_px: int = 10) -> Dict:
    """
    Builds a capture history over hedge segments.

    Each input is a binary map from one detector: the reference product, or a
    model at a given threshold, or a different model. Two detections refer to
    the same segment if they overlap once dilated by the tolerance, which
    absorbs the several-metre positional uncertainty the reference is
    documented to have.

    Returns, for each distinct segment, the number of detectors that saw it.
    """
    from scipy import ndimage as ndi

    if not detections:
        return {"histories": np.zeros(0), "n_detectors": 0}

    st = np.ones((3, 3), bool)
    dil = [ndi.binary_dilation(d > 0, structure=st, iterations=tolerance_px)
           for d in detections]

    union = np.zeros_like(dil[0], dtype=bool)
    for d in dil:
        union |= d
    lbl, n = ndi.label(union, structure=st)
    if n == 0:
        return {"histories": np.zeros(0), "n_detectors": len(detections)}

    hist = np.zeros(n, dtype=int)
    lengths = np.zeros(n, dtype=float)
    for d_raw, d_dil in zip(detections, dil):
        seen = np.unique(lbl[d_dil & (lbl > 0)])
        hist[seen[seen > 0] - 1] += 1
    for i in range(1, n + 1):
        lengths[i - 1] = float((lbl == i).sum())

    return {"histories": hist, "segment_sizes": lengths,
            "n_segments_observed": int(n), "n_detectors": len(detections)}


def lincoln_petersen(n_ref: float, n_pred: float, n_matched: float) -> Dict:
    """
    Two-sample estimator with Chapman's small-sample correction.

    The reference is capture one, the model is capture two, the overlap is the
    recapture. Applied to length in kilometres rather than to individuals.
    """
    if n_matched <= 0:
        return {"estimate": float("nan"), "note": "no overlap, estimator undefined"}

    chapman = ((n_ref + 1.0) * (n_pred + 1.0)) / (n_matched + 1.0) - 1.0
    var = (((n_ref + 1) * (n_pred + 1) * (n_ref - n_matched) * (n_pred - n_matched))
           / ((n_matched + 1) ** 2 * (n_matched + 2)))
    se = float(np.sqrt(max(var, 0.0)))

    return {
        "estimate": float(chapman),
        "standard_error": se,
        "ci95": [float(chapman - 1.96 * se), float(chapman + 1.96 * se)],
        "reference_coverage": float(n_ref / chapman) if chapman > 0 else float("nan"),
        "model_coverage": float(n_pred / chapman) if chapman > 0 else float("nan"),
        "assumption_violated": (
            "Independence between the two detectors is assumed and is not "
            "satisfied here: the reference and the model tend to miss the same "
            "difficult hedges, which biases the estimate downward. Report this "
            "as a lower bound, not as an unbiased estimate."),
    }


def chao_estimator(histories: np.ndarray,
                   sizes: Optional[np.ndarray] = None) -> Dict:
    """
    Chao's lower-bound estimator for a population with heterogeneous detection
    probability.

    More robust than Lincoln-Petersen for this problem, because detection
    probability genuinely varies: a tall isolated hedge is easy for every
    detector, a low fragmented one is hard for all of them. Chao uses only the
    counts of segments seen exactly once and exactly twice, which is precisely
    the part of the data that carries information about what was missed.

    Supply several detectors: the reference, each model variant, and each
    model at a second threshold.
    """
    h = np.asarray(histories)
    h = h[h > 0]
    if h.size == 0:
        return {"estimate": float("nan")}

    s_obs = float(h.size)
    f1 = float((h == 1).sum())
    f2 = float((h == 2).sum())

    if f2 > 0:
        est = s_obs + (f1 ** 2) / (2.0 * f2)
        var = f2 * (0.25 * (f1 / f2) ** 4 + (f1 / f2) ** 3 + 0.5 * (f1 / f2) ** 2)
    else:
        est = s_obs + f1 * (f1 - 1.0) / 2.0
        var = 0.25 * f1 * (2 * f1 - 1) ** 2 / 4.0 - f1 ** 4 / (4.0 * max(est, 1))

    se = float(np.sqrt(max(var, 0.0)))
    out = {
        "observed_segments": int(s_obs),
        "f1_seen_once": int(f1), "f2_seen_twice": int(f2),
        "estimated_total_segments": float(est),
        "standard_error": se,
        "ci95": [float(max(s_obs, est - 1.96 * se)), float(est + 1.96 * se)],
        "completeness": float(s_obs / est) if est > 0 else float("nan"),
        "note": ("Chao is a lower bound by construction, so the true network "
                 "is at least this large."),
    }

    # the estimator is driven by f1^2 / 2 f2, so a large f1 with a small f2
    # produces an enormous and meaningless number. That happens when the
    # detectors fragment the network differently rather than missing whole
    # segments, which is exactly what a per-pixel dropout does.
    if f2 < 5 or (f1 > 0 and f2 > 0 and f1 / f2 > 6.0):
        out["unstable"] = True
        out["warning"] = (
            "f1 is large relative to f2, so the estimate is dominated by "
            "singletons and should not be reported. Usually this means the "
            "detectors are fragmenting the same hedges differently instead of "
            "missing different hedges. Increase the match tolerance so that "
            "fragments of one hedge merge into one segment, or compare at the "
            "level of whole hedges rather than connected components.")
    else:
        out["unstable"] = False

    if sizes is not None and sizes.size == np.asarray(histories).size:
        # scale from segment counts to length using the mean size of segments
        # seen exactly once, since undetected segments resemble those
        s = np.asarray(sizes)[np.asarray(histories) > 0]
        rare = s[h == 1]
        mean_rare = float(rare.mean()) if rare.size else float(s.mean())
        out["estimated_unseen_length_px"] = float((est - s_obs) * mean_rare)
        out["observed_length_px"] = float(s.sum())
        out["estimated_total_length_px"] = float(s.sum() + (est - s_obs) * mean_rare)

    return out


def network_length_report(reference: np.ndarray, predictions: Dict[str, np.ndarray],
                          tolerance_px: int = 10,
                          pixel_size_m: float = PIXEL_SIZE_M) -> Dict:
    """
    The headline number: how long is the hedgerow network really, and what
    share of it does each source see?

    This is the sentence no one has been able to write for this landscape.
    """
    detectors = [reference] + list(predictions.values())
    names = ["reference"] + list(predictions)

    cap = segment_capture_history(detectors, tolerance_px)
    chao = chao_estimator(cap["histories"], cap.get("segment_sizes"))

    km = lambda a: float(kulpa_length(a, pixel_size_m)["length_km"])
    ref_km = km(reference)

    two_sample = {}
    for name, pred in predictions.items():
        from scipy import ndimage as ndi
        st = np.ones((3, 3), bool)
        overlap = (ndi.binary_dilation(reference > 0, st, tolerance_px)
                   & (pred > 0))
        two_sample[name] = lincoln_petersen(ref_km, km(pred), km(overlap))

    total_est_km = None
    if "estimated_total_length_px" in chao:
        total_est_km = chao["estimated_total_length_px"] * pixel_size_m / 1000.0

    return {
        "detectors": names,
        "reference_length_km": ref_km,
        "predicted_length_km": {n: km(p) for n, p in predictions.items()},
        "two_sample_estimates_km": two_sample,
        "multi_sample_chao": chao,
        "estimated_true_network_km": total_est_km,
        "coverage": ({n: km(p) / total_est_km for n, p in predictions.items()}
                     if total_est_km else {}),
        "reference_coverage": ref_km / total_est_km if total_est_km else None,
    }


# ==========================================================================
# 7.4  positive-unlabelled correction
# ==========================================================================
def estimate_label_frequency(scores_on_known_positives: Sequence[float],
                             threshold: float = 0.5) -> Dict:
    """
    Estimates c = P(labelled | positive), the label frequency, by the method
    of Elkan and Noto: the mean predicted score on a held-out set of known
    positives.

    Requires a subset of labelled positives that was never used for training
    or threshold selection, otherwise c is optimistic.
    """
    s = np.asarray(scores_on_known_positives, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"c": float("nan")}
    return {
        "c_mean_score": float(s.mean()),
        "c_detection_rate": float((s > threshold).mean()),
        "n": int(s.size),
        "recommended": float(s.mean()),
    }


def pu_corrected_rates(precision_obs: float, recall_obs: float,
                       c: float) -> Dict:
    """
    Recovers true precision and recall from the values measured against an
    incomplete label.

    Under the selected-completely-at-random assumption, observed recall is
    true recall times c, so dividing recovers it. Precision is corrected by
    inflating the true-positive count by the same factor.

    That assumption is violated here, because the reference systematically
    masks woodland, urban land and rough grassland rather than dropping
    hedges at random. The correction should therefore be computed only within
    the unmasked domain, and the paper must say so.
    """
    if not np.isfinite(c) or c <= 0:
        return {"error": "invalid label frequency"}

    recall_true = min(recall_obs / c, 1.0)
    tp_scale = 1.0 / c
    precision_true = (precision_obs * tp_scale
                      / (precision_obs * tp_scale + (1.0 - precision_obs)))
    f1_true = (2 * precision_true * recall_true / (precision_true + recall_true)
               if (precision_true + recall_true) > 0 else 0.0)

    return {
        "label_frequency_c": float(c),
        "precision_observed": precision_obs, "recall_observed": recall_obs,
        "precision_corrected": float(min(precision_true, 1.0)),
        "recall_corrected": float(recall_true),
        "f1_corrected": float(f1_true),
        "caveat": ("Valid only inside the unmasked domain of the reference "
                   "product, since woodland, urban and rough grassland are "
                   "excluded by design rather than at random."),
    }
