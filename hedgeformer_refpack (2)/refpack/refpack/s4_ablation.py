"""
Section 4 of the work plan: ablating the loss.

Four components are claimed as contributions and none of them is currently
ablated. This module defines the run grid and, critically, implements the
textbook clDice so that the two corrections made in the paper can be shown to
matter rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# the configuration grid
# --------------------------------------------------------------------------
@dataclass
class LossConfig:
    """Mirrors the switches in main.py CFG. Defaults are the published setup."""
    w_mse: float = 0.5
    w_mae: float = 0.5
    w_cldice: float = 0.5
    w_merge: float = 0.5
    margin: float = 0.05
    skeleton_boost: float = 3.0
    pu_threshold: float = 0.85
    cldice_iters: int = 20
    cldice_textbook: bool = False      # restore the epsilon in the numerator
    cldice_include_empty: bool = False  # fold degenerate pairs into the mean
    name: str = "full"
    tests: str = ""


ABLATION_GRID: List[LossConfig] = [
    LossConfig(name="full",
               tests="reference configuration"),
    LossConfig(name="no_pu", pu_threshold=1.01,
               tests="the positive-unlabelled mask never fires, so confident "
                     "predictions in unlabelled regions are penalised again"),
    LossConfig(name="no_margin", margin=0.0,
               tests="tolerance to sub-margin annotation noise"),
    LossConfig(name="no_skeleton_boost", skeleton_boost=0.0,
               tests="spatial reweighting toward the exact centreline"),
    LossConfig(name="no_cldice", w_cldice=0.0,
               tests="the topological term as a whole"),
    LossConfig(name="textbook_cldice", cldice_textbook=True,
               cldice_include_empty=True,
               tests="the two corrections made to the standard clDice"),
    LossConfig(name="no_merge", w_merge=0.0,
               tests="the cross-class exclusivity constraint"),
]

PRIORITY_ABLATIONS = ["full", "no_pu", "textbook_cldice", "no_merge"]


def minimal_grid() -> List[LossConfig]:
    return [c for c in ABLATION_GRID if c.name in PRIORITY_ABLATIONS]


# --------------------------------------------------------------------------
# soft skeletonisation, matching the paper's formulation
# --------------------------------------------------------------------------
def soft_erode(x: torch.Tensor) -> torch.Tensor:
    a = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    b = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(a, b)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeleton(x: torch.Tensor, iters: int = 20) -> torch.Tensor:
    x1 = soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = soft_erode(x)
        x1 = soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


# --------------------------------------------------------------------------
# clDice, both the corrected and the textbook form
# --------------------------------------------------------------------------
def cldice_loss(pred: torch.Tensor, target: torch.Tensor,
                iters: int = 20, eps: float = 1e-6,
                textbook: bool = False, include_empty: bool = False
                ) -> Dict[str, torch.Tensor]:
    """
    pred, target: (B, C, H, W) in [0, 1].

    Two switches reproduce the failure the paper's version is designed to
    avoid.

    textbook=True puts the epsilon back in the numerator. A channel where both
    skeletons are empty then scores cl = 1, a perfect result for predicting
    nothing. Since most patches contain only a few of the seven classes, a
    model can lift its macro clDice simply by staying silent on absent
    classes. Removing the numerator epsilon makes a genuine miss score zero,
    which is what it should be.

    include_empty=True folds those degenerate pairs into the mean instead of
    dropping them, so classes that are legitimately absent from a patch start
    contributing a score.
    """
    skel_p = soft_skeleton(pred, iters)
    skel_t = soft_skeleton(target, iters)

    t_prec = (skel_p * target).sum(dim=(2, 3)) / (skel_p.sum(dim=(2, 3)) + eps)
    t_sens = (skel_t * pred).sum(dim=(2, 3)) / (skel_t.sum(dim=(2, 3)) + eps)

    num = 2.0 * t_prec * t_sens
    if textbook:
        num = num + eps
    cl = num / (t_prec + t_sens + eps)

    per_pair = 1.0 - cl                                  # (B, C)
    has_true = skel_t.sum(dim=(2, 3)) > 0                # (B, C)

    if include_empty:
        loss = per_pair.mean()
        n_used = per_pair.numel()
    else:
        if has_true.any():
            loss = per_pair[has_true].mean()
            n_used = int(has_true.sum())
        else:
            loss = per_pair.sum() * 0.0
            n_used = 0

    return {"loss": loss, "cl": cl, "per_pair": per_pair,
            "has_true": has_true, "n_pairs_used": n_used,
            "n_pairs_total": per_pair.numel()}


# --------------------------------------------------------------------------
# diagnostic: how much does the textbook form actually reward silence?
# --------------------------------------------------------------------------
@torch.no_grad()
def silence_exploit_probe(target: torch.Tensor, iters: int = 20) -> Dict[str, float]:
    """
    Quantifies the loophole directly, without training anything.

    Feeds an all-zero prediction against a real target and reports the clDice
    score under both formulations. Under the textbook form the score on absent
    classes is near perfect; under the corrected form it is near zero. This is
    a one-line result for the ablation table and it costs nothing to produce.
    """
    zeros = torch.zeros_like(target)
    tb = cldice_loss(zeros, target, iters, textbook=True, include_empty=True)
    ours = cldice_loss(zeros, target, iters, textbook=False, include_empty=False)

    absent = ~(target.sum(dim=(2, 3)) > 0)
    tb_absent = float(tb["cl"][absent].mean()) if absent.any() else float("nan")

    return {
        "textbook_macro_cl_for_silent_model": float(tb["cl"].mean()),
        "textbook_cl_on_absent_classes": tb_absent,
        "corrected_macro_cl_for_silent_model": float(ours["cl"].mean()),
        "fraction_of_class_slots_absent": float(absent.float().mean()),
        "reading": ("A silent model should score zero. Any gap between the "
                    "two columns is score the textbook formulation hands out "
                    "for predicting nothing."),
    }


# --------------------------------------------------------------------------
# margin-tolerant, PU-masked regression, with every switch exposed
# --------------------------------------------------------------------------
def pu_masked_regression(pred: torch.Tensor, target: torch.Tensor,
                         skeleton: torch.Tensor, valid: torch.Tensor,
                         cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """
    valid is the validity mask, that is 1 where the pixel is observed.
    Note that dataset.py returns the nodata mask, so pass 1 - mask here.
    """
    err = (pred - target).abs()
    err = torch.clamp(err - cfg.margin, min=0.0)

    weight = (1.0 + cfg.skeleton_boost * skeleton) * valid
    if cfg.pu_threshold <= 1.0:
        unlabelled_but_confident = (target < 0.01) & (pred > cfg.pu_threshold)
        weight = weight * (~unlabelled_but_confident).float()

    denom = weight.sum(dim=(0, 2, 3)).clamp(min=1.0)
    mse = (weight * err.pow(2)).sum(dim=(0, 2, 3)) / denom
    mae = (weight * err).sum(dim=(0, 2, 3)) / denom

    return {"mse_per_class": mse, "mae_per_class": mae,
            "mse": mse.mean(), "mae": mae.mean(),
            "pu_pixels_excluded": float(
                (weight == 0).float().sum() - (valid == 0).float().sum())}


def total_loss(pred: torch.Tensor, target: torch.Tensor, skeleton: torch.Tensor,
               valid: torch.Tensor, cfg: LossConfig) -> Dict[str, torch.Tensor]:
    reg = pu_masked_regression(pred, target, skeleton, valid, cfg)
    cld = cldice_loss(pred * valid, target * valid, cfg.cldice_iters,
                      textbook=cfg.cldice_textbook,
                      include_empty=cfg.cldice_include_empty)

    per_class = cfg.w_mse * reg["mse"] + cfg.w_mae * reg["mae"] \
        + cfg.w_cldice * cld["loss"]

    merged = torch.zeros((), device=pred.device)
    if cfg.w_merge > 0:
        pm = pred.sum(dim=1, keepdim=True).clamp(0, 1)
        tm = target.sum(dim=1, keepdim=True).clamp(0, 1)
        sm = skeleton.sum(dim=1, keepdim=True).clamp(0, 1)
        mreg = pu_masked_regression(pm, tm, sm, valid, cfg)
        mcl = cldice_loss(pm * valid, tm * valid, cfg.cldice_iters,
                          textbook=cfg.cldice_textbook,
                          include_empty=cfg.cldice_include_empty)
        merged = cfg.w_mse * mreg["mse"] + cfg.w_mae * mreg["mae"] \
            + cfg.w_cldice * mcl["loss"]

    return {"total": per_class + cfg.w_merge * merged,
            "per_class": per_class, "merged": merged,
            "mse": reg["mse"], "mae": reg["mae"], "cldice": cld["loss"],
            "pu_pixels_excluded": reg["pu_pixels_excluded"]}


def grid_as_table() -> List[Dict]:
    return [{**asdict(c)} for c in ABLATION_GRID]
