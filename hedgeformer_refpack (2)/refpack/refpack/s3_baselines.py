"""
Section 3 of the work plan: the baselines the manuscript is missing.

The current comparison is internal only, three variants of one architecture.
This module adds the four learned baselines and, more importantly for a remote
sensing readership, a classical baseline that uses no learning at all.

The learned baselines deliberately reuse HedgeFormer's decoder, head and loss
so that the only thing that changes is how, or whether, the two modalities
meet.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASS_LO_M, N_CLASSES, PIXEL_SIZE_M


# ==========================================================================
# 3a / 3b / 3c   learned baselines
# ==========================================================================
class SingleModalityHedgeFormer(nn.Module):
    """
    One encoder, no fusion. Answers the first question any reviewer asks:
    was fusion necessary at all?

    Expect a strong asymmetry across classes. The elevation-only model should
    hold up in classes 4 and 6 and collapse in 1a, the spectral-only model the
    other way round. That asymmetry, plotted per class, is a more convincing
    argument for fusion than any aggregate score.
    """

    def __init__(self, modality: str = "ms", num_class: int = N_CLASSES,
                 decoder_dim: int = 128):
        super().__init__()
        import segmentation_models_pytorch as smp
        from model import SegFormerDecoder, _Head

        assert modality in ("ms", "dem")
        self.modality = modality
        in_ch = 6 if modality == "ms" else 1
        self.encoder = smp.encoders.get_encoder(
            "mit_b5", in_channels=in_ch, depth=4, weights="imagenet")
        chans = [c for c in self.encoder.out_channels if c > in_ch]
        self.decoder = SegFormerDecoder(chans, decoder_dim)
        self.seg_head = _Head(decoder_dim, num_class)
        self._in_ch = in_ch

    def forward(self, dtm: torch.Tensor, ms: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = ms if self.modality == "ms" else dtm
        feats = [f for f in self.encoder(x) if f.shape[1] > self._in_ch]
        out = self.decoder(feats[:3])
        out = F.interpolate(out, size=dtm.shape[-2:], mode="bilinear",
                            align_corners=False)
        return {"seg_logits": self.seg_head(out)}


class EarlyFusionHedgeFormer(nn.Module):
    """
    Seven input channels stacked at the input, one encoder.

    This is the cheapest possible fusion and it is the baseline the paper has
    to beat before any claim about attention is worth making. It also has
    roughly half the encoder parameters of the dual-stream models, so if it
    comes close, the honest reading is that the second encoder is doing the
    work rather than the fusion operator.
    """

    def __init__(self, num_class: int = N_CLASSES, decoder_dim: int = 128):
        super().__init__()
        import segmentation_models_pytorch as smp
        from model import SegFormerDecoder, _Head
        self.encoder = smp.encoders.get_encoder(
            "mit_b5", in_channels=7, depth=4, weights="imagenet")
        chans = [c for c in self.encoder.out_channels if c > 7]
        self.decoder = SegFormerDecoder(chans, decoder_dim)
        self.seg_head = _Head(decoder_dim, num_class)

    def forward(self, dtm, ms):
        x = torch.cat([ms, dtm], dim=1)
        feats = [f for f in self.encoder(x) if f.shape[1] > 7]
        out = self.decoder(feats[:3])
        out = F.interpolate(out, size=dtm.shape[-2:], mode="bilinear",
                            align_corners=False)
        return {"seg_logits": self.seg_head(out)}


class LateFusionHedgeFormer(nn.Module):
    """
    Two encoders that never interact, logits averaged at the very end.

    This separates two effects the paper currently conflates: how much comes
    from having two specialised streams, and how much comes from letting them
    exchange information at every scale. If late fusion matches the
    concatenation variant, then cross-scale interaction is what matters; if it
    does not, capacity is.
    """

    def __init__(self, num_class: int = N_CLASSES, decoder_dim: int = 128,
                 mode: str = "mean"):
        super().__init__()
        self.ms_branch = SingleModalityHedgeFormer("ms", num_class, decoder_dim)
        self.dem_branch = SingleModalityHedgeFormer("dem", num_class, decoder_dim)
        assert mode in ("mean", "learned")
        self.mode = mode
        if mode == "learned":
            self.alpha = nn.Parameter(torch.zeros(1, num_class, 1, 1))

    def forward(self, dtm, ms):
        a = self.ms_branch(dtm, ms)["seg_logits"]
        b = self.dem_branch(dtm, ms)["seg_logits"]
        if self.mode == "mean":
            return {"seg_logits": 0.5 * (a + b)}
        w = torch.sigmoid(self.alpha)
        return {"seg_logits": w * a + (1.0 - w) * b}


def build_baseline(name: str, num_class: int = N_CLASSES, **kw) -> nn.Module:
    """Single entry point so the training harness can swap any model in."""
    from model import HedgeFormer
    name = name.lower()
    if name in ("ttt", "sra", "simple"):
        return HedgeFormer(num_class=num_class,
                           attention_type={"ttt": "TTT", "sra": "SRA",
                                           "simple": "Simple"}[name], **kw)
    if name == "ms_only":
        return SingleModalityHedgeFormer("ms", num_class)
    if name == "dem_only":
        return SingleModalityHedgeFormer("dem", num_class)
    if name == "early":
        return EarlyFusionHedgeFormer(num_class)
    if name == "late":
        return LateFusionHedgeFormer(num_class)
    raise ValueError(f"unknown baseline {name!r}")


BASELINE_SET = ["ms_only", "dem_only", "early", "late", "simple", "sra", "ttt"]


# ==========================================================================
# 3e   the classical baseline, no learning at all
# ==========================================================================
def _pca_elongation(ys: np.ndarray, xs: np.ndarray):
    """Ratio of principal axis lengths, and the minor-axis extent in pixels.
    A hedge is long and thin; a wood patch is not."""
    if ys.size < 8:
        return 0.0, 0.0, 0.0
    pts = np.stack([ys.astype(float), xs.astype(float)], axis=1)
    pts = pts - pts.mean(0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-9, None)
    major, minor = float(np.sqrt(ev[1])), float(np.sqrt(ev[0]))
    angle = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    return major / minor, 4.0 * minor, float(np.degrees(angle))


def classical_hedgerow_baseline(ms: np.ndarray, chm: np.ndarray,
                                band_index: Dict[str, int] | None = None,
                                ndvi_thr: float = 0.35,
                                ndre_thr: float = 0.15,
                                height_thr: float = 0.5,
                                min_elongation: float = 3.0,
                                max_width_px: float = 8.0,
                                min_pixels: int = 40) -> Dict[str, np.ndarray]:
    """
    A spectral index plus a height threshold plus a linearity test, which is
    how this problem was solved before deep learning.

    The point of including it is not to win. It is to show, with a number,
    exactly where the classical approach breaks. The prediction is that it
    stays competitive in classes 4 and 6, where the canopy clears the noise
    floor of the elevation surface, and falls apart in 1a and 1b, where it
    cannot separate a low hedge from a wall or an unmown margin. If that is
    what happens, the physical argument in the introduction stops being an
    assertion and becomes a result.

    ms  : (6, H, W) reflectance or DN, Pleiades Neo band order
          0 deep blue, 1 blue, 2 green, 3 red, 4 red edge, 5 near infrared
    chm : (H, W) canopy height in metres
    """
    from scipy import ndimage as ndi

    bi = band_index or {"red": 3, "red_edge": 4, "nir": 5}
    red = ms[bi["red"]].astype(np.float32)
    rededge = ms[bi["red_edge"]].astype(np.float32)
    nir = ms[bi["nir"]].astype(np.float32)

    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)

    woody = (ndvi > ndvi_thr) & (ndre > ndre_thr)
    tall = chm > height_thr
    candidate = woody & tall

    # remove speckle, then close small gaps along the line
    candidate = ndi.binary_opening(candidate, structure=np.ones((3, 3)))
    candidate = ndi.binary_closing(candidate, structure=np.ones((3, 3)))

    labels, n = ndi.label(candidate, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, dtype=bool)
    diag = {}
    for comp_id in range(1, n + 1):
        ys, xs = np.where(labels == comp_id)
        if ys.size < min_pixels:
            continue
        elong, width, angle = _pca_elongation(ys, xs)
        if elong >= min_elongation and width <= max_width_px:
            keep[comp_id] = True
            diag[comp_id] = {"elongation": elong, "width_px": width,
                             "angle_deg": angle, "n_px": int(ys.size)}

    linear_mask = keep[labels]

    # height classes come straight from the CHM, using the UKCEH breakpoints
    class_map = np.zeros_like(chm, dtype=np.uint8)
    edges = np.array(CLASS_LO_M, dtype=np.float32)
    idx = np.digitize(chm, edges, right=False)      # 0 below 0.5 m, 7 above 6 m
    class_map[linear_mask] = np.clip(idx[linear_mask], 0, N_CLASSES).astype(np.uint8)

    return {
        "binary": linear_mask.astype(np.uint8),
        "class_map": class_map,
        "ndvi": ndvi, "ndre": ndre,
        "n_components_kept": int(keep.sum()),
        "n_components_total": int(n),
        "component_diagnostics": diag,
    }


def classical_to_proximity_field(class_map: np.ndarray,
                                 dist_cap_m: float = 10.0) -> np.ndarray:
    """
    Converts the classical hard output into the same bounded proximity field
    the network predicts, so that every metric in the evaluation protocol can
    be applied to it without modification. Without this the classical baseline
    could only be scored on a subset of the metrics and the comparison would
    be incomplete.
    """
    from scipy import ndimage as ndi
    out = np.zeros((N_CLASSES,) + class_map.shape, dtype=np.float32)
    for c in range(1, N_CLASSES + 1):
        m = class_map == c
        if not m.any():
            continue
        d = ndi.distance_transform_edt(~m) * PIXEL_SIZE_M
        field = np.clip((dist_cap_m - d) / dist_cap_m, 0.0, 1.0)
        # restrict to the buffer around this class, matching target construction
        field[d > dist_cap_m] = 0.0
        out[c - 1] = field
    return out


def tune_classical_thresholds(patches: List[dict], grid: Dict[str, List[float]],
                              score_fn) -> Dict:
    """
    The classical baseline must be given the same chance to be tuned as the
    networks were. Grid search on the validation split only, then frozen.
    Anything less and a reviewer will say the comparison was rigged.
    """
    best = {"score": -np.inf}
    for ndvi_thr in grid.get("ndvi_thr", [0.25, 0.35, 0.45]):
        for ndre_thr in grid.get("ndre_thr", [0.05, 0.15, 0.25]):
            for elong in grid.get("min_elongation", [2.0, 3.0, 4.0]):
                scores = []
                for p in patches:
                    out = classical_hedgerow_baseline(
                        p["ms"], p["chm"], ndvi_thr=ndvi_thr,
                        ndre_thr=ndre_thr, min_elongation=elong)
                    scores.append(score_fn(out["class_map"], p["target"]))
                s = float(np.mean(scores))
                if s > best["score"]:
                    best = {"score": s, "ndvi_thr": ndvi_thr,
                            "ndre_thr": ndre_thr, "min_elongation": elong}
    return best
