"""
Section 5 of the work plan: show the mechanism, do not infer it from a metric.

At present the paper argues that test-time adaptation works because the score
improves in class 1a. That is circumstantial. These three experiments measure
the mechanism directly.

  5.1  Fast-weight probe. How far do the fast weights actually travel, and
       does that distance track the physical properties of the scene?
  5.2  Modality ablation at inference. Which stream is each operator relying
       on, per height class, with no retraining.
  5.3  Synthetic phenology stress. Degrade the canopy signal the way a
       leaf-off survey does and watch each operator degrade with it.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASS_CODE, N_CLASSES


# ==========================================================================
# 5.1  fast-weight probe
# ==========================================================================
@dataclass
class ProbeRecord:
    scale: int
    direction: str
    d_linear: float
    d_spatial: float
    w1_norm: float
    w2_norm: float
    w3_norm: float


class FastWeightProbe:
    """
    Instruments TTTCrossAttn without editing model.py.

    The module computes the updated fast weights inside forward, so the
    quantity we want is not a parameter and not an activation and cannot be
    read with a standard forward hook. Instead we wrap the two inner-update
    methods, capture their return values, and compare them with the stored
    slow weights they started from.

    Direction is recovered from call ordering: BidirectionTTTCrossAttn calls
    the shared TTT module twice per fusion stage, first with the multispectral
    stream as query and second with the elevation stream as query.
    """

    DIRECTIONS = ["ms_query", "dem_query"]

    def __init__(self, model):
        self.model = model
        self.records: List[ProbeRecord] = []
        self._orig: Dict = {}
        self._call_idx = 0
        self._modules = self._find_ttt_modules()

    def _find_ttt_modules(self):
        mods = []
        for i, fuse in enumerate(getattr(self.model, "fuse_modules", [])):
            ca = getattr(fuse, "cross_attn", None)
            ttt = getattr(ca, "ttt", None) if ca is not None else None
            if ttt is not None:
                mods.append((i + 1, ttt))
        return mods

    @property
    def available(self) -> bool:
        return len(self._modules) > 0

    @contextmanager
    def capture(self):
        if not self.available:
            yield self
            return
        self._install()
        try:
            yield self
        finally:
            self._remove()

    def _install(self):
        for scale, ttt in self._modules:
            self._orig[(scale, "lin")] = ttt.inner_train_simplified_swiglu
            self._orig[(scale, "dwc")] = ttt.inner_train_3x3dwc

            def make_lin(ttt_ref, scale_ref, orig):
                def wrapped(k, v, w1, w2, lr=1.0):
                    nw1, nw2 = orig(k, v, w1, w2, lr)
                    d = (nw1 - w1).norm().item() + (nw2 - w2).norm().item()
                    ttt_ref._probe_lin = (d, w1.norm().item(), w2.norm().item())
                    return nw1, nw2
                return wrapped

            def make_dwc(ttt_ref, scale_ref, orig):
                def wrapped(k, v, w, lr=1.0, implementation="prod"):
                    nw = orig(k, v, w, lr, implementation)
                    B = nw.shape[0] // max(w.shape[0], 1)
                    base = w.repeat(max(B, 1), 1, 1, 1) if nw.shape != w.shape else w
                    d = (nw - base).norm().item()
                    ttt_ref._probe_dwc = (d, w.norm().item())
                    self._flush(scale_ref, ttt_ref)
                    return nw
                return wrapped

            ttt.inner_train_simplified_swiglu = make_lin(ttt, scale, self._orig[(scale, "lin")])
            ttt.inner_train_3x3dwc = make_dwc(ttt, scale, self._orig[(scale, "dwc")])

    def _flush(self, scale, ttt):
        lin = getattr(ttt, "_probe_lin", None)
        dwc = getattr(ttt, "_probe_dwc", None)
        if lin is None or dwc is None:
            return
        direction = self.DIRECTIONS[self._call_idx % 2]
        self._call_idx += 1
        self.records.append(ProbeRecord(
            scale=scale, direction=direction,
            d_linear=lin[0], d_spatial=dwc[0],
            w1_norm=lin[1], w2_norm=lin[2], w3_norm=dwc[1]))
        ttt._probe_lin = None
        ttt._probe_dwc = None

    def _remove(self):
        for scale, ttt in self._modules:
            ttt.inner_train_simplified_swiglu = self._orig[(scale, "lin")]
            ttt.inner_train_3x3dwc = self._orig[(scale, "dwc")]
        self._orig.clear()

    def reset(self):
        self.records.clear()
        self._call_idx = 0

    def summary(self) -> Dict[str, float]:
        """
        Collapses one forward pass into the quantity of interest: how much of
        the total fast-weight movement is spent on the direction that queries
        with the elevation stream.

        If the operator is doing what the paper claims, this share should fall
        in scenes where the elevation cue is uninformative.
        """
        if not self.records:
            return {}
        by_dir = {d: 0.0 for d in self.DIRECTIONS}
        for r in self.records:
            by_dir[r.direction] += r.d_linear + r.d_spatial
        total = sum(by_dir.values()) + 1e-12
        out = {f"movement_{k}": v for k, v in by_dir.items()}
        out["dem_query_share"] = by_dir["dem_query"] / total
        out["total_movement"] = total
        for r in self.records:
            out[f"s{r.scale}_{r.direction}_linear"] = r.d_linear
            out[f"s{r.scale}_{r.direction}_spatial"] = r.d_spatial
        return out


def scene_covariates(dtm: torch.Tensor, ms: torch.Tensor,
                     target: torch.Tensor,
                     nir_band: int = 5, red_band: int = 3) -> Dict[str, float]:
    """
    The physical properties the fast-weight movement is regressed against.

    chm_snr is the quantity that matters most: the canopy signal on labelled
    hedge pixels divided by the local roughness of the surrounding surface.
    Where it approaches one, the elevation stream carries no usable
    information and an adaptive operator should be seen to discount it.
    """
    d = dtm[0, 0].detach().float()
    t = target[0].detach().float()
    hedge = t.sum(0) > 0.5

    if hedge.sum() < 10:
        return {"n_hedge_px": float(hedge.sum()), "valid": 0.0}

    hedge_h = d[hedge]
    k = 9
    local_mean = F.avg_pool2d(d[None, None], k, 1, k // 2)[0, 0]
    rough = (d - local_mean).abs()
    background_rough = rough[~hedge].mean().clamp(min=1e-6)

    red = ms[0, red_band].detach().float()
    nir = ms[0, nir_band].detach().float()
    ndvi = (nir - red) / (nir + red + 1e-6)

    gy, gx = torch.gradient(d)
    slope = torch.hypot(gx, gy)

    return {
        "valid": 1.0,
        "n_hedge_px": float(hedge.sum()),
        "median_height": float(hedge_h.median()),
        "mean_height": float(hedge_h.mean()),
        "chm_snr": float(hedge_h.abs().mean() / background_rough),
        "background_roughness": float(background_rough),
        "canopy_ndvi": float(ndvi[hedge].mean()),
        "terrain_slope": float(slope.mean()),
        "dominant_class": int(t.sum(dim=(1, 2)).argmax().item()) + 1,
    }


@torch.no_grad()
def run_fast_weight_probe(model, loader, device, max_batches: int = 500) -> List[Dict]:
    """
    Produces the signature figure of the paper: fast-weight movement against
    scene physics.

    Run the same loop on the SRA variant for contrast. Its projections are
    frozen, so the equivalent quantity should be flat. The contrast between
    the two scatter plots is the visual proof of the mechanism.
    """
    model.eval()
    probe = FastWeightProbe(model)
    if not probe.available:
        raise RuntimeError("no TTT fusion modules found; probe applies to the "
                           "TTT variant only")
    rows = []
    with probe.capture():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ms, dem, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            probe.reset()
            model(dem, ms)
            rec = probe.summary()
            cov = scene_covariates(dem, ms, target)
            if cov.get("valid", 0.0) > 0:
                rows.append({**rec, **cov})
    return rows


def analyse_probe(rows: List[Dict]) -> Dict:
    """Correlation between fast-weight movement and the physical covariates."""
    if not rows:
        return {}
    import numpy as np
    out = {}
    y = np.array([r["dem_query_share"] for r in rows])
    for key in ("median_height", "chm_snr", "canopy_ndvi", "terrain_slope"):
        x = np.array([r.get(key, np.nan) for r in rows])
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 10:
            continue
        r = float(np.corrcoef(x[keep], y[keep])[0, 1])
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(x[keep], y[keep])
        except Exception:
            rho, p = float("nan"), float("nan")
        out[key] = {"pearson_r": r, "spearman_rho": float(rho),
                    "p_value": float(p), "n": int(keep.sum())}
    per_class = {}
    for c in range(1, N_CLASSES + 1):
        vals = [r["dem_query_share"] for r in rows if r.get("dominant_class") == c]
        if vals:
            per_class[CLASS_CODE[c - 1]] = {"mean_dem_query_share": float(np.mean(vals)),
                                            "n": len(vals)}
    out["by_dominant_class"] = per_class
    out["prediction_under_test"] = (
        "dem_query_share should rise with median_height and with chm_snr. "
        "A flat relationship falsifies the scene-adaptive claim.")
    return out


# ==========================================================================
# 5.2  modality ablation at inference
# ==========================================================================
PERTURBATIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "clean": lambda x: x,
    "dem_zero": lambda x: torch.zeros_like(x),
    "dem_noise_1m": lambda x: x + torch.randn_like(x) * 1.0,
    "dem_noise_3m": lambda x: x + torch.randn_like(x) * 3.0,
    "dem_smooth": lambda x: F.avg_pool2d(x, 7, 1, 3),
    "dem_shuffle": lambda x: x.flatten(2)[
        :, :, torch.randperm(x.shape[-1] * x.shape[-2], device=x.device)
    ].view_as(x),
}

MS_PERTURBATIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "clean": lambda x: x,
    "ms_zero": lambda x: torch.zeros_like(x),
    "ms_noise": lambda x: x + torch.randn_like(x) * 1.0,
    "ms_grey": lambda x: x.mean(dim=1, keepdim=True).expand_as(x).clone(),
}


@torch.no_grad()
def modality_ablation(model, loader, device, metric_fn,
                      perturbations: Optional[Dict] = None,
                      target_stream: str = "dem",
                      seed: int = 0) -> Dict[str, Dict]:
    """
    metric_fn(pred_probs, target, valid) must return a dict of per-class
    scores, so that the result can be reported per height class. The
    cross-class pattern is the whole point; a macro number would hide it.

    The prediction being tested: the TTT operator should lose least in class
    1a when the elevation stream is destroyed, because it has already learned
    not to lean on elevation in that regime, and lose most in class 6, where
    it genuinely uses it. A frozen operator should show a flatter, more
    uniform degradation.
    """
    torch.manual_seed(seed)
    perturbations = perturbations or (
        PERTURBATIONS if target_stream == "dem" else MS_PERTURBATIONS)
    model.eval()
    acc = {name: [] for name in perturbations}

    for batch in loader:
        ms, dem, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        valid = 1.0 - batch[3].to(device)
        for name, fn in perturbations.items():
            d, m = (fn(dem), ms) if target_stream == "dem" else (dem, fn(ms))
            probs = torch.sigmoid(model(d, m)["seg_logits"])
            acc[name].append(metric_fn(probs, target, valid))

    out = {}
    base = None
    for name, vals in acc.items():
        keys = vals[0].keys()
        mean = {k: float(np.mean([v[k] for v in vals])) for k in keys}
        out[name] = mean
        if name == "clean":
            base = mean
    if base:
        for name, mean in out.items():
            if name == "clean":
                continue
            out[name] = {
                **mean,
                **{f"drop_{k}": base[k] - mean[k] for k in mean},
                **{f"rel_drop_{k}": (base[k] - mean[k]) / abs(base[k])
                   if base[k] else float("nan") for k in mean},
            }
    return out


# ==========================================================================
# 5.3  synthetic phenology stress
# ==========================================================================
def phenology_stress(dem_norm: torch.Tensor, alpha: float,
                     ground_percentile: float = 5.0) -> torch.Tensor:
    """
    Attenuates the canopy component of the elevation layer by alpha.

    alpha = 1 leaves the scene as acquired. alpha = 0 removes the canopy
    entirely, which is the limiting case of a fully transmissive leaf-off
    survey. Intermediate values interpolate the partial transmission that a
    real winter flight produces over a deciduous hedge.

    The input is the normalised tensor the network sees, so ground is
    estimated per patch from a low percentile rather than assumed to be zero.
    """
    B = dem_norm.shape[0]
    flat = dem_norm.view(B, -1)
    ground = torch.quantile(flat, ground_percentile / 100.0, dim=1)
    ground = ground.view(B, 1, 1, 1)
    canopy = dem_norm - ground
    return ground + canopy * alpha


@torch.no_grad()
def phenology_curve(model, loader, device, metric_fn,
                    alphas=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0)) -> Dict[float, Dict]:
    """
    The degradation curve. A shallower slope for the adaptive operator is the
    direct measurement of the claim made in the introduction, which is
    currently supported only by argument.
    """
    model.eval()
    curve = {}
    for alpha in alphas:
        vals = []
        for batch in loader:
            ms, dem, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            valid = 1.0 - batch[3].to(device)
            probs = torch.sigmoid(model(phenology_stress(dem, alpha), ms)["seg_logits"])
            vals.append(metric_fn(probs, target, valid))
        keys = vals[0].keys()
        curve[alpha] = {k: float(np.mean([v[k] for v in vals])) for k in keys}
    return curve


def curve_slope(curve: Dict[float, Dict], key: str) -> float:
    """Least-squares slope of the degradation curve; less negative is more
    robust to the phenological mismatch."""
    a = np.array(sorted(curve))
    y = np.array([curve[x][key] for x in a])
    return float(np.polyfit(a, y, 1)[0])


def compare_robustness(curves: Dict[str, Dict[float, Dict]], key: str) -> Dict:
    return {
        variant: {"slope": curve_slope(c, key),
                  "value_at_alpha_0": c[min(c)][key],
                  "value_at_alpha_1": c[max(c)][key],
                  "retained_fraction": (c[min(c)][key] / c[max(c)][key]
                                        if c[max(c)][key] else float("nan"))}
        for variant, c in curves.items()
    }
