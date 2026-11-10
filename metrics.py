"""
metrics.py  --  single source of truth for every metric / differentiable skeleton op used by
main.py (training), validation.py (evaluation) and analysis.py (extra studies).

Previously the same accumulator code lived in three places (main.RegressionMetrics,
validation.FullMetrics, validation.FastMetrics).  Everything is now defined once, here.

DEFINITIONS (all pooled over the evaluation set unless stated otherwise)
------------------------------------------------------------------------
pred_bin = sigmoid(logit) >= thresh          gt_bin = target > gt_thresh (0.01)
Both are multiplied by the validity mask (nodata pixels never count).
tolerance is a CHEBYSHEV radius (square structuring element of side 2*tol+1).

Relaxed precision   P = |pred_bin  ∩ dilate(gt_bin , tol)| / |pred_bin|
Relaxed recall      R = |gt_bin    ∩ dilate(pred_bin, tol)| / |gt_bin|
Relaxed F1          2PR/(P+R)
Relaxed IoU         "quality" Q = PR/(P+R-PR)  (== TP/(TP+FP+FN) when TP is counted consistently)
   NOTE: the old code used the *prediction-side* match count in BOTH precision and recall
   (recall = tp_pred/(tp_pred+fn_gt)), mixing units of two different pixel sets.  Fixed.

BF (boundary F1, tolerance bf2_tolerance): same P/R/F1 but computed on the 1-px BOUNDARIES
   (mask minus its 3x3 erosion), which is the actual Csurka et al. definition.  The old code
   ran it on the full masks (i.e. it was just a second relaxed-F1 with a smaller tolerance).

clDice  (Shit et al. 2021)   Tprec = |S_P ∩ V_L| / |S_P|,   Tsens = |S_L ∩ V_P| / |S_L|,
   clDice = 2*Tprec*Tsens/(Tprec+Tsens); S_* = skeleton (soft_skeletonize on the binary mask,
   the SAME operator the training loss uses), V_* = binary mask (P = prediction, L = label).

Connectivity: 8-connected components; correctness = matched_pred/n_pred, completeness =
   matched_gt/n_gt (matched = shares >=1 pixel with the other mask), connectivity = F1 of the two.
topo_lc = mean over patches with >=1 component of  1 - |n_pred-n_gt|/max(n_pred,n_gt).
   (Patches with no component in both masks are EXCLUDED: previously they scored a perfect 1.0,
   which inflated per-class values for classes absent from most patches.)

Chamfer  = 0.5*(mean_{p in S_P} d(p,C_L) + mean_{c in C_L} d(c,S_P))   [px]
HD95     = max(P95_{p in S_P} d(p,C_L), P95_{c in C_L} d(c,S_P))        [px]
   S_P = skeleton of the predicted mask, C_L = exact rasterised label line (dataset `skeleton`).
   Every skeleton pixel is used (no raster-order sub-sampling).  Frechet was REMOVED: it needs an
   ordered polyline and unordered pixel sets have no meaningful order.
curve_coverage = n_patches_with_both_curves / n_patches_with_gt_curve.  Chamfer/HD95 are only
   defined on those patches, so ALWAYS read them together with coverage (a model that predicts
   nothing gets no chamfer at all, it does not get a "good" one).

Macro = unweighted mean over classes that have ground-truth support (gt_px > 0) in the
   evaluated set; classes absent from the set no longer enter the mean as F1 = 0.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree

_CONN8 = np.ones((3, 3), dtype=int)

KULPA_AXIS = 0.948   # corrected length of a horizontal/vertical chain step
KULPA_DIAG = 1.343   # corrected length of a diagonal chain step


# ----------------------------------------------------------------------------------------
#  differentiable soft morphology (clDice, Shit et al. CVPR 2021) -- identical to reference
# ----------------------------------------------------------------------------------------
def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.shape[2] <= 2 or img.shape[3] <= 2:
        return img
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 20) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    skel = F.relu(img - soft_open(img))
    x = img.clone()
    for _ in range(int(iters)):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return torch.clamp(skel, 0.0, 1.0)


def soft_cldice_loss(pred: torch.Tensor, target: torch.Tensor, iters: int = 20, eps: float = 1e-6,
                     mode: str = 'fixed', valid: torch.Tensor = None) -> torch.Tensor:
    """
    Probability-space clDice loss on (B, C, H, W) tensors.

      Tprec = sum(S_pred * target) / sum(S_pred)        Tsens = sum(S_true * pred) / sum(S_true)
      clDice = 2 Tprec Tsens / (Tprec + Tsens)          loss = 1 - clDice

    mode='fixed'         : no epsilon in the numerator (0/0 -> clDice 0, i.e. loss 1, when the model
                           predicts nothing for a class that IS present) and (B,C) entries whose
                           target has no skeleton are excluded from the mean.
    mode='conventional'  : textbook form (2*Tprec*Tsens + eps)/(Tprec + Tsens + eps), averaged over
                           ALL entries (used only for the ablation that shows why the fix matters).
    valid (B,1,H,W)      : nodata mask; pixels with valid==0 are removed from BOTH pred and target so
                           clDice sees the same pixels as the MSE/MAE terms.
    Always fp32 regardless of an enclosing autocast.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred = pred.float()
        target = torch.clamp(target.float(), 0.0, 1.0)
        if valid is not None:
            v = valid.float()
            pred = pred * v
            target = target * v
        skel_pred = soft_skeletonize(pred, iters=iters)
        skel_true = soft_skeletonize(target, iters=iters)

        true_sum = skel_true.sum(dim=(2, 3))
        tprec = (skel_pred * target).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + eps)
        tsens = (skel_true * pred).sum(dim=(2, 3)) / (true_sum + eps)

        if mode == 'fixed':
            cl = (2.0 * tprec * tsens) / (tprec + tsens + eps)
            has_target = (true_sum > 0).float()
            return ((1.0 - cl) * has_target).sum() / has_target.sum().clamp(min=1.0)
        if mode == 'conventional':
            cl = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
            return (1.0 - cl).mean()
        raise ValueError(f"unknown clDice mode '{mode}'")


# ----------------------------------------------------------------------------------------
#  Kulpa digital-length estimator (chain-code step counting with corrected step lengths)
# ----------------------------------------------------------------------------------------
def _diag_edges(line: np.ndarray):
    """Non-redundant diagonal neighbour pairs of a boolean array (...,H,W).
    A diagonal pair is dropped when a shared 4-neighbour is also set (the path already goes
    through that 4-neighbour, so counting the diagonal too would double-count the step)."""
    a = line[..., :-1, :-1]; b = line[..., 1:, 1:]      # '\' pair
    c = line[..., :-1, 1:];  d = line[..., 1:, :-1]     # '/' pair
    e_bs = a & b & ~c & ~d
    e_fs = c & d & ~a & ~b
    return e_bs, e_fs


def kulpa_length_px(line: np.ndarray) -> float:
    """Length (pixels) of a thin (skeleton-like) boolean line image: 0.948*n_axis + 1.343*n_diag."""
    line = np.asarray(line).astype(bool)
    n_axis = (line[..., :, :-1] & line[..., :, 1:]).sum() + (line[..., :-1, :] & line[..., 1:, :]).sum()
    e_bs, e_fs = _diag_edges(line)
    n_diag = e_bs.sum() + e_fs.sum()
    return float(KULPA_AXIS * n_axis + KULPA_DIAG * n_diag)


def kulpa_pixel_weights(line: np.ndarray) -> np.ndarray:
    """Per-pixel length attribution: every kept edge gives half its corrected length to each end pixel,
    so weights.sum() == kulpa_length_px(line) exactly."""
    line = np.asarray(line).astype(bool)
    w = np.zeros(line.shape, dtype=np.float64)
    h = (line[..., :, :-1] & line[..., :, 1:]).astype(np.float64) * (0.5 * KULPA_AXIS)
    v = (line[..., :-1, :] & line[..., 1:, :]).astype(np.float64) * (0.5 * KULPA_AXIS)
    w[..., :, :-1] += h; w[..., :, 1:] += h
    w[..., :-1, :] += v; w[..., 1:, :] += v
    e_bs, e_fs = _diag_edges(line)
    a = e_bs.astype(np.float64) * (0.5 * KULPA_DIAG)
    b = e_fs.astype(np.float64) * (0.5 * KULPA_DIAG)
    w[..., :-1, :-1] += a; w[..., 1:, 1:] += a
    w[..., :-1, 1:] += b;  w[..., 1:, :-1] += b
    return w


# ----------------------------------------------------------------------------------------
#  segmentation metric engine
# ----------------------------------------------------------------------------------------
BASE_KEYS = [
    'active_mse', 'active_mae', 'active_rmse',
    'relaxed_precision', 'relaxed_recall', 'relaxed_f1', 'relaxed_iou',
    'bf2_precision', 'bf2_recall', 'bf2',
    'cldice',
]
TOPO_KEYS = [
    'connectivity_correctness', 'connectivity_completeness', 'connectivity', 'topo_lc',
    'chamfer_distance', 'hausdorff95_distance', 'curve_coverage',
    'chamfer_distance_matched', 'hausdorff95_distance_matched',
]
SUPPORT_KEYS = ['gt_px', 'pred_px', 'gt_line_px', 'n_patches_gt']
METRIC_KEYS = BASE_KEYS + TOPO_KEYS

_ACC_KEYS = [
    'mse', 'mae', 'nact',
    'p_num', 'p_den', 'r_num', 'r_den',
    'bp_num', 'bp_den', 'br_num', 'br_den',
    'cl_pnum', 'cl_pden', 'cl_snum', 'cl_sden',
    'conn_pm', 'conn_pn', 'conn_gm', 'conn_gn', 'topo_sum', 'topo_n',
    'ch_sum', 'hd_sum', 'n_curve', 'n_gtpatch', 'chm_sum', 'hdm_sum', 'n_curve_m',
    'gt_px', 'pred_px', 'gt_line_px',
]


def _div(a, b, default=0.0):
    return a / b if b > 0 else default


def _f1(p, r):
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def _quality(p, r):
    d = p + r - p * r
    return p * r / d if d > 0 else 0.0


def _curve_distances(pts_a: np.ndarray, pts_b: np.ndarray):
    """Directed distances a->b and b->a (both non-empty (n,2) arrays)."""
    d_ab, _ = cKDTree(pts_b).query(pts_a)
    d_ba, _ = cKDTree(pts_a).query(pts_b)
    return d_ab, d_ba


def chamfer_hd95(pts_a: np.ndarray, pts_b: np.ndarray):
    d_ab, d_ba = _curve_distances(pts_a, pts_b)
    chamfer = 0.5 * (float(d_ab.mean()) + float(d_ba.mean()))
    hd95 = max(float(np.percentile(d_ab, 95)), float(np.percentile(d_ba, 95)))
    return chamfer, hd95


class SegMetrics:
    """
    Accumulates every metric for the N per-height-class channels plus the 'merged' group
    (channels summed and clipped to 1 = the scenario-1 "is there a hedge at all" map).
    Group index g = 0..N-1 -> class_{g+1}, g = N -> merged.

    update(logits, target, skeleton, valid, names)  with tensors (B, N, H, W) / valid (B, 1, H, W).
    compute() -> {'merged': {...}, 'macro': {...}, 'class_1': {...}, ...}
    per_patch() -> {'merged'|'class_i': {metric: np.array(P)}}, names  (needs keep_per_patch=True)
    """

    def __init__(self, num_classes: int, thresh: float = 0.3, tolerance: int = 3, bf2_tolerance: int = 2,
                 gt_thresh: float = 0.01, topology: bool = True, keep_per_patch: bool = False,
                 skel_iters: int = 20):
        self.N = num_classes
        self.G = num_classes + 1
        self.thresh = thresh
        self.tolerance = tolerance
        self.bf2_tolerance = bf2_tolerance
        self.gt_thresh = gt_thresh
        self.topology = topology
        self.keep_per_patch = keep_per_patch
        self.skel_iters = skel_iters
        self.reset()

    def reset(self):
        self.acc = {k: np.zeros(self.G, dtype=np.float64) for k in _ACC_KEYS}
        self.n_patches = 0
        self._pp = []
        self._names = []

    # -- helpers -----------------------------------------------------------------------
    @staticmethod
    def _dilate(x, r):
        return F.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r) if r > 0 else x

    @staticmethod
    def _boundary(x):
        return x * (1.0 - (-F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)))

    @torch.no_grad()
    def update(self, logits, target, skeleton=None, valid=None, names=None):
        dev = logits.device
        pred = torch.sigmoid(logits.float())
        tgt = target.float().to(dev)
        B = pred.shape[0]
        skel = skeleton.float().to(dev) if skeleton is not None else None
        if valid is not None:
            v = valid.float().to(dev)
            pred = pred * v
            tgt = tgt * v
            if skel is not None:
                skel = skel * v

        P = torch.cat([pred, pred.sum(1, keepdim=True).clamp(0.0, 1.0)], 1)     # (B,G,H,W)
        T = torch.cat([tgt, tgt.sum(1, keepdim=True).clamp(0.0, 1.0)], 1)
        pb = (P >= self.thresh).float()
        tb = (T > self.gt_thresh).float()
        if valid is not None:
            pb = pb * v
            tb = tb * v

        def s(x):                                    # per-sample, per-group sums -> (B,G) float64 numpy
            return x.sum(dim=(2, 3)).double().cpu().numpy()

        # regression error on the union ("active") region
        act = torch.clamp(pb + tb, max=1.0)
        err = P - T
        rec = dict(mse=s(err ** 2 * act), mae=s(err.abs() * act), nact=s(act))

        # relaxed precision / recall (tolerance)
        rec['p_num'] = s(pb * self._dilate(tb, self.tolerance))
        rec['p_den'] = s(pb)
        rec['r_num'] = s(tb * self._dilate(pb, self.tolerance))
        rec['r_den'] = s(tb)

        # boundary F (bf2_tolerance)
        bp, bt = self._boundary(pb), self._boundary(tb)
        rec['bp_num'] = s(bp * self._dilate(bt, self.bf2_tolerance))
        rec['bp_den'] = s(bp)
        rec['br_num'] = s(bt * self._dilate(bp, self.bf2_tolerance))
        rec['br_den'] = s(bt)

        # clDice with the same soft skeleton operator as the loss
        Sp = (soft_skeletonize(pb, self.skel_iters) > 0.5).float()
        Sl = (soft_skeletonize(tb, self.skel_iters) > 0.5).float()
        rec['cl_pnum'] = s(Sp * tb); rec['cl_pden'] = s(Sp)
        rec['cl_snum'] = s(Sl * pb); rec['cl_sden'] = s(Sl)

        if skel is not None:
            S = torch.cat([skel, skel.sum(1, keepdim=True).clamp(0.0, 1.0)], 1)
            gt_curve = (S > 0.5).float()
        else:
            gt_curve = Sl
        rec['gt_px'] = s(tb)
        rec['pred_px'] = s(pb)
        rec['gt_line_px'] = s(gt_curve)

        # topology / curve metrics (CPU loops, only where something exists)
        nan = np.full((B, self.G), np.nan)
        for k in ('conn_pm', 'conn_pn', 'conn_gm', 'conn_gn', 'topo_sum', 'topo_n', 'ch_sum', 'hd_sum',
                  'n_curve', 'n_gtpatch', 'chm_sum', 'hdm_sum', 'n_curve_m'):
            rec[k] = np.zeros((B, self.G))
        chamfer_pp, hd95_pp = nan.copy(), nan.copy()

        if self.topology:
            pb_n = pb.bool().cpu().numpy(); tb_n = tb.bool().cpu().numpy()
            sp_n = Sp.bool().cpu().numpy(); gc_n = gt_curve.bool().cpu().numpy()
            for b in range(B):
                for g in range(self.G):
                    pm, tm = pb_n[b, g], tb_n[b, g]
                    has_p, has_t = pm.any(), tm.any()
                    if not (has_p or has_t):
                        continue
                    lbl_p, n_p = ndimage.label(pm, structure=_CONN8)
                    lbl_t, n_t = ndimage.label(tm, structure=_CONN8)
                    overlap = pm & tm
                    mp_ids = np.unique(lbl_p[overlap]) if overlap.any() else np.empty(0, dtype=int)
                    mt_ids = np.unique(lbl_t[overlap]) if overlap.any() else np.empty(0, dtype=int)
                    rec['conn_pn'][b, g] = n_p; rec['conn_gn'][b, g] = n_t
                    rec['conn_pm'][b, g] = mp_ids.size; rec['conn_gm'][b, g] = mt_ids.size
                    if max(n_p, n_t) > 0:
                        rec['topo_sum'][b, g] = 1.0 - abs(n_p - n_t) / max(n_p, n_t)
                        rec['topo_n'][b, g] = 1.0

                    pts_gt = np.argwhere(gc_n[b, g])
                    pts_pr = np.argwhere(sp_n[b, g])
                    if pts_gt.shape[0] > 0:
                        rec['n_gtpatch'][b, g] = 1.0
                        if pts_pr.shape[0] > 0:
                            ch, hd = chamfer_hd95(pts_pr.astype(np.float64), pts_gt.astype(np.float64))
                            rec['ch_sum'][b, g], rec['hd_sum'][b, g], rec['n_curve'][b, g] = ch, hd, 1.0
                            chamfer_pp[b, g], hd95_pp[b, g] = ch, hd
                        # matched-components-only variant
                        if mp_ids.size and mt_ids.size:
                            keep_p = np.isin(lbl_p, mp_ids) & sp_n[b, g]
                            keep_t = np.isin(lbl_t, mt_ids) & gc_n[b, g]
                            if keep_p.any() and keep_t.any():
                                ch, hd = chamfer_hd95(np.argwhere(keep_p).astype(np.float64),
                                                      np.argwhere(keep_t).astype(np.float64))
                                rec['chm_sum'][b, g], rec['hdm_sum'][b, g], rec['n_curve_m'][b, g] = ch, hd, 1.0

        for k in _ACC_KEYS:
            self.acc[k] += rec[k].sum(axis=0)
        self.n_patches += B

        if self.keep_per_patch:
            rec['chamfer'] = chamfer_pp
            rec['hd95'] = hd95_pp
            self._pp.append(rec)
            self._names.extend(list(names) if names is not None else [f'patch_{self.n_patches - B + i}' for i in range(B)])

    # -- finalisation --------------------------------------------------------------------
    def _finalize_group(self, g: int) -> dict:
        a = {k: float(v[g]) for k, v in self.acc.items()}
        mse = _div(a['mse'], a['nact'], float('nan'))
        mae = _div(a['mae'], a['nact'], float('nan'))
        P = _div(a['p_num'], a['p_den']); R = _div(a['r_num'], a['r_den'])
        bP = _div(a['bp_num'], a['bp_den']); bR = _div(a['br_num'], a['br_den'])
        tprec = _div(a['cl_pnum'], a['cl_pden']); tsens = _div(a['cl_snum'], a['cl_sden'])
        corr = _div(a['conn_pm'], a['conn_pn']); comp = _div(a['conn_gm'], a['conn_gn'])
        nan = float('nan')
        out = dict(
            active_mse=mse, active_mae=mae, active_rmse=math.sqrt(mse) if mse == mse else nan,
            relaxed_precision=P, relaxed_recall=R, relaxed_f1=_f1(P, R), relaxed_iou=_quality(P, R),
            bf2_precision=bP, bf2_recall=bR, bf2=_f1(bP, bR),
            cldice=_f1(tprec, tsens),
        )
        if self.topology:
            out.update(
                connectivity_correctness=corr, connectivity_completeness=comp, connectivity=_f1(corr, comp),
                topo_lc=_div(a['topo_sum'], a['topo_n'], nan),
                chamfer_distance=_div(a['ch_sum'], a['n_curve'], nan),
                hausdorff95_distance=_div(a['hd_sum'], a['n_curve'], nan),
                curve_coverage=_div(a['n_curve'], a['n_gtpatch'], nan),
                chamfer_distance_matched=_div(a['chm_sum'], a['n_curve_m'], nan),
                hausdorff95_distance_matched=_div(a['hdm_sum'], a['n_curve_m'], nan),
            )
        else:
            out.update({k: nan for k in TOPO_KEYS})
        out.update(gt_px=a['gt_px'], pred_px=a['pred_px'], gt_line_px=a['gt_line_px'],
                   n_patches_gt=a['n_gtpatch'])
        return out

    def compute(self) -> dict:
        per_class = [self._finalize_group(g) for g in range(self.N)]
        merged = self._finalize_group(self.N)
        supported = [pc for pc in per_class if pc['gt_px'] > 0]
        macro = {}
        for k in METRIC_KEYS:
            vals = np.array([pc[k] for pc in supported], dtype=np.float64)
            macro[k] = float(np.nanmean(vals)) if vals.size and np.isfinite(vals).any() else float('nan')
        macro['gt_px'] = float(sum(pc['gt_px'] for pc in per_class))
        macro['pred_px'] = float(sum(pc['pred_px'] for pc in per_class))
        macro['gt_line_px'] = float(sum(pc['gt_line_px'] for pc in per_class))
        macro['n_patches_gt'] = merged['n_patches_gt']
        macro['n_classes_supported'] = float(len(supported))
        out = dict(merged=merged, macro=macro)
        for i, pc in enumerate(per_class):
            out[f'class_{i + 1}'] = pc
        return out

    def per_patch(self):
        """Per-patch scores for paired tests.  NaN = undefined for that patch/group (e.g. GT and
        prediction both empty)."""
        if not self._pp:
            return {}, []
        cat = {k: np.concatenate([r[k] for r in self._pp], axis=0) for k in self._pp[0].keys()}
        groups = {}
        for g in range(self.G):
            name = 'merged' if g == self.N else f'class_{g + 1}'
            pn, pd_, rn, rd = cat['p_num'][:, g], cat['p_den'][:, g], cat['r_num'][:, g], cat['r_den'][:, g]
            both_empty = (pd_ == 0) & (rd == 0)
            P = np.where(pd_ > 0, pn / np.maximum(pd_, 1), 0.0)
            R = np.where(rd > 0, rn / np.maximum(rd, 1), 0.0)
            f1 = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)
            q = np.where(P + R - P * R > 0, P * R / np.maximum(P + R - P * R, 1e-12), 0.0)
            f1 = np.where(both_empty, np.nan, f1); q = np.where(both_empty, np.nan, q)
            tp = np.where(cat['cl_pden'][:, g] > 0, cat['cl_pnum'][:, g] / np.maximum(cat['cl_pden'][:, g], 1), 0.0)
            ts = np.where(cat['cl_sden'][:, g] > 0, cat['cl_snum'][:, g] / np.maximum(cat['cl_sden'][:, g], 1), 0.0)
            cl = np.where(tp + ts > 0, 2 * tp * ts / np.maximum(tp + ts, 1e-12), 0.0)
            cl = np.where((cat['cl_pden'][:, g] == 0) & (cat['cl_sden'][:, g] == 0), np.nan, cl)
            groups[name] = dict(relaxed_f1=f1, relaxed_iou=q, cldice=cl,
                                chamfer_distance=cat['chamfer'][:, g], hausdorff95_distance=cat['hd95'][:, g])
        return groups, list(self._names)


# ----------------------------------------------------------------------------------------
#  ordinal (height-class) evaluation + confusion matrix in km of hedge length
# ----------------------------------------------------------------------------------------
def quadratic_weighted_kappa(D: np.ndarray) -> float:
    """D[i, j]: (weighted) count of reference class i predicted as class j.  Ordinal classes 0..N-1."""
    D = np.asarray(D, dtype=np.float64)
    N = D.shape[0]
    tot = D.sum()
    if tot <= 0 or N < 2:
        return float('nan')
    O = D / tot
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.indices((N, N))
    W = ((i - j) / (N - 1.0)) ** 2
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 0 else float('nan')


def macro_f1_from_confusion(D: np.ndarray) -> float:
    """Macro F1 over classes with reference support; rows = reference, cols = prediction."""
    D = np.asarray(D, dtype=np.float64)
    f1s = []
    for k in range(D.shape[0]):
        tp = D[k, k]; fn = D[k, :].sum() - tp; fp = D[:, k].sum() - tp
        if D[k, :].sum() <= 0:
            continue
        den = 2 * tp + fp + fn
        f1s.append(2 * tp / den if den > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else float('nan')


def ordinal_summary(D: np.ndarray, class_mid=None) -> dict:
    D = np.asarray(D, dtype=np.float64)
    N = D.shape[0]
    tot = D.sum()
    if tot <= 0:
        return {k: float('nan') for k in ('exact', 'within_one', 'class_mae', 'qwk', 'macro_f1_class',
                                          'height_mae_m', 'height_rmse_m')}
    i, j = np.indices((N, N))
    out = dict(
        exact=float(np.trace(D) / tot),
        within_one=float(D[np.abs(i - j) <= 1].sum() / tot),
        class_mae=float((np.abs(i - j) * D).sum() / tot),
        qwk=quadratic_weighted_kappa(D),
        macro_f1_class=macro_f1_from_confusion(D),
    )
    if class_mid is not None and len(class_mid) == N:
        mid = np.asarray(class_mid, dtype=np.float64)
        dh = mid[None, :] - mid[:, None]          # predicted height - reference height
        out['height_mae_m'] = float((np.abs(dh) * D).sum() / tot)
        out['height_rmse_m'] = float(math.sqrt((dh ** 2 * D).sum() / tot))
    else:
        out['height_mae_m'] = out['height_rmse_m'] = float('nan')
    return out


class OrdinalKm:
    """
    Length-weighted class confusion on the exact GT centre-lines.
      * every GT line pixel carries its Kulpa length share (metres = weight * pixel_size),
      * its predicted class is argmax_c p_c if the merged probability >= thresh, else 'missed'.
    confusion_km has shape (N, N+1): rows = reference class, columns = predicted class, last column = missed.
    Ordinal scores (exact, within-one, class MAE, QWK, height error in m) use the detected part only;
    `detection_rate` says which fraction of the reference length that is.
    """

    def __init__(self, num_classes: int, thresh: float = 0.3, class_mid=None):
        self.N = num_classes
        self.thresh = thresh
        self.class_mid = class_mid
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.N, self.N + 1), dtype=np.float64)     # km

    @torch.no_grad()
    def update(self, logits, skeleton, valid=None, pixel_size_m=None):
        pred = torch.sigmoid(logits.float())
        if valid is not None:
            pred = pred * valid.float().to(pred.device)
        B = pred.shape[0]
        merged = pred.sum(1).clamp(0.0, 1.0)
        pr_cls = pred.argmax(1)
        pr_cls = torch.where(merged >= self.thresh, pr_cls, torch.full_like(pr_cls, self.N))
        line = (skeleton.float().to(pred.device) > 0.5)
        gt_cls = line.float().argmax(1)
        line_any = line.any(1).cpu().numpy()
        w = kulpa_pixel_weights(line_any)                                  # (B,H,W) px
        px = np.ones(B) if pixel_size_m is None else np.asarray(pixel_size_m, dtype=np.float64)
        w = w * px[:, None, None] / 1000.0                                 # km
        idx = (gt_cls.cpu().numpy() * (self.N + 1) + pr_cls.cpu().numpy())[line_any]
        self.cm += np.bincount(idx, weights=w[line_any], minlength=self.N * (self.N + 1)).reshape(self.N, self.N + 1)

    def compute(self) -> dict:
        D = self.cm[:, :self.N]
        total = self.cm.sum()
        out = ordinal_summary(D, self.class_mid)
        out['detection_rate'] = float(D.sum() / total) if total > 0 else float('nan')
        out['reference_km'] = float(total)
        out['reference_km_per_class'] = self.cm.sum(1).tolist()
        out['confusion_km'] = self.cm.tolist()
        return out
