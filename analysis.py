"""
analysis.py -- studies that go beyond the standard metrics.  Every function is independent and is switched
on/off from run_all.py.

  paired_stats            per-patch paired comparison of two models (Wilcoxon, Cliff's delta, bootstrap CI)
  achievable_ceiling      best possible class-assignment scores under a noisy-label reference (analytic)
  coarea / Kulpa / Chapman  network-length estimators (+ synthetic calibration of the coarea formula)
  perturbation_eval       inference-time modality perturbations (DEM removed/noised/smoothed, MS removed)
  fast_weight_probe       size of the TTT fast-weight update per direction/scale vs patch covariates
  complexity_report       parameters, FLOPs, latency
"""
import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage, stats
from skimage.draw import line as sk_line
from skimage.morphology import skeletonize

from metrics import (SegMetrics, ordinal_summary, kulpa_length_px, kulpa_pixel_weights)
from dataset import load_band_statistics


# =========================================================================================
#  1. paired statistics
# =========================================================================================
def paired_stats(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """a, b: per-patch scores of two models on the SAME patches (NaN = undefined -> pair dropped).
    Reports mean difference with a paired bootstrap 95% CI, Wilcoxon signed-rank p, Cliff's delta
    (dominance of a over b, via the Mann-Whitney U) and the rank-biserial correlation of the paired differences."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    n = d.size
    out = dict(n=int(n), mean_a=float(a.mean()) if n else np.nan, mean_b=float(b.mean()) if n else np.nan,
               mean_diff=float(d.mean()) if n else np.nan, ci_lo=np.nan, ci_hi=np.nan,
               wilcoxon_p=np.nan, cliffs_delta=np.nan, rank_biserial=np.nan)
    if n < 3:
        return out
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, int(2e7 // n))
    for i in range(0, n_boot, chunk):
        m = min(chunk, n_boot - i)
        means[i:i + m] = d[rng.integers(0, n, size=(m, n))].mean(axis=1)
    out['ci_lo'], out['ci_hi'] = [float(x) for x in np.percentile(means, [2.5, 97.5])]
    nz = d[d != 0]
    if nz.size:
        try:
            out['wilcoxon_p'] = float(stats.wilcoxon(nz).pvalue)
        except Exception:
            pass
        r = stats.rankdata(np.abs(nz))
        wp, wm = r[nz > 0].sum(), r[nz < 0].sum()
        out['rank_biserial'] = float((wp - wm) / (wp + wm))
    u = stats.mannwhitneyu(a, b, alternative='two-sided').statistic
    out['cliffs_delta'] = float(2.0 * u / (a.size * b.size) - 1.0)
    return out


# =========================================================================================
#  2. achievable ceiling under noisy height-class labels
# =========================================================================================
def reference_confusion(n_classes: int, exact: float = 0.32, within_one: float = 0.63, decay: float = 0.5):
    """C[i, j] = P(reference label j | true class i): diagonal = exact, the two neighbours share
    (within_one - exact), the remaining (1 - within_one) decays geometrically with |i-j| (>=2).
    Every row sums to 1 and every row has P(|i-j|<=1) = within_one (edge classes give all neighbour mass to
    their single neighbour)."""
    n = n_classes
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = exact
        nb = [j for j in (i - 1, i + 1) if 0 <= j < n]
        for j in nb:
            C[i, j] = (within_one - exact) / len(nb)
        far = [j for j in range(n) if abs(j - i) >= 2]
        if far:
            wts = np.array([decay ** (abs(j - i) - 2) for j in far])
            for j, w in zip(far, wts / wts.sum()):
                C[i, j] = (1.0 - within_one) * w
        else:                                       # tiny n: renormalise
            C[i] /= C[i].sum()
    return C


def achievable_ceiling(class_prior, class_mid=None, exact=0.32, within_one=0.63, decay=0.5) -> dict:
    """Expected class-assignment scores of a PERFECT model judged against noisy labels (analytic, exact expectation).
    D[label, pred] = prior[pred] * C[pred, label]  (pred = true class, label = noisy reference)."""
    prior = np.asarray(class_prior, dtype=float)
    prior = prior / prior.sum()
    C = reference_confusion(len(prior), exact, within_one, decay)
    D = (prior[:, None] * C).T
    return ordinal_summary(D, class_mid)


# =========================================================================================
#  3. length estimators
# =========================================================================================
def coarea_length_px(field: np.ndarray, T_px: float, n_ends: float = 2.0, floor: float = 0.0):
    """
    field : (H, W) in [0,1]; the ramp  D = T_px * field  rises with slope 1 px/px towards the line centre.
    Coarea formula:  integral |grad D| dA = integral_0^T Length{D = s} ds = 2*L*T + n_ends*(pi*T^2/2)
                     (two offset curves of length L per line + a half-disc cap of area pi*T^2/2 per free end)
                     =>  L = (I - n_ends*pi*T^2/2) / (2*T).
    (n_ends=2 corresponds to ONE isolated segment; a network with E free ends needs E.)
    Returns (L_px, I).
    """
    D = np.where(field >= floor, field, 0.0) * T_px
    gy, gx = np.gradient(D)
    I = float(np.hypot(gx, gy).sum())
    L = (I - n_ends * 0.5 * math.pi * T_px ** 2) / (2.0 * T_px)
    return max(L, 0.0), I


def skeleton_endpoints(skel: np.ndarray, border: int = 2) -> int:
    """Free ends of a skeleton (pixels with exactly one 8-neighbour), ignoring ends cut by the patch border."""
    s = skel.astype(np.uint8)
    nb = ndimage.convolve(s, np.ones((3, 3), dtype=np.uint8), mode='constant') - s
    ends = (s == 1) & (nb == 1)
    if border > 0:
        ends[:border, :] = False; ends[-border:, :] = False; ends[:, :border] = False; ends[:, -border:] = False
    return int(ends.sum())


def synthetic_calibration(T_px: float = 5.0, size: int = 256, n_trials: int = 40, n_lines: int = 4, seed: int = 0) -> dict:
    """Ratio estimate/true length on synthetic networks with KNOWN length, for the coarea formula (n_ends = 2*n_lines)
    and for Kulpa on the raster line.  'isolated' lines are rejection-sampled to stay >2T+1 apart; 'crossing'
    lines are unconstrained (they cross), which is where the coarea formula is expected to under-estimate."""
    rng = np.random.default_rng(seed)
    margin = int(T_px) + 3
    res = {}
    for scenario in ('isolated', 'crossing'):
        ratios_c, ratios_k = [], []
        for _ in range(n_trials):
            lines, mask, true_len = [], np.zeros((size, size), bool), 0.0
            tries = 0
            while len(lines) < n_lines and tries < 500:
                tries += 1
                p0 = rng.integers(margin, size - margin, 2); ang = rng.uniform(0, np.pi)
                ln = rng.uniform(40, 110)
                p1 = np.clip((p0 + ln * np.array([np.sin(ang), np.cos(ang)])).astype(int), margin, size - margin - 1)
                rr, cc = sk_line(p0[0], p0[1], p1[0], p1[1])
                seg = np.zeros((size, size), bool); seg[rr, cc] = True
                if scenario == 'isolated' and mask.any():
                    if ndimage.distance_transform_edt(~mask)[seg].min() < 2 * T_px + 2:
                        continue
                lines.append(seg); mask |= seg
                true_len += float(np.hypot(*(p1 - p0)))
            if len(lines) < 2 or true_len <= 0:
                continue
            d = ndimage.distance_transform_edt(~mask)
            field = np.clip(T_px - d, 0, T_px) / T_px
            l_est, _ = coarea_length_px(field, T_px, n_ends=2 * len(lines))
            ratios_c.append(l_est / true_len)
            ratios_k.append(kulpa_length_px(mask) / true_len)
        res[scenario] = dict(n=len(ratios_c), coarea_ratio_mean=float(np.mean(ratios_c)), coarea_ratio_std=float(np.std(ratios_c)),
                             kulpa_ratio_mean=float(np.mean(ratios_k)), kulpa_ratio_std=float(np.std(ratios_k)))
    return res


def scene_meta(cfg: dict) -> dict:
    """{scene: dict(pixel_size_m, dist_cap_m)} from the NPZ metadata written by dataprep.py."""
    out = {}
    for sc in cfg['scenes']:
        d = os.path.join(cfg['base_dir'], sc, 'npz')
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith('.npz'))
        if files:
            m = np.load(os.path.join(d, files[0]), allow_pickle=True)['meta'].item()
            out[sc] = dict(pixel_size_m=float(m.get('pixel_size_x', 1.0)), dist_cap_m=float(m.get('dist_cap_m', 5.0)))
    return out


def _scene_of(name, scenes):
    for s in scenes:
        if name.startswith(s + '_'):
            return s
    return scenes[0]


def per_patch_lengths(cache: dict, cfg: dict, thresh: float, match_tol_m: float = 10.0) -> pd.DataFrame:
    """Per-patch lengths (px) of the reference line and of the prediction under each estimator."""
    meta = scene_meta(cfg)
    rows = []
    n = cache['logits'].shape[0]
    for i in range(n):
        name = cache['names'][i]
        sc = _scene_of(name, cfg['scenes']); pm = meta.get(sc, dict(pixel_size_m=1.0, dist_cap_m=5.0))
        px, T = pm['pixel_size_m'], pm['dist_cap_m'] / pm['pixel_size_m']
        v = cache['valid'][i, 0].numpy().astype(bool)
        prob = torch.sigmoid(cache['logits'][i].float()).numpy()
        field = np.clip(prob.sum(0), 0, 1) * v
        pred_bin = field >= thresh
        ref = (cache['skeleton'][i].numpy().sum(0) > 0) & v
        pred_skel = skeletonize(pred_bin) if pred_bin.any() else np.zeros_like(pred_bin)
        E = skeleton_endpoints(pred_skel)
        l_noend, I = coarea_length_px(field, T, n_ends=0.0)
        l_coarea, _ = coarea_length_px(field, T, n_ends=float(E))
        tol = match_tol_m / px
        if ref.any() and pred_skel.any():
            w_ref, w_pred = kulpa_pixel_weights(ref), kulpa_pixel_weights(pred_skel)
            d_to_pred = ndimage.distance_transform_edt(~pred_skel)
            d_to_ref = ndimage.distance_transform_edt(~ref)
            m1 = float(w_ref[(d_to_pred <= tol) & ref].sum()); m2 = float(w_pred[(d_to_ref <= tol) & pred_skel].sum())
        else:
            m1 = m2 = 0.0
        rows.append(dict(name=name, px_m=px, T_px=T, L_ref=kulpa_length_px(ref), L_kulpa=kulpa_length_px(pred_skel),
                         L_coarea=l_coarea, L_coarea_noend=l_noend, n_ends=E, m1=m1, m2=m2))
    return pd.DataFrame(rows)


def chapman(L1: float, L2: float, m: float) -> float:
    """Chapman-corrected Lincoln-Petersen estimate of the total length (units of the inputs).
    List 1 = reference map, list 2 = model, m = length found by both (mean of the two one-sided matches).
    Assumes the two 'captures' are independent -- they are not (both miss what is hard to see), so the result is
    a lower bound on the true total length."""
    return (L1 + 1.0) * (L2 + 1.0) / (m + 1.0) - 1.0


def estimator_eval(cache_val: dict, cache_test: dict, cfg: dict, thresh: float, match_tol_m: float = 10.0) -> dict:
    """Length of the hedge network over the held-out test patches, from the model output, by several estimators.
    The scale factor of the coarea estimator is fitted on VALIDATION patches (through the origin) and applied to test."""
    dv = per_patch_lengths(cache_val, cfg, thresh, match_tol_m)
    dt = per_patch_lengths(cache_test, cfg, thresh, match_tol_m)
    km = lambda df, col: float((df[col] * df['px_m']).sum() / 1000.0)
    out = dict(n_val=len(dv), n_test=len(dt), reference_km=km(dt, 'L_ref'))
    for col in ('L_kulpa', 'L_coarea', 'L_coarea_noend'):
        alpha = float((dv['L_ref'].sum()) / max(dv[col].sum(), 1e-9))
        est = km(dt, col)
        per = (dt[col] * dt['px_m'] / 1000.0).values
        ref = (dt['L_ref'] * dt['px_m'] / 1000.0).values
        out[col] = dict(km=est, rel_err_pct=100.0 * (est - out['reference_km']) / max(out['reference_km'], 1e-9),
                        val_scale=alpha, calibrated_km=alpha * est,
                        calibrated_rel_err_pct=100.0 * (alpha * est - out['reference_km']) / max(out['reference_km'], 1e-9),
                        per_patch_mae_km=float(np.mean(np.abs(alpha * per - ref))),
                        per_patch_pearson=float(np.corrcoef(per, ref)[0, 1]) if per.std() > 0 and ref.std() > 0 else np.nan)
    L1, L2 = float((dt['L_ref'] * dt['px_m']).sum()), float((dt['L_kulpa'] * dt['px_m']).sum())
    m = float(0.5 * ((dt['m1'] + dt['m2']) * dt['px_m']).sum())
    out['capture_recapture'] = dict(list1_ref_km=L1 / 1000, list2_model_km=L2 / 1000, matched_km=m / 1000,
                                    chapman_total_km=chapman(L1, L2, m) / 1000, match_tol_m=match_tol_m,
                                    note='lower bound; lists are not independent')
    out['per_patch_test'] = dt.to_dict(orient='list')
    return out


# =========================================================================================
#  4. inference-time modality perturbations
# =========================================================================================
def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2); k = k / k.sum()
    c = x.shape[1]
    xp = F.pad(x, (r, r, r, r), mode='reflect')
    xp = F.conv2d(xp, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(xp, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def dem_std_by_scene(cfg: dict) -> dict:
    out = {}
    for sc in cfg['scenes']:
        p = os.path.join(cfg['base_dir'], sc, 'band_statistics.csv')
        if os.path.exists(p):
            _, _, dm, ds = load_band_statistics(p)
            out[sc] = (float(dm[0]), float(ds[0]))
    return out


DEFAULT_PERTURBATIONS = ['clean', 'dem_zero', 'ms_zero', 'dem_noise_1m', 'dem_noise_3m', 'dem_smooth_3px']


def make_perturbation(name: str, dem_std: torch.Tensor, gen: torch.Generator):
    """Returns f(dem_norm, ms_norm) -> (dem_norm', ms_norm').  Inputs are the model inputs (z-scored), so
    'zero' means the scene mean (= no information), and metre-noise is divided by the scene DEM std."""
    if name == 'clean':
        return lambda d, m: (d, m)
    if name == 'dem_zero':
        return lambda d, m: (torch.zeros_like(d), m)
    if name == 'ms_zero':
        return lambda d, m: (d, torch.zeros_like(m))
    if name.startswith('dem_noise_'):
        sigma_m = float(name.split('_')[-1].replace('m', ''))
        def f(d, m):
            noise = torch.randn(d.shape, generator=gen, device='cpu').to(d.device)
            return d + noise * (sigma_m / dem_std.to(d.device)), m
        return f
    if name.startswith('dem_smooth_'):
        s = float(name.split('_')[-1].replace('px', ''))
        return lambda d, m: (_gaussian_blur(d, s), m)
    raise KeyError(name)


@torch.no_grad()
def perturbation_eval(model, loader, cfg: dict, num_classes: int, device, perturbations=None, run_step_fn=None,
                      loss_fn=None, seed: int = 0) -> dict:
    """Per-perturbation SegMetrics (no topology).  Reports summary['macro'|'merged'|'class_i'] for each one."""
    perturbations = perturbations or DEFAULT_PERTURBATIONS
    stats_by_scene = dem_std_by_scene(cfg)
    results = {}
    model.eval()
    for pname in perturbations:
        gen = torch.Generator().manual_seed(seed)
        eng = SegMetrics(num_classes, thresh=cfg['seg_thresh'], tolerance=cfg['metric_tolerance'],
                         bf2_tolerance=cfg['bf2_tolerance'], gt_thresh=cfg.get('gt_thresh', 0.01), topology=False)
        for ms, dem, dist_gt, mask, skeleton, names in loader:
            std = torch.tensor([stats_by_scene.get(_scene_of(n, cfg['scenes']), (0.0, 1.0))[1] for n in names],
                               dtype=torch.float32).view(-1, 1, 1, 1)
            f = make_perturbation(pname, std, gen)
            ms, dem = ms.to(device), dem.to(device)
            dem_p, ms_p = f(dem, ms)
            _, _, logits, gt, skel, valid = run_step_fn(ms_p, dem_p, dist_gt.to(device), mask.to(device),
                                                        skeleton.to(device), model, loss_fn)
            eng.update(logits, gt, skel, valid, names)
        results[pname] = eng.compute()
    return results


# =========================================================================================
#  5. fast-weight probe (TTT only)
# =========================================================================================
@torch.no_grad()
def fast_weight_probe(model, loader, cfg: dict, device, max_patches: int = 400) -> pd.DataFrame:
    """
    For every patch and every fusion scale s: size of the fast-weight update produced by each direction
        d_lin  = ||W1'-W1|| + ||W2'-W2||   (SwiGLU branch)      d_spa = ||W3'-W3||  (3x3 depth-wise conv branch)
    'dem->ms' = keys/values from the DEM stream, applied to MS queries;  'ms->dem' the opposite.
    dem_reliance = d(dem->ms) / (d(dem->ms) + d(ms->dem)).  Covariates: hedge area fraction, dominant height
    class, median raw DEM value at hedge pixels and DEM std of the patch.
    """
    stats_by_scene = dem_std_by_scene(cfg)
    model.eval(); model.set_probe(True)
    rows, seen = [], 0
    for ms, dem, dist_gt, mask, skeleton, names in loader:
        model(dem.to(device), ms.to(device))
        probes = model.pop_probes()
        if not probes:
            raise RuntimeError('this model has no TTT fast weights (probe needs variant TTT)')
        B = dem.shape[0]
        gt = dist_gt.sum(1)
        for b in range(B):
            sc = _scene_of(names[b], cfg['scenes']); mu, sd = stats_by_scene.get(sc, (0.0, 1.0))
            hedge = (gt[b] > 0.01)
            raw = dem[b, 0] * sd + mu
            area = dist_gt[b].sum((1, 2)).numpy()
            row = dict(name=names[b], hedge_frac=float(hedge.float().mean()),
                       dominant_class=int(area.argmax() + 1) if area.max() > 0 else 0,
                       dem_at_hedge_median=float(raw[hedge].median()) if hedge.any() else np.nan,
                       dem_std=float(raw.std()))
            for s, recs in enumerate(probes):
                d = {r['direction']: r for r in recs}
                for kind in ('d_linear', 'd_spatial'):
                    a, o = float(d['dem->ms'][kind][b]), float(d['ms->dem'][kind][b])
                    row[f's{s + 1}_{kind}_dem2ms'], row[f's{s + 1}_{kind}_ms2dem'] = a, o
                    row[f's{s + 1}_{kind}_dem_reliance'] = a / (a + o + 1e-12)
            rows.append(row)
        seen += B
        if seen >= max_patches:
            break
    model.set_probe(False)
    return pd.DataFrame(rows)


def probe_correlations(df: pd.DataFrame) -> dict:
    """Spearman correlation of every dem_reliance column with hedge height (median raw DEM at hedge pixels)."""
    out = {}
    ok = df['dem_at_hedge_median'].notna()
    for c in [c for c in df.columns if c.endswith('dem_reliance')]:
        if ok.sum() > 5:
            rho, p = stats.spearmanr(df.loc[ok, 'dem_at_hedge_median'], df.loc[ok, c])
            out[c] = dict(spearman_rho=float(rho), p=float(p), n=int(ok.sum()))
    return out


# =========================================================================================
#  6. complexity
# =========================================================================================
def complexity_report(build_fn, variants: list, device, img: int = 256, batch: int = 1, warmup: int = 3,
                      iters: int = 10, px_m: float = 1.0, stride_px: int = 192) -> pd.DataFrame:
    """Parameters, forward GFLOPs and latency per variant.
    FLOPs come from torch's FlopCounterMode: it counts matmul/conv/attention only (2 FLOPs per MAC); the
    element-wise operations inside the TTT fast-weight update are NOT counted, so TTT is slightly under-reported."""
    from torch.utils.flop_counter import FlopCounterMode
    rows = []
    for v in variants:
        net = build_fn(v).to(device).eval()
        dem = torch.randn(batch, 1, img, img, device=device); ms = torch.randn(batch, 6, img, img, device=device)
        n_params = sum(p.numel() for p in net.parameters())
        enc_ids = net.encoder_param_ids()
        n_enc = sum(p.numel() for p in net.parameters() if id(p) in enc_ids)
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                net(dem, ms)
            flops = fc.get_total_flops()
            for _ in range(warmup):
                net(dem, ms)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                net(dem, ms)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            lat = (time.time() - t0) / iters
        tiles_km2 = (1000.0 / (stride_px * px_m)) ** 2
        rows.append(dict(variant=v, params_M=n_params / 1e6, encoder_params_M=n_enc / 1e6,
                         non_encoder_params_M=(n_params - n_enc) / 1e6, gflops_per_patch=flops / 1e9 / batch,
                         latency_ms_per_patch=1000 * lat / batch, sec_per_km2=lat / batch * tiles_km2))
        del net
    return pd.DataFrame(rows)
