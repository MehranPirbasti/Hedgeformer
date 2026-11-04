"""
run_all.py -- one script that runs the whole study end to end:

    selftest -> train -> evaluate -> perturb -> probe -> ceiling -> complexity -> estimators -> stats -> report

    python run_all.py                       # everything that is enabled below
    python run_all.py --list                # show the job table and which switches are on, then exit
    python run_all.py --dry-run             # same as --list
    python run_all.py --only train,evaluate # run only these stages
    python run_all.py --skip perturb,probe  # run everything except these
    python run_all.py --groups fusion_core  # only these run groups
    python run_all.py --seeds 42            # override the seed list
    python run_all.py --epochs 10           # override epochs (quick trial)
    python run_all.py --smoke               # CPU smoke test on synthetic data with a tiny encoder (minutes)

HOW TO DISABLE A SCENARIO: flip `enabled` in GROUPS / STAGES below (or use --groups / --only / --skip).
Everything is idempotent: finished runs (done.json) and finished evaluations (metrics.json) are skipped, so
you can re-launch after a crash or after enabling more scenarios and only the missing work is done.
A failure in one job is logged and the runner continues with the next job.
"""
import os
import sys

if '--smoke' in sys.argv:                       # must be set before model.py is imported
    os.environ['HEDGE_ENCODER'] = 'mit_b0'

import argparse
import copy
import gc
import json
import time
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

import main as M
import validation as V
import analysis as A
from model import build_model, load_checkpoint, VARIANTS
from dataset import discover_num_classes

# =========================================================================================
#  1. USER SETTINGS
# =========================================================================================
OUT_ROOT = 'experiments'
SEEDS = [42, 1337, 2024]

BASE = dict(
    base_dir = r'B:\PhD\Classification\Data',
    scenes   = ['Scene1', 'Scene2'],
    epochs   = 50,
    # class names / metres per class (used only for labelling and the ordinal height error). Ids follow the
    # sorted hghtcls strings written by dataprep.py -- the runner prints the map, check the order matches!
    class_names = None,                 # e.g. ['1a', '1b', '1c', '2', '3', '4', '6']
    class_mid   = None,                 # e.g. [0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0]
)

# ---- switches: one line per stage -------------------------------------------------------
STAGES = dict(
    selftest   = True,     # math unit tests (loss, metrics, kulpa, soups, models) -- aborts the run if they fail
    train      = True,     # train every enabled job (each run also builds its model soups)
    evaluate   = True,     # test-set evaluation of every trained run (metrics, per-patch scores, km confusion, plots)
    perturb    = True,     # inference-time modality perturbations
    probe      = True,     # TTT fast-weight probe
    ceiling    = True,     # achievable ceiling under noisy labels vs measured ordinal scores
    complexity = True,     # params / FLOPs / latency
    estimators = True,     # coarea / Kulpa / capture-recapture network-length estimates
    stats      = True,     # paired significance tests between variants
    report     = True,     # aggregate tables (mean +- std over seeds), ablation deltas, figures
)

# ---- run groups: each entry expands to (variant x seed) training jobs -----------------------
GROUPS = dict(
    fusion_core = dict(enabled=True, variants=['Simple', 'SRA', 'TTT'], seeds=SEEDS),
    solo        = dict(enabled=True, variants=['DEM_only', 'MS_only'],  seeds=SEEDS),
    early_late  = dict(enabled=True, variants=['Early', 'Late'],        seeds=SEEDS),
    loss_ablation = dict(
        enabled=True, base_variant='TTT', seed=42,
        ablations={                                     # name -> (enabled, CFG overrides)
            'no_pu':              (True, dict(pu_threshold=1.01)),
            'no_margin':          (True, dict(margin=0.0)),
            'no_skeleton_boost':  (True, dict(skeleton_boost=0.0)),
            'no_cldice':          (True, dict(w_cldice=0.0)),
            'conventional_cldice':(True, dict(cldice_mode='conventional')),
            'no_merge_loss':      (True, dict(w_merge=0.0)),
        },
    ),
)

# ---- which weights of every run are evaluated on the test set ---------------------------------
EVAL_WEIGHTS = ['best_f1', 'soup_greedy', 'soup_uniform']     # also possible: best_rmse, best_chamfer, best_hd95, final
PRIMARY_WEIGHTS = 'best_f1'                                    # used by perturb / probe / estimators / stats

PERTURB_VARIANTS = ['Simple', 'SRA', 'TTT', 'Late']            # runs (all their seeds) that get perturbation tests
PERTURBATIONS = A.DEFAULT_PERTURBATIONS
PROBE_VARIANTS = ['TTT']
PROBE_MAX_PATCHES = 400
ESTIMATOR_VARIANTS = ['Simple', 'SRA', 'TTT']
ESTIMATOR_MATCH_TOL_M = 10.0
COMPLEXITY_VARIANTS = ['Simple', 'SRA', 'TTT', 'Early', 'Late', 'DEM_only', 'MS_only']

# paired tests: (model A, model B)  -- A minus B
STATS_PAIRS = [('TTT', 'Simple'), ('TTT', 'SRA'), ('SRA', 'Simple'), ('TTT', 'Early'), ('TTT', 'Late'),
               ('TTT', 'DEM_only'), ('TTT', 'MS_only')]
STATS_TARGETS = ['merged__relaxed_f1', 'merged__cldice', 'merged__chamfer_distance']
STATS_PER_CLASS_F1 = True

REPORT_METRICS = ['relaxed_f1', 'relaxed_iou', 'bf2', 'cldice', 'connectivity', 'topo_lc',
                  'chamfer_distance', 'hausdorff95_distance', 'curve_coverage', 'active_rmse']


# =========================================================================================
#  2. jobs
# =========================================================================================
class Job:
    def __init__(self, run_id, variant, seed, group, overrides=None):
        self.run_id, self.variant, self.seed, self.group = run_id, variant, seed, group
        self.overrides = overrides or {}

    def run_dir(self):
        return os.path.join(OUT_ROOT, 'runs', self.run_id)

    def cfg(self):
        cfg = copy.deepcopy(M.CFG)
        cfg.update(copy.deepcopy(BASE))
        cfg.update(self.overrides)
        cfg.update(variant=self.variant, seed=self.seed, run_dir=self.run_dir())
        return cfg


def build_jobs(groups_filter=None, seeds_override=None) -> list:
    jobs = {}
    def add(job):
        jobs.setdefault(job.run_id, job)
    for gname, g in GROUPS.items():
        if not g.get('enabled') or (groups_filter and gname not in groups_filter):
            continue
        if gname == 'loss_ablation':
            base_seed = seeds_override[0] if seeds_override else g['seed']
            bv = g['base_variant']
            add(Job(f'{bv}_s{base_seed}', bv, base_seed, 'fusion_core'))       # the "full loss" reference run
            for name, (on, ov) in g['ablations'].items():
                if on:
                    add(Job(f'{bv}_abl-{name}_s{base_seed}', bv, base_seed, 'loss_ablation', ov))
        else:
            for v in g['variants']:
                for s in (seeds_override or g['seeds']):
                    add(Job(f'{v}_s{s}', v, s, gname))
    return list(jobs.values())


# =========================================================================================
#  3. helpers
# =========================================================================================
_LOG = None


def log(msg=''):
    print(msg, flush=True)
    if _LOG:
        with open(_LOG, 'a') as f:
            f.write(msg + '\n')


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def weights_path(job: Job, name: str):
    p = os.path.join(job.run_dir(), f'{name}.pth')
    return p if os.path.exists(p) else None


def finished(job: Job) -> bool:
    return os.path.exists(os.path.join(job.run_dir(), 'done.json'))


def eval_dir(job: Job, wname: str) -> str:
    return os.path.join(OUT_ROOT, 'eval', job.run_id, wname)


def stage(name, fn, *a, **kw):
    t0 = time.time()
    log(f'\n{"#" * 78}\n#  STAGE: {name}\n{"#" * 78}')
    try:
        fn(*a, **kw)
        log(f'[stage {name}] done in {(time.time() - t0) / 60:.1f} min')
        return True
    except Exception:
        log(f'[stage {name}] FAILED:\n{traceback.format_exc()}')
        return False


def safe(label, fn, *a, **kw):
    """Run one job; a failure is logged and the runner moves on."""
    try:
        return fn(*a, **kw)
    except Exception:
        log(f'!! {label} FAILED:\n{traceback.format_exc()}')
        FAILURES.append(label)
        free()
        return None


FAILURES = []


# =========================================================================================
#  4. self-tests (math)
# =========================================================================================
def selftest():
    from scipy import ndimage
    from metrics import (soft_erode, soft_dilate, soft_skeletonize, soft_cldice_loss, SegMetrics, kulpa_length_px,
                         kulpa_pixel_weights, quadratic_weighted_kappa, OrdinalKm)
    torch.manual_seed(0)
    ok = lambda name: log(f'  [PASS] {name}')
    cross = ndimage.generate_binary_structure(2, 1)

    x = torch.rand(1, 1, 20, 20)
    assert np.allclose(soft_erode(x)[0, 0].numpy(), ndimage.minimum_filter(x[0, 0].numpy(), footprint=cross, mode='nearest'))
    assert np.allclose(soft_dilate(x)[0, 0].numpy(), ndimage.maximum_filter(x[0, 0].numpy(), size=3, mode='nearest'))
    ok('soft erode/dilate equal scipy min/max filters')

    def np_skel(img, it):
        er = lambda a: ndimage.minimum_filter(a, footprint=cross, mode='nearest')
        di = lambda a: ndimage.maximum_filter(a, size=3, mode='nearest')
        op = lambda a: di(er(a))
        sk = np.maximum(img - op(img), 0); a = img.copy()
        for _ in range(it):
            a = er(a); dl = np.maximum(a - op(a), 0); sk = sk + np.maximum(dl - sk * dl, 0)
        return np.clip(sk, 0, 1)
    m = np.zeros((32, 32), np.float32); m[10:17, 3:29] = 1
    assert np.allclose(soft_skeletonize(torch.tensor(m)[None, None], 10)[0, 0].numpy(), np_skel(m, 10), atol=1e-6)
    ok('soft skeleton equals a literal transcription of the clDice algorithm')

    t = torch.zeros(1, 1, 32, 32); t[0, 0, 10:17, 3:29] = 1; z = torch.zeros_like(t)
    assert soft_cldice_loss(t, t).item() < 1e-4
    assert abs(soft_cldice_loss(z, t, mode='fixed').item() - 1.0) < 1e-6
    assert soft_cldice_loss(z, t, mode='conventional').item() < 1e-3           # the old bug: "predict nothing" = perfect
    assert soft_cldice_loss(z, z).item() == 0.0
    ok('clDice loss: perfect->0, empty prediction->1 (fixed) [conventional gives ~0 = the bug], empty/empty->0')

    B, N, H, W = 2, 3, 12, 12
    lg = torch.randn(B, N, H, W) * 2; tg = torch.rand(B, N, H, W) * (torch.rand(B, N, H, W) > 0.6)
    sk = (torch.rand(B, N, H, W) > 0.9).float(); vd = (torch.rand(B, 1, H, W) > 0.15).float()
    L = M.LOSS(margin=0.05, skeleton_boost=3.0, pu_threshold=0.85, w_cldice=0.0, w_merge=0.0)
    got = L(lg, tg, vd, sk)[0].item()
    p = 1 / (1 + np.exp(-lg.numpy().astype(np.float64))); tt = tg.numpy().astype(np.float64)
    ms_, ma_ = [], []
    for c in range(N):
        nm = na = dn = 0.0
        for b in range(B):
            for y in range(H):
                for xx in range(W):
                    e = max(abs(p[b, c, y, xx] - tt[b, c, y, xx]) - 0.05, 0.0)
                    w = 1 + 3.0 * sk[b, c, y, xx].item()
                    if tt[b, c, y, xx] < 0.01 and p[b, c, y, xx] > 0.85:
                        w = 0.0
                    w *= vd[b, 0, y, xx].item(); nm += e * e * w; na += e * w; dn += w
        dn = max(dn, 1.0); ms_.append(nm / dn); ma_.append(na / dn)
    assert abs(got - (0.5 * np.mean(ms_) + 0.5 * np.mean(ma_))) < 1e-5
    ok('weighted dead-zone MSE/MAE (skeleton boost, PU mask, valid mask) equals brute-force loops')

    l1 = torch.randn(2, 1, 12, 12); t1 = torch.rand(2, 1, 12, 12) * (torch.rand(2, 1, 12, 12) > 0.5)
    s1 = (torch.rand(2, 1, 12, 12) > 0.9).float(); v1 = torch.ones(2, 1, 12, 12)
    a0 = M.LOSS(w_merge=0.0, cldice_iters=4)(l1, t1, v1, s1)[0].item(); a1 = M.LOSS(w_merge=0.5, cldice_iters=4)(l1, t1, v1, s1)[0].item()
    assert abs(a1 - 1.5 * a0) < 1e-5
    ok('merge-consistency loss equals the per-class loss when N=1')

    lg2 = torch.randn(1, 2, 16, 16); tg2 = torch.zeros(1, 2, 16, 16); tg2[0, 0, 8, 2:14] = 1
    vm = torch.ones(1, 1, 16, 16); vm[..., :4, :] = 0; lg3 = lg2.clone(); lg3[..., :4, :] = 20.0
    Lc = M.LOSS(w_mse=0, w_mae=0, w_merge=0, cldice_iters=4)
    assert abs(Lc(lg2, tg2, vm, None)[0].item() - Lc(lg3, tg2, vm, None)[0].item()) < 1e-6
    ok('nodata pixels do not influence the clDice term')

    # metrics vs brute force
    Nn, H2 = 2, 48
    T = np.zeros((1, Nn, H2, H2), np.float32); Ln = np.zeros_like(T)
    T[0, 0, 20:27, 4:44] = 1; Ln[0, 0, 23, 4:44] = 1; T[0, 1, 5:9, 10:14] = 0.5
    P = T.copy(); P[0, 0] = np.roll(P[0, 0], 4, axis=0) * 0.9; P[0, 1] = 0
    lgt = torch.logit(torch.tensor(np.clip(P, 1e-4, 1 - 1e-4)))
    sm = SegMetrics(Nn, thresh=0.3, tolerance=2, bf2_tolerance=2); sm.update(lgt, torch.tensor(T), torch.tensor(Ln))
    r = sm.compute()['class_1']
    pb = P[0, 0] >= 0.3; tb = T[0, 0] > 0.01
    dt = ndimage.distance_transform_cdt(~tb, metric='chessboard'); dp = ndimage.distance_transform_cdt(~pb, metric='chessboard')
    Pb, Rb = (dt[pb] <= 2).mean(), (dp[tb] <= 2).mean()
    assert abs(r['relaxed_precision'] - Pb) < 1e-9 and abs(r['relaxed_recall'] - Rb) < 1e-9
    assert abs(r['relaxed_f1'] - 2 * Pb * Rb / (Pb + Rb)) < 1e-9 and 0 < r['relaxed_recall'] < 1
    bnd = lambda a: a & ~ndimage.binary_erosion(a, structure=np.ones((3, 3)), border_value=1)
    bp, bt = bnd(pb), bnd(tb)
    dbt = ndimage.distance_transform_cdt(~bt, metric='chessboard'); dbp = ndimage.distance_transform_cdt(~bp, metric='chessboard')
    BP, BR = (dbt[bp] <= 2).mean(), (dbp[bt] <= 2).mean()
    assert abs(r['bf2'] - 2 * BP * BR / (BP + BR)) < 1e-9
    ok('relaxed precision/recall/F1 and boundary-F equal brute-force chessboard-distance computations')

    lp = torch.logit(torch.tensor(np.clip(T, 1e-4, 1 - 1e-4)))
    s2 = SegMetrics(Nn, thresh=0.3); s2.update(lp, torch.tensor(T), torch.tensor(Ln)); c = s2.compute()['class_1']
    assert min(c['relaxed_f1'], c['relaxed_iou'], c['bf2'], c['cldice'], c['connectivity'], c['topo_lc']) > 0.999
    assert c['curve_coverage'] == 1.0 and c['chamfer_distance'] < 1.0
    T3 = T.copy(); T3[0, 1] = 0
    s3 = SegMetrics(Nn); s3.update(lp, torch.tensor(T3), torch.tensor(Ln)); r3 = s3.compute()
    assert r3['macro']['n_classes_supported'] == 1.0 and r3['macro']['relaxed_f1'] > 0.999
    ok('perfect prediction scores 1; a class absent from the set is excluded from the macro mean')

    e = SegMetrics(Nn); e.update(torch.full((1, Nn, 48, 48), -20.0), torch.tensor(T), torch.tensor(Ln)); re = e.compute()['class_1']
    assert re['relaxed_recall'] == 0 and re['relaxed_f1'] == 0 and np.isnan(re['chamfer_distance']) and re['curve_coverage'] == 0
    ok('empty prediction: recall 0, chamfer undefined (NaN, not "perfect"), coverage 0')

    ln = np.zeros((10, 10), bool); ln[3, 0:8] = 1
    assert abs(kulpa_length_px(ln) - 0.948 * 7) < 1e-9
    ln = np.zeros((10, 10), bool); ln[0, 0] = ln[0, 1] = ln[1, 1] = 1
    assert abs(kulpa_length_px(ln) - 2 * 0.948) < 1e-9 and abs(kulpa_pixel_weights(ln).sum() - kulpa_length_px(ln)) < 1e-9
    ok('Kulpa length: straight line, no double counting at corners, pixel weights sum to length')

    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(0); a = rng.integers(0, 4, 400); b = np.clip(a + rng.integers(-1, 2, 400), 0, 3)
    D = np.zeros((4, 4))
    for i, j in zip(a, b):
        D[i, j] += 1
    assert abs(quadratic_weighted_kappa(D) - cohen_kappa_score(a, b, weights='quadratic')) < 1e-12
    ok('quadratic weighted kappa equals sklearn')

    C = A.reference_confusion(7)
    assert np.allclose(C.sum(1), 1) and all(abs(sum(C[i, j] for j in range(7) if abs(i - j) <= 1) - 0.63) < 1e-9 for i in range(7))
    ceil = A.achievable_ceiling(np.ones(7) / 7)
    assert 0.3 < ceil['exact'] < 0.34 and abs(ceil['within_one'] - 0.63) < 1e-9
    ok('reference confusion: rows sum to 1, exact=0.32, within-one=0.63; ceiling is analytic')

    cal = A.synthetic_calibration(T_px=5.0, n_trials=12, seed=1)
    assert 0.85 < cal['isolated']['coarea_ratio_mean'] < 1.15 and 0.9 < cal['isolated']['kulpa_ratio_mean'] < 1.1, cal
    log(f'      synthetic length calibration: {json.dumps(cal)}')
    ok('coarea-formula length estimator recovers the length of isolated synthetic lines (within 15%)')

    sds = [{'w': torch.full((2,), float(i)), 'n': torch.tensor(i)} for i in (1, 2, 6)]
    av = M.average_state_dicts(sds)
    assert torch.allclose(av['w'], torch.full((2,), 3.0)) and int(av['n']) == 6
    ok('soup averaging: uniform mean of float tensors, integer buffers kept')

    x1, m1 = torch.randn(1, 1, 64, 64), torch.randn(1, 6, 64, 64)
    for v in VARIANTS:
        net = build_model(v, 4, pretrained=False, img_size=64).eval()
        with torch.no_grad():
            assert net(x1, m1)['seg_logits'].shape == (1, 4, 64, 64)
        load_checkpoint(build_model(v, 4, pretrained=False, img_size=64), net.state_dict(), verbose=False)
        del net
    try:
        load_checkpoint(build_model('Simple', 4, pretrained=False, img_size=64), build_model('TTT', 4, pretrained=False, img_size=64).state_dict(), verbose=False)
        raise AssertionError('strict loading accepted a wrong-variant checkpoint')
    except RuntimeError:
        pass
    ok(f'all {len(VARIANTS)} model variants: forward shape, strict round-trip load, wrong-variant load is rejected')
    free()


# =========================================================================================
#  5. synthetic data for --smoke
# =========================================================================================
def make_synthetic_dataset(root: str, scenes=('SynA', 'SynB'), seed=0):
    from skimage.draw import line as sk_line
    from scipy import ndimage
    from dataset import _SUB_COORDS
    rng = np.random.default_rng(seed)
    for sc in scenes:
        d = os.path.join(root, sc); os.makedirs(os.path.join(d, 'npz'), exist_ok=True)
        rows = {'train': [], 'val': [], 'test': []}
        k = 0
        for split, n in (('train', 3), ('val', 1), ('test', 2)):
            for i in range(n):
                empty = split == 'train' and i == 0
                fname = f'{sc}_R{k:05d}_C00000.npz'; k += 1
                H = W = 512
                line = np.zeros((H, W), np.uint16)
                if not empty:
                    for _ in range(5):
                        p0 = rng.integers(30, 480, 2); a = rng.uniform(0, np.pi); ln = rng.uniform(80, 250)
                        p1 = np.clip((p0 + ln * np.array([np.sin(a), np.cos(a)])).astype(int), 30, 480)
                        rr, cc = sk_line(*p0, *p1); line[rr, cc] = rng.integers(1, 4)
                dist, ind = ndimage.distance_transform_edt(line == 0, return_indices=True)
                cls = line[ind[0], ind[1]]
                buf = np.where(dist <= 5, cls, 0).astype(np.uint16)
                ramp = np.where(dist <= 5, 5 - dist, 0).astype(np.float32)
                ms = (rng.integers(100, 3000, (6, H, W)) + (buf > 0) * 800).astype(np.uint16)
                dem = (rng.normal(0, 0.3, (1, H, W)) + (buf > 0) * 2.0).astype(np.float32)
                meta = dict(hghtcls_map=str({'1': 1, '2': 2, '3': 3}), dist_cap_m=5.0, pixel_size_x=1.0, pixel_size_y=1.0)
                np.savez_compressed(os.path.join(d, 'npz', fname), multispectral=ms, dem=dem, label_line=line,
                                    label_buffer=buf, label_distance=ramp, mask=np.zeros((H, W), bool),
                                    meta=np.array(meta, dtype=object))
                for sub in ('tl', 'ce'):
                    r0, r1, c0, c1 = _SUB_COORDS[sub]
                    lab = bool((line[r0:r1, c0:c1] > 0).any() or (buf[r0:r1, c0:c1] > 0).any())
                    for aug in ('non', 'hflip'):
                        rows[split].append(dict(Sample_ID=fname[:-4], SubPatch=sub, Rel_Path=f'{sc}/npz/{fname}',
                                                File_Has_Labels=int(not empty), is_empty=int(not lab), is_masked=0,
                                                Augmentation=aug))
        for split, r in rows.items():
            pd.DataFrame(r).to_csv(os.path.join(d, f'{split}_grouped.csv'), index=False)
        recs = [dict(source='multispectral', band=b + 1, mean=1500.0, std=800.0, min=0, max=4000, valid_px_count=1) for b in range(6)]
        recs.append(dict(source='dem', band=1, mean=0.0, std=1.0, min=-1, max=3, valid_px_count=1))
        pd.DataFrame(recs).to_csv(os.path.join(d, 'band_statistics.csv'), index=False)


# =========================================================================================
#  6. stages
# =========================================================================================
def stage_train(jobs):
    for i, j in enumerate(jobs, 1):
        if finished(j):
            log(f'[{i}/{len(jobs)}] {j.run_id}: already done'); continue
        log(f'\n[{i}/{len(jobs)}] TRAIN {j.run_id}  ({j.group})')
        safe(f'train {j.run_id}', M.train_run, j.cfg())
        free()


def _num_classes(cfg):
    return discover_num_classes(cfg['base_dir'], cfg['scenes'])


def stage_evaluate(jobs):
    cfg0 = jobs[0].cfg(); nc = _num_classes(cfg0)
    val_cfg = dict(cfg0); val_cfg.update(num_vis_samples=min(4, cfg0.get('num_vis_samples', 4)))
    test_loader = V.get_eval_loader(cfg0, 'test', nc)
    vis_loader = M.make_loader(cfg0, 'test', nc, batch_size=1, shuffle=True)
    loss_fn = M.loss_from_cfg(cfg0)
    for j in jobs:
        if not finished(j):
            log(f'skip eval {j.run_id}: not trained'); continue
        for w in EVAL_WEIGHTS:
            wp = weights_path(j, w)
            out = eval_dir(j, w)
            if wp is None:
                log(f'  {j.run_id}/{w}: weights not found, skipped'); continue
            if os.path.exists(os.path.join(out, 'metrics.json')):
                continue
            log(f'EVAL {j.run_id} / {w}')
            r = safe(f'eval {j.run_id}/{w}', V.evaluate_weights, j.cfg(), j.variant, wp, nc, test_loader, out,
                     loss_fn, vis_loader if w == PRIMARY_WEIGHTS and j.seed == cfg0['seed'] else None)
            if r is not None:
                s = r['summary']
                log(f'    macro F1={s["macro"]["relaxed_f1"]:.4f}  merged F1={s["merged"]["relaxed_f1"]:.4f}  '
                    f'clDice={s["merged"]["cldice"]:.4f}  chamfer={s["merged"]["chamfer_distance"]:.3f}px '
                    f'(coverage {s["merged"]["curve_coverage"]:.2f})')
                del r
            free()


def _load_model(j, w, nc, device):
    wp = weights_path(j, w)
    if wp is None:
        return None
    net = build_model(j.variant, nc, pretrained=False).to(device)
    load_checkpoint(net, wp, map_location=device, verbose=False)
    return net.eval()


def stage_perturb(jobs):
    cfg0 = jobs[0].cfg(); nc = _num_classes(cfg0)
    loader = V.get_eval_loader(cfg0, 'test', nc); loss_fn = M.loss_from_cfg(cfg0)
    outdir = os.path.join(OUT_ROOT, 'analysis', 'perturb'); os.makedirs(outdir, exist_ok=True)
    for j in jobs:
        if j.variant not in PERTURB_VARIANTS or j.group == 'loss_ablation' or not finished(j):
            continue
        out = os.path.join(outdir, f'{j.run_id}.json')
        if os.path.exists(out):
            continue
        net = _load_model(j, PRIMARY_WEIGHTS, nc, M.DEVICE)
        if net is None:
            continue
        log(f'PERTURB {j.run_id}')
        res = safe(f'perturb {j.run_id}', A.perturbation_eval, net, loader, cfg0, nc, M.DEVICE, PERTURBATIONS,
                   M.run_step, loss_fn)
        if res is not None:
            json.dump(M._jsonable(res), open(out, 'w'), indent=1)
        del net; free()


def stage_probe(jobs):
    import matplotlib.pyplot as plt
    cfg0 = jobs[0].cfg(); nc = _num_classes(cfg0)
    loader = V.get_eval_loader(cfg0, 'test', nc)
    outdir = os.path.join(OUT_ROOT, 'analysis', 'probe'); os.makedirs(outdir, exist_ok=True)
    for j in jobs:
        if j.variant not in PROBE_VARIANTS or j.group == 'loss_ablation' or not finished(j):
            continue
        out = os.path.join(outdir, f'{j.run_id}.csv')
        if os.path.exists(out):
            continue
        net = _load_model(j, PRIMARY_WEIGHTS, nc, M.DEVICE)
        if net is None:
            continue
        log(f'PROBE {j.run_id}')
        df = safe(f'probe {j.run_id}', A.fast_weight_probe, net, loader, cfg0, M.DEVICE, PROBE_MAX_PATCHES)
        if df is not None:
            df.to_csv(out, index=False)
            corr = A.probe_correlations(df)
            json.dump(corr, open(os.path.join(outdir, f'{j.run_id}_spearman.json'), 'w'), indent=2)
            cols = [c for c in df.columns if c.endswith('_d_linear_dem_reliance')]
            fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.6), squeeze=False)
            for ax, c in zip(axes[0], cols):
                ax.scatter(df['dem_at_hedge_median'], df[c], s=8, alpha=0.5)
                ax.set_xlabel('median raw DEM at hedge pixels'); ax.set_ylabel('DEM reliance'); ax.set_title(c.split('_')[0])
            fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{j.run_id}.png'), dpi=150); plt.close(fig)
        del net; free()


def stage_ceiling(jobs):
    """Ceiling from the reference class prior of the test set (row totals of the km confusion matrix)."""
    outdir = os.path.join(OUT_ROOT, 'analysis'); os.makedirs(outdir, exist_ok=True)
    prior, cm = None, None
    for j in jobs:
        p = os.path.join(eval_dir(j, PRIMARY_WEIGHTS), 'metrics.json')
        if os.path.exists(p):
            o = json.load(open(p))['ordinal']; prior = o['reference_km_per_class']; break
    if prior is None:
        log('ceiling: no evaluated run found (run the evaluate stage first)'); return
    mid = BASE.get('class_mid')
    ceil = A.achievable_ceiling(prior, mid)
    rows = []
    for j in jobs:
        p = os.path.join(eval_dir(j, PRIMARY_WEIGHTS), 'metrics.json')
        if os.path.exists(p):
            o = json.load(open(p))['ordinal']
            rows.append(dict(run=j.run_id, **{k: o[k] for k in ceil}, **{f'{k}__of_ceiling': o[k] / ceil[k] if ceil[k] else np.nan
                                                                         for k in ('exact', 'within_one', 'qwk', 'macro_f1_class')}))
    json.dump(dict(ceiling=ceil, reference_prior_km=prior), open(os.path.join(outdir, 'ceiling.json'), 'w'), indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'ordinal_vs_ceiling.csv'), index=False)
    log(f'ceiling (perfect model vs noisy labels): {json.dumps({k: round(v, 3) for k, v in ceil.items() if v == v})}')


def stage_complexity(jobs):
    cfg0 = jobs[0].cfg(); nc = _num_classes(cfg0)
    meta = A.scene_meta(cfg0); px = next(iter(meta.values()))['pixel_size_m'] if meta else 1.0
    df = A.complexity_report(lambda v: build_model(v, nc, pretrained=False), COMPLEXITY_VARIANTS, M.DEVICE,
                             img=cfg0['img_size'], px_m=px)
    os.makedirs(os.path.join(OUT_ROOT, 'analysis'), exist_ok=True)
    df.to_csv(os.path.join(OUT_ROOT, 'analysis', 'complexity.csv'), index=False)
    log(df.round(2).to_string(index=False))


def stage_estimators(jobs):
    cfg0 = jobs[0].cfg(); nc = _num_classes(cfg0)
    outdir = os.path.join(OUT_ROOT, 'analysis', 'estimators'); os.makedirs(outdir, exist_ok=True)
    cal_path = os.path.join(outdir, 'synthetic_calibration.json')
    if not os.path.exists(cal_path):
        meta = A.scene_meta(cfg0); T = next(iter(meta.values()))
        json.dump(A.synthetic_calibration(T['dist_cap_m'] / T['pixel_size_m']), open(cal_path, 'w'), indent=2)
    val_loader = V.get_eval_loader(cfg0, 'val', nc); test_loader = V.get_eval_loader(cfg0, 'test', nc)
    loss_fn = M.loss_from_cfg(cfg0)
    for j in jobs:
        if j.variant not in ESTIMATOR_VARIANTS or j.group == 'loss_ablation' or not finished(j):
            continue
        out = os.path.join(outdir, f'{j.run_id}.json')
        if os.path.exists(out):
            continue
        net = _load_model(j, PRIMARY_WEIGHTS, nc, M.DEVICE)
        if net is None:
            continue
        log(f'ESTIMATORS {j.run_id}')
        def go():
            cv = V.collect_predictions(net, val_loader, loss_fn); ct = V.collect_predictions(net, test_loader, loss_fn)
            return A.estimator_eval(cv, ct, cfg0, cfg0['seg_thresh'], ESTIMATOR_MATCH_TOL_M)
        res = safe(f'estimators {j.run_id}', go)
        if res is not None:
            res.pop('per_patch_test', None) if False else None
            json.dump(M._jsonable(res), open(out, 'w'), indent=1)
            for col in ('L_kulpa', 'L_coarea'):
                log(f'    {col}: {res[col]["km"]:.2f} km raw ({res[col]["rel_err_pct"]:+.1f}%), '
                    f'calibrated {res[col]["calibrated_km"]:.2f} km ({res[col]["calibrated_rel_err_pct"]:+.1f}%) '
                    f'vs reference {res["reference_km"]:.2f} km')
        del net; free()


def _per_patch(j, w=None):
    p = os.path.join(eval_dir(j, w or PRIMARY_WEIGHTS), 'per_patch.npz')
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def stage_stats(jobs):
    rows = []
    by = {(j.variant, j.seed): j for j in jobs if j.group != 'loss_ablation'}
    targets = list(STATS_TARGETS)
    for va, vb in STATS_PAIRS:
        seeds = sorted({s for (v, s) in by if v == va} & {s for (v, s) in by if v == vb})
        if not seeds:
            continue
        data = {}
        for s in seeds:
            pa, pb = _per_patch(by[(va, s)]), _per_patch(by[(vb, s)])
            if pa is None or pb is None:
                continue
            if list(pa['names']) != list(pb['names']):
                log(f'!! patch order differs between {va}_s{s} and {vb}_s{s}; skipped'); continue
            data[s] = (pa, pb)
        if not data:
            continue
        tg = targets + ([k for k in next(iter(data.values()))[0] if k.endswith('__relaxed_f1') and k.startswith('class_')]
                        if STATS_PER_CLASS_F1 else [])
        for t in tg:
            for s, (pa, pb) in data.items():
                rows.append(dict(A=va, B=vb, target=t, seed=s, **A.paired_stats(pa[t], pb[t])))
            if len(data) > 1:
                ma = np.nanmean([pa[t] for pa, _ in data.values()], axis=0)
                mb = np.nanmean([pb[t] for _, pb in data.values()], axis=0)
                rows.append(dict(A=va, B=vb, target=t, seed='mean_of_seeds', **A.paired_stats(ma, mb)))
    if rows:
        os.makedirs(os.path.join(OUT_ROOT, 'report'), exist_ok=True)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT_ROOT, 'report', 'paired_stats.csv'), index=False)
        log(df[df['target'].isin(targets)].round(4).to_string(index=False))


def stage_report(jobs):
    import matplotlib.pyplot as plt
    outdir = os.path.join(OUT_ROOT, 'report'); os.makedirs(outdir, exist_ok=True)
    recs = []
    for j in jobs:
        for w in EVAL_WEIGHTS:
            p = os.path.join(eval_dir(j, w), 'metrics.json')
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            r = dict(run=j.run_id, group=j.group, variant=j.variant, seed=j.seed, weights=w)
            for g in ('macro', 'merged'):
                for k in REPORT_METRICS:
                    r[f'{g}__{k}'] = d['summary'][g].get(k)
            for k in ('exact', 'within_one', 'class_mae', 'qwk', 'macro_f1_class', 'height_mae_m', 'detection_rate', 'reference_km'):
                r[f'ord__{k}'] = d['ordinal'].get(k)
            recs.append(r)
    if not recs:
        log('report: nothing evaluated yet'); return
    df = pd.DataFrame(recs); df.to_csv(os.path.join(outdir, 'all_runs.csv'), index=False)
    main_df = df[df['group'] != 'loss_ablation']
    val_cols = [c for c in df.columns if '__' in c]
    agg = main_df.groupby(['variant', 'weights'])[val_cols].agg(['mean', 'std', 'count'])
    agg.columns = [f'{a}|{b}' for a, b in agg.columns]
    agg.reset_index().to_csv(os.path.join(outdir, 'variants_mean_std.csv'), index=False)

    abl = df[(df['weights'] == PRIMARY_WEIGHTS)]
    full = abl[(abl['group'] == 'fusion_core') & (abl['run'].str.endswith(f'_s{jobs[0].cfg()["seed"]}')) & (abl['variant'] == 'TTT')]
    ab = abl[abl['group'] == 'loss_ablation']
    if len(full) and len(ab):
        base = full.iloc[0]
        delta = ab[['run']].copy()
        for c in val_cols:
            delta[f'd_{c}'] = ab[c].values - base[c]
        delta.to_csv(os.path.join(outdir, 'loss_ablation_deltas_vs_full.csv'), index=False)

    perturb = []
    for f in sorted(os.listdir(os.path.join(OUT_ROOT, 'analysis', 'perturb'))) if os.path.isdir(os.path.join(OUT_ROOT, 'analysis', 'perturb')) else []:
        d = json.load(open(os.path.join(OUT_ROOT, 'analysis', 'perturb', f)))
        run = f[:-5]; clean = d.get('clean', {}).get('macro', {}).get('relaxed_f1')
        for pn, res in d.items():
            row = dict(run=run, perturbation=pn, macro_f1=res['macro']['relaxed_f1'], merged_f1=res['merged']['relaxed_f1'],
                       macro_f1_drop=(clean - res['macro']['relaxed_f1']) if clean is not None else None)
            for k, v in res.items():
                if k.startswith('class_'):
                    row[f'{k}_f1'] = v['relaxed_f1']
            perturb.append(row)
    if perturb:
        pd.DataFrame(perturb).to_csv(os.path.join(outdir, 'perturbation_results.csv'), index=False)

    with pd.ExcelWriter(os.path.join(outdir, 'summary.xlsx'), engine='openpyxl') as w:
        df.to_excel(w, sheet_name='all_runs', index=False)
        agg.reset_index().to_excel(w, sheet_name='variants_mean_std', index=False)
        if len(full) and len(ab):
            delta.to_excel(w, sheet_name='loss_ablation', index=False)
        if perturb:
            pd.DataFrame(perturb).to_excel(w, sheet_name='perturbation', index=False)
    pv = main_df[main_df['weights'] == PRIMARY_WEIGHTS]
    if len(pv):
        fig, ax = plt.subplots(figsize=(1.3 * pv['variant'].nunique() + 3, 4))
        for i, (v, g) in enumerate(pv.groupby('variant', sort=False)):
            ax.bar(i, g['macro__relaxed_f1'].mean(), yerr=g['macro__relaxed_f1'].std() if len(g) > 1 else None, alpha=0.7, capsize=4)
            ax.scatter([i] * len(g), g['macro__relaxed_f1'], color='k', s=12, zorder=3)
        ax.set_xticks(range(pv['variant'].nunique())); ax.set_xticklabels(list(pv.groupby('variant', sort=False).groups), rotation=30, ha='right')
        ax.set_ylabel('macro relaxed F1 (test)'); ax.set_title(f'Variants ({PRIMARY_WEIGHTS}; bars = mean±std over seeds, dots = seeds)')
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'variants_macro_f1.png'), dpi=200); plt.close(fig)
    log(f'report written to {outdir}')


# =========================================================================================
#  7. main
# =========================================================================================
def print_plan(jobs, stages):
    log(f'\nOutput root : {OUT_ROOT}')
    log('Stages      : ' + ', '.join(f'{k}={"ON" if v else "off"}' for k, v in stages.items()))
    log('Groups      : ' + ', '.join(f'{k}={"ON" if v.get("enabled") else "off"}' for k, v in GROUPS.items()))
    log(f'Jobs        : {len(jobs)}')
    for j in jobs:
        st = 'done' if finished(j) else ('resumable' if os.path.exists(os.path.join(j.run_dir(), 'last.pt')) else 'todo')
        log(f'   {j.run_id:<34} {j.group:<14} {st:<9} {j.overrides if j.overrides else ""}')
    log(f'Eval weights: {EVAL_WEIGHTS}   (primary: {PRIMARY_WEIGHTS})')


def main():
    global OUT_ROOT, _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=''); ap.add_argument('--skip', default='')
    ap.add_argument('--groups', default=''); ap.add_argument('--seeds', default='')
    ap.add_argument('--epochs', type=int); ap.add_argument('--out')
    ap.add_argument('--list', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--no-selftest', action='store_true')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override BASE keys (JSON values)')
    a = ap.parse_args()

    if a.out:
        OUT_ROOT = a.out
    for kv in a.set:
        k, v = kv.split('=', 1); BASE[k] = M._parse_value(v)
    if a.epochs:
        BASE['epochs'] = a.epochs
    if a.smoke:
        OUT_ROOT = a.out or 'smoke_experiments'
        root = os.path.join(OUT_ROOT, 'synthetic_data')
        if not os.path.exists(os.path.join(root, 'SynA', 'test_grouped.csv')):
            make_synthetic_dataset(root)
        BASE.update(base_dir=root, scenes=['SynA', 'SynB'], epochs=a.epochs or 2, batch_size=2, pretrained=False,
                    keep_last_k=2, save_last_every=1, vis_every=0, threshold_sweep=[0.2, 0.5], cldice_iters=4,
                    num_vis_samples=2, class_mid=[1.0, 2.0, 3.0])
        PROBE_MAX_PATCHES_LOCAL = 4
        globals()['PROBE_MAX_PATCHES'] = PROBE_MAX_PATCHES_LOCAL
    os.makedirs(OUT_ROOT, exist_ok=True)
    _LOG = os.path.join(OUT_ROOT, 'runner.log')

    stages = dict(STAGES)
    if a.only:
        on = set(a.only.split(','))
        stages = {k: (k in on) for k in stages}
    for s in filter(None, a.skip.split(',')):
        stages[s] = False
    if a.no_selftest:
        stages['selftest'] = False
    gf = set(filter(None, a.groups.split(','))) or None
    seeds = [int(s) for s in a.seeds.split(',')] if a.seeds else None
    jobs = build_jobs(gf, seeds)
    if not jobs:
        log('no jobs: everything is disabled'); return
    print_plan(jobs, stages)
    if a.list or a.dry_run:
        return

    t0 = time.time()
    log(f'device: {M.DEVICE}')
    if stages['selftest'] and not stage('selftest', selftest):
        log('SELF-TEST FAILED -- refusing to start the experiments (use --no-selftest to override).'); sys.exit(1)
    if stages['train']:      stage('train', stage_train, jobs)
    if stages['evaluate']:   stage('evaluate', stage_evaluate, jobs)
    if stages['perturb']:    stage('perturb', stage_perturb, jobs)
    if stages['probe']:      stage('probe', stage_probe, jobs)
    if stages['ceiling']:    stage('ceiling', stage_ceiling, jobs)
    if stages['complexity']: stage('complexity', stage_complexity, jobs)
    if stages['estimators']: stage('estimators', stage_estimators, jobs)
    if stages['stats']:      stage('stats', stage_stats, jobs)
    if stages['report']:     stage('report', stage_report, jobs)
    log(f'\nALL DONE in {(time.time() - t0) / 3600:.2f} h.  Failed jobs: {FAILURES if FAILURES else "none"}')


if __name__ == '__main__':
    main()
