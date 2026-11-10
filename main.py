import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import copy
import csv
import glob
import json
import random
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import HedgeDistanceDataset, discover_num_classes
from model import build_model, load_checkpoint, VARIANTS
from metrics import SegMetrics, soft_cldice_loss

# =========================================================================================
#  CONFIG  (one dict per run; run_all.py deep-copies this and overrides keys)
# =========================================================================================
CFG = dict(
    base_dir    = r'/home/basir/Hedge/data/',
    scenes      = ['Scene1', 'Scene2'],   # each scene: base_dir/<scene>/{npz,band_statistics.csv,*_grouped.csv}
    run_dir     = 'runs/TTT_s42',         # everything of one run lives here
    variant     = 'TTT',                  # Simple | SRA | TTT | Early | Late | DEM_only | MS_only
    pretrained  = True,                   # ImageNet weights for the MiT-B5 encoders

    img_size = 256,
    batch_size  = 4,
    num_workers = 2,
    epochs      = 100,
    lr          = 1e-4,
    weight_decay= 1e-4,
    encoder_lr_mult = 1.0,

    # loss
    w_mse           = 0.5,
    w_mae           = 0.5,
    w_cldice        = 0.5,
    w_merge         = 0.5,   # weight of the "merged classes -> scenario 1" consistency loss
    skeleton_boost  = 3.0,
    margin          = 0.05,
    pu_threshold    = 0.85,  # >1 disables the PU masking
    cldice_mode     = 'fixed',   # 'fixed' | 'conventional'
    cldice_iters    = 20,

    # metrics
    seg_thresh       = 0.30,
    gt_thresh        = 0.01,
    metric_tolerance = 3,
    bf2_tolerance    = 2,
    val_topology     = True,     # connectivity/Chamfer/HD95 every epoch (needed for best_chamfer / best_hd95)
    threshold_sweep  = [round(float(t), 2) for t in np.arange(0.05, 0.96, 0.05)],

    # data selection
    train_empty_frac  = 0.25,    # fraction of empty training rows kept (sampled with the RUN seed)
    eval_empty_frac   = 1.0,     # fraction of empty val/test rows kept
    eval_non_aug_only = True,    # val/test: only Augmentation=='non' rows (the 5 other rows are rotated/flipped copies)

    # checkpoints / soups
    ckpt_criteria     = ['f1', 'rmse', 'chamfer', 'hd95'],   # which best_<name>.pth files to keep
    keep_epoch_ckpts  = 'last_k',   # 'none' | 'last_k' | 'all'
    keep_last_k       = 10,
    save_last_every   = 5,          # resume file every n epochs (0 = no resume file)
    vis_every         = 10,
    make_soups        = True,       # greedy + uniform model soup after training (Wortsman et al., ICML 2022)
    soup_use_epoch_ckpts = True,

    use_ema   = True,
    ema_decay = 0.999,
    seed      = 42,
    force     = False,              # retrain even if run_dir/done.json exists

    jitter_cfg = dict(
        p          = 0.5,
        bias_range = 20.0,
        gain_range = 0.08,
        noise_std  = 10.0,
        dropout_p  = 0.0,
        dem_noise_std = 0.5,
        swap_p     = 0.0,
    ),
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.ema   = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.n_skipped_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Blend model weights into the EMA.  Non-finite tensors are skipped so one bad step can never
        permanently poison the EMA (d*ema + (1-d)*NaN == NaN forever)."""
        d   = self.decay
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            mv = msd[k]
            if ema_v.is_floating_point():
                if not torch.isfinite(mv).all():
                    self.n_skipped_updates += 1
                    continue
                fp = ema_v.float().mul_(d).add_(mv.float(), alpha=1.0 - d)
                ema_v.copy_(fp.to(ema_v.dtype))
            else:
                ema_v.copy_(mv)


# =========================================================================================
#  LOSS
# =========================================================================================
class LOSS(nn.Module):
    """
    For every class channel c (pred p_c = sigmoid(logit_c), target t_c in [0,1]):

        e      = max(|p - t| - margin, 0)                       dead-zone absolute error
        w      = (1 + skeleton_boost * S_c) * valid * (1 - PU)  PU = 1 where t<0.01 and p>pu_threshold
        MAE_c  = sum(e   * w) / max(sum(w), 1)                  weighted means, per class
        MSE_c  = sum(e^2 * w) / max(sum(w), 1)
        per-class term = w_mse*mean_c(MSE_c) + w_mae*mean_c(MAE_c) + w_cldice*clDice_loss

    The same formula is applied to the merged maps  min(sum_c p_c, 1)  and  min(sum_c t_c, 1)
    (scenario-1 consistency) and added with weight w_merge.  valid masks nodata for ALL terms
    (clDice included -- previously it ignored the mask).
    """

    def __init__(self, w_mse: float = 0.5, w_mae: float = 0.5, w_cldice: float = 0.5, w_merge: float = 0.5,
                 skeleton_boost: float = 3.0, margin: float = 0.05, pu_threshold: float = 0.85,
                 cldice_iters: int = 20, cldice_mode: str = 'fixed'):
        super().__init__()
        self.w_mse = w_mse
        self.w_mae = w_mae
        self.w_cldice = w_cldice
        self.w_merge = w_merge
        self.skeleton_boost = skeleton_boost
        self.margin = margin
        self.pu_threshold = pu_threshold
        self.cldice_iters = cldice_iters
        self.cldice_mode = cldice_mode

    def _weighted_mse_mae(self, pred, target, valid_mask, skeleton):
        abs_error = torch.abs(pred - target)
        margin_error = torch.clamp(abs_error - self.margin, min=0.0)

        spatial_weight = torch.ones_like(target)
        if skeleton is not None:
            spatial_weight = spatial_weight + (skeleton * self.skeleton_boost)

        missing_gt_mask = (target < 0.01) & (pred > self.pu_threshold)
        spatial_weight = spatial_weight.masked_fill(missing_gt_mask, 0.0)

        if valid_mask is not None:
            spatial_weight = spatial_weight * valid_mask

        weighted_mae = margin_error * spatial_weight
        weighted_mse = (margin_error ** 2) * spatial_weight

        norm = torch.clamp(spatial_weight.sum(dim=(0, 2, 3)), min=1.0)
        return (weighted_mse.sum(dim=(0, 2, 3)) / norm).mean(), (weighted_mae.sum(dim=(0, 2, 3)) / norm).mean()

    def _cldice(self, pred, target, valid_mask):
        if self.w_cldice == 0.0:
            return pred.new_zeros(())
        return soft_cldice_loss(pred, target, iters=self.cldice_iters, mode=self.cldice_mode, valid=valid_mask)

    def forward(self, logits, target, valid_mask=None, skeleton=None):
        pred = torch.sigmoid(logits)

        mse_loss, mae_loss = self._weighted_mse_mae(pred, target, valid_mask, skeleton)
        cldice_loss = self._cldice(pred, target, valid_mask)
        per_class_total = self.w_mse * mse_loss + self.w_mae * mae_loss + self.w_cldice * cldice_loss

        total = per_class_total
        m_mse = m_mae = m_cldice = merge_total = pred.new_zeros(())
        if self.w_merge != 0.0:
            merged_pred   = torch.clamp(pred.sum(dim=1, keepdim=True), 0.0, 1.0)
            merged_target = torch.clamp(target.sum(dim=1, keepdim=True), 0.0, 1.0)
            merged_skel   = torch.clamp(skeleton.sum(dim=1, keepdim=True), 0.0, 1.0) if skeleton is not None else None
            m_mse, m_mae = self._weighted_mse_mae(merged_pred, merged_target, valid_mask, merged_skel)
            m_cldice = self._cldice(merged_pred, merged_target, valid_mask)
            merge_total = self.w_mse * m_mse + self.w_mae * m_mae + self.w_cldice * m_cldice
            total = per_class_total + self.w_merge * merge_total

        breakdown = dict(
            mse=mse_loss.item(), mae=mae_loss.item(), cldice=cldice_loss.item(),
            merge_mse=m_mse.item(), merge_mae=m_mae.item(), merge_cldice=m_cldice.item(),
            merge_total=merge_total.item(), total=total.item(),
        )
        return total, breakdown


def loss_from_cfg(cfg: dict) -> LOSS:
    return LOSS(
        w_mse=cfg['w_mse'], w_mae=cfg['w_mae'], w_cldice=cfg['w_cldice'], w_merge=cfg['w_merge'],
        skeleton_boost=cfg['skeleton_boost'], margin=cfg['margin'], pu_threshold=cfg['pu_threshold'],
        cldice_iters=cfg.get('cldice_iters', 20), cldice_mode=cfg.get('cldice_mode', 'fixed'),
    )


def metrics_from_cfg(cfg: dict, num_classes: int, topology=None, keep_per_patch=False, thresh=None) -> SegMetrics:
    return SegMetrics(
        num_classes   = num_classes,
        thresh        = cfg['seg_thresh'] if thresh is None else thresh,
        tolerance     = cfg['metric_tolerance'],
        bf2_tolerance = cfg['bf2_tolerance'],
        gt_thresh     = cfg.get('gt_thresh', 0.01),
        topology      = cfg.get('val_topology', True) if topology is None else topology,
        keep_per_patch= keep_per_patch,
    )


# =========================================================================================
#  DATA
# =========================================================================================
def _load_csv(splits_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(splits_dir, f'{split}_grouped.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV not found: {path}')
    df = pd.read_csv(path)
    print(f'  [{split}] {len(df):>7} rows  →  {path}')
    return df


def build_split_df(cfg: dict, split: str):
    dfs = []
    for scene in cfg['scenes']:
        try:
            dfs.append(_load_csv(os.path.join(cfg['base_dir'], scene), split))
        except FileNotFoundError as exc:
            print(f'  WARNING: {exc}  — skipping scene "{scene}" for split "{split}"')
    if not dfs:
        return None
    df = pd.concat(dfs, axis=0, ignore_index=True)

    if split != 'train' and cfg.get('eval_non_aug_only', True) and 'Augmentation' in df.columns:
        n0 = len(df)
        df = df[df['Augmentation'] == 'non'].reset_index(drop=True)
        print(f'  [{split}] evaluation uses un-augmented rows only: {n0} -> {len(df)}')

    frac = cfg.get('train_empty_frac', 0.25) if split == 'train' else cfg.get('eval_empty_frac', 1.0)
    labelled   = df[df['is_empty'] == 0]
    unlabelled = df[df['is_empty'] == 1]
    if frac < 1.0:
        unlabelled = unlabelled.sample(frac=frac, random_state=cfg['seed'])
    return pd.concat([unlabelled, labelled], axis=0).reset_index(drop=True)


def make_loader(cfg: dict, split: str, num_classes: int, batch_size=None, shuffle=None, seed=None):
    df = build_split_df(cfg, split)
    if df is None:
        return None
    train = split == 'train'
    ds = HedgeDistanceDataset(
        base_dir=cfg['base_dir'], dataframe=df, num_classes=num_classes, split=split,
        img_size=cfg['img_size'], jitter_cfg=cfg.get('jitter_cfg'),
    )
    shuffle = train if shuffle is None else shuffle
    gen = torch.Generator().manual_seed(cfg['seed'] if seed is None else seed)
    nw = cfg['num_workers']
    loader = DataLoader(
        ds, batch_size=batch_size or cfg['batch_size'], shuffle=shuffle, num_workers=nw,
        pin_memory=DEVICE.type == 'cuda', drop_last=train, generator=gen if shuffle else None,
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
    )
    print(f'  [{split}] {len(ds):>6} samples  |  {len(loader)} batches')
    return loader


def get_loaders(cfg: dict, num_classes: int):
    return (make_loader(cfg, 'train', num_classes),
            make_loader(cfg, 'val', num_classes, shuffle=False),
            make_loader(cfg, 'test', num_classes, shuffle=False))


# =========================================================================================
#  STEPS
# =========================================================================================
def run_step(ms, dem, dist_gt, mask, skeleton, model, loss_fn):
    """Returns loss, breakdown, logits, dist_gt, skeleton, valid  (valid = 1 - nodata mask, (B,1,H,W))."""
    if dist_gt.ndim == 3:
        dist_gt = dist_gt.unsqueeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if skeleton.ndim == 3:
        skeleton = skeleton.unsqueeze(1)

    outputs = model(dem, ms)
    logits  = outputs['seg_logits']
    if logits.shape[-2:] != dist_gt.shape[-2:]:
        dist_gt  = F.interpolate(dist_gt,  size=logits.shape[-2:], mode='bilinear', align_corners=False)
        mask     = F.interpolate(mask,     size=logits.shape[-2:], mode='nearest')
        skeleton = F.interpolate(skeleton, size=logits.shape[-2:], mode='nearest')
    valid = 1.0 - mask
    loss, breakdown = loss_fn(logits, dist_gt, valid, skeleton)
    return loss, breakdown, logits, dist_gt, skeleton, valid


def train_epoch(loader, model, optimizer, loss_fn, scaler, epoch, ema=None):
    model.train()
    running = 0.0
    breakdown_sum = {}
    n_samples = 0
    n_skipped_batches = 0
    t0 = time.time()
    pbar = tqdm(loader, desc=f'Train {epoch:03d}', leave=False, dynamic_ncols=True)

    for i, (ms, dem, dist_gt, mask, skeleton, _) in enumerate(pbar):
        mask = mask.to(DEVICE)
        ms, dem, dist_gt, skeleton = ms.to(DEVICE), dem.to(DEVICE), dist_gt.to(DEVICE), skeleton.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            loss, breakdown, logits, _, _, _ = run_step(ms, dem, dist_gt, mask, skeleton, model, loss_fn)

        # GradScaler only guards non-finite GRADIENTS; a non-finite loss/logits must be caught here,
        # otherwise ema.update() would absorb NaN permanently.
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            n_skipped_batches += 1
            print(f'\n  !! non-finite loss/logits at epoch {epoch} batch {i} — skipping batch '
                  f'(total skipped this epoch: {n_skipped_batches})')
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        params_finite = all(torch.isfinite(p).all() for p in model.parameters())
        if ema is not None and params_finite:
            ema.update(model)
        elif ema is not None:
            n_skipped_batches += 1
            print(f'\n  !! non-finite parameters after step at epoch {epoch} batch {i} — EMA update skipped.')

        running += loss.item()
        for k, v in breakdown.items():
            breakdown_sum[k] = breakdown_sum.get(k, 0.0) + v
        n_samples += ms.shape[0]
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'skipped': n_skipped_batches})

    epoch_time = time.time() - t0
    n_counted  = max(len(loader) - n_skipped_batches, 1)
    avg_breakdown = {k: v / n_counted for k, v in breakdown_sum.items()}
    if n_skipped_batches:
        print(f'  WARNING: skipped {n_skipped_batches}/{len(loader)} batches this epoch.')
    return running / n_counted, avg_breakdown, epoch_time, n_samples, n_skipped_batches


@torch.no_grad()
def eval_epoch(loader, model, loss_fn, metrics: SegMetrics, epoch, tag='Val', verbose=True):
    model.eval()
    metrics.reset()
    running = 0.0
    breakdown_sum = {}
    n_samples = 0
    t0 = time.time()
    pbar = tqdm(loader, desc=f'{tag}   {epoch:03d}', leave=False, dynamic_ncols=True)

    for ms, dem, dist_gt, mask, skeleton, names in pbar:
        mask = mask.to(DEVICE)
        ms, dem, dist_gt, skeleton = ms.to(DEVICE), dem.to(DEVICE), dist_gt.to(DEVICE), skeleton.to(DEVICE)
        loss, breakdown, logits, dist_gt, skeleton, valid = run_step(ms, dem, dist_gt, mask, skeleton, model, loss_fn)
        running += loss.item()
        for k, v in breakdown.items():
            breakdown_sum[k] = breakdown_sum.get(k, 0.0) + v
        n_samples += ms.shape[0]
        metrics.update(logits, dist_gt, skeleton, valid, names)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_time    = time.time() - t0
    n_batches     = max(len(loader), 1)
    avg           = running / n_batches
    avg_breakdown = {k: v / n_batches for k, v in breakdown_sum.items()}
    res = metrics.compute()
    if verbose:
        macro, merged = res['macro'], res['merged']
        print(f'\n[Epoch {epoch} {tag}]  loss={avg:.4f}')
        for lbl, g in (('macro (supported classes)', macro), ('merged (scenario 1)', merged)):
            print(f'  [{lbl:<25}] F1={g["relaxed_f1"]:.4f} IoU={g["relaxed_iou"]:.4f} BF={g["bf2"]:.4f} '
                  f'clDice={g["cldice"]:.4f} Conn={g["connectivity"]:.4f} TopoLc={g["topo_lc"]:.4f} '
                  f'Chamfer={g["chamfer_distance"]:.3f} HD95={g["hausdorff95_distance"]:.3f} '
                  f'cov={g["curve_coverage"]:.3f} RMSE={g["active_rmse"]:.4f}')
        for i in range(metrics.N):
            c = res[f'class_{i + 1}']
            print(f'    class_{i + 1}: gt_px={c["gt_px"]:.0f} F1={c["relaxed_f1"]:.4f} clDice={c["cldice"]:.4f} '
                  f'BF={c["bf2"]:.4f} Conn={c["connectivity"]:.4f}')
    return avg, avg_breakdown, res, epoch_time, n_samples


# =========================================================================================
#  MODEL SOUPS  (Wortsman et al., "Model soups", ICML 2022)
# =========================================================================================
def average_state_dicts(state_dicts: list) -> dict:
    """Uniform average of floating tensors; integer buffers (e.g. BN num_batches_tracked) from the last one."""
    out = {}
    n = len(state_dicts)
    for k in state_dicts[0].keys():
        if state_dicts[0][k].is_floating_point():
            acc = sum(sd[k].double() for sd in state_dicts) / n
            out[k] = acc.to(state_dicts[0][k].dtype)
        else:
            out[k] = state_dicts[-1][k].clone()
    return out


@torch.no_grad()
def score_state_dict(model, sd, loader, cfg, num_classes, loss_fn) -> float:
    """Validation macro relaxed-F1 of a state dict (fast metrics only, no topology)."""
    load_checkpoint(model, sd, verbose=False)
    m = metrics_from_cfg(cfg, num_classes, topology=False)
    _, _, res, _, _ = eval_epoch(loader, model, loss_fn, m, epoch=0, tag='Soup', verbose=False)
    return float(res['macro']['relaxed_f1'])


def build_soups(cfg: dict, num_classes: int, val_loader, run_dir: str = None) -> dict:
    """
    uniform soup : mean of ALL ingredients.
    greedy soup  : sort ingredients by validation score (best first); start from the best; add an ingredient
                   only if the average of (current soup + candidate) is NOT worse on validation (Alg. 1 of
                   the paper).  Scored on macro relaxed-F1 of the VALIDATION set -- the test set is never seen.
    Ingredients  : every best_<criterion>.pth of the run (+ the rolling last-k epoch checkpoints).
    Caveat       : soups only make sense between checkpoints of ONE run (same initialisation / basin).
                   Averaging different seeds (different random decoder/fusion init) is not meaningful.
    """
    run_dir = run_dir or cfg['run_dir']
    paths = sorted(glob.glob(os.path.join(run_dir, 'best_*.pth')))
    if cfg.get('soup_use_epoch_ckpts', True):
        paths += sorted(glob.glob(os.path.join(run_dir, 'epoch_*.pth')))
    paths = list(dict.fromkeys(paths))
    if len(paths) < 2:
        print('  soups: fewer than 2 ingredients on disk — skipped.')
        return {}

    model = build_model(cfg['variant'], num_classes, pretrained=False).to(DEVICE)
    loss_fn = loss_from_cfg(cfg)
    scores = {}
    for p in paths:
        sd = torch.load(p, map_location='cpu')
        scores[p] = score_state_dict(model, sd, val_loader, cfg, num_classes, loss_fn)
        print(f'  soup ingredient {os.path.basename(p):<28} val macro-F1 = {scores[p]:.4f}')
        del sd
    order = sorted(paths, key=lambda p: -scores[p])

    # greedy soup
    chosen = [order[0]]
    cur_sds = [torch.load(order[0], map_location='cpu')]
    cur_score = scores[order[0]]
    log = [dict(path=order[0], accepted=True, val_f1=cur_score)]
    for p in order[1:]:
        cand = torch.load(p, map_location='cpu')
        avg = average_state_dicts(cur_sds + [cand])
        s = score_state_dict(model, avg, val_loader, cfg, num_classes, loss_fn)
        ok = s >= cur_score
        log.append(dict(path=p, accepted=bool(ok), val_f1=s))
        print(f'  greedy: + {os.path.basename(p):<28} -> {s:.4f} (current {cur_score:.4f}) '
              f'{"ACCEPT" if ok else "reject"}')
        if ok:
            cur_sds.append(cand); chosen.append(p); cur_score = s
        del avg
    greedy_sd = average_state_dicts(cur_sds)
    torch.save(greedy_sd, os.path.join(run_dir, 'soup_greedy.pth'))
    del cur_sds

    # uniform soup
    sds = [torch.load(p, map_location='cpu') for p in order]
    uni_sd = average_state_dicts(sds)
    del sds
    uni_score = score_state_dict(model, uni_sd, val_loader, cfg, num_classes, loss_fn)
    torch.save(uni_sd, os.path.join(run_dir, 'soup_uniform.pth'))

    report = dict(ingredient_val_f1={os.path.basename(p): scores[p] for p in order},
                  greedy_chosen=[os.path.basename(p) for p in chosen], greedy_val_f1=cur_score,
                  greedy_log=[dict(l, path=os.path.basename(l['path'])) for l in log],
                  uniform_val_f1=uni_score, best_single_val_f1=scores[order[0]])
    with open(os.path.join(run_dir, 'soup_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  soups: best single {scores[order[0]]:.4f} | greedy {cur_score:.4f} '
          f'({len(chosen)} ingredients) | uniform {uni_score:.4f}')
    return report


# =========================================================================================
#  PLOTS / VISUALISATION
# =========================================================================================
@torch.no_grad()
def visualise(model, loader, loss_fn, thresh, save_path, epoch):
    model.eval()
    ms_b, dem_b, dist_gt_b, mask_b, skeleton_b, names = next(iter(loader))
    b = 0
    ms_b, dem_b, dist_gt_b, mask_b, skeleton_b = (
        ms_b.to(DEVICE), dem_b.to(DEVICE), dist_gt_b.to(DEVICE), mask_b.to(DEVICE), skeleton_b.to(DEVICE))
    _, _, logits, dist_gt_b, skeleton_b, _ = run_step(ms_b, dem_b, dist_gt_b, mask_b, skeleton_b, model, loss_fn)

    pred_map = np.clip(torch.sigmoid(logits[b]).cpu().numpy().sum(axis=0), 0.0, 1.0)
    gt_map   = np.clip(dist_gt_b[b].cpu().numpy().sum(axis=0), 0.0, 1.0)
    bin_pred = (pred_map >= thresh).astype(np.float32)
    bin_gt   = (gt_map > 0.0).astype(np.float32)

    def _norm(t):
        arr = t.cpu().numpy()
        out = np.zeros_like(arr, dtype=np.float32)
        for c in range(arr.shape[0]):
            lo, hi = np.percentile(arr[c], 2), np.percentile(arr[c], 98)
            out[c] = np.clip((arr[c] - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0
        return out.transpose(1, 2, 0)

    fig, axs = plt.subplots(1, 6, figsize=(18, 3))
    fig.suptitle(f'Epoch {epoch} | {names[b]}', fontsize=8)
    titles = ['MS (bands 1-3)', 'DEM', 'GT heatmap', 'Pred heatmap', 'GT binary', 'Pred binary']
    imgs   = [_norm(ms_b[b, :3]), dem_b.cpu().numpy()[b, 0], gt_map, pred_map, bin_gt, bin_pred]
    cmaps  = [None, None, 'magma', 'magma', 'gray', 'gray']
    for ax, img, title, cmap in zip(axs, imgs, titles, cmaps):
        if title in ('MS (bands 1-3)', 'DEM'):
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=7)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_history(history, save_dir):
    if not history['train_loss']:
        return
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(ep, history['train_loss'], label='train')
    if history['val_loss']:
        axes[0].plot(ep, history['val_loss'], label='val')
    axes[0].set_title('Loss'); axes[0].legend()
    M = history['metrics']
    if M:
        for key in ('active_rmse', 'active_mae'):
            axes[1].plot(ep, [m['macro'][key] for m in M], label=f'macro_{key}')
            axes[1].plot(ep, [m['merged'][key] for m in M], '--', label=f'merged_{key}')
        axes[1].set_title('Regression errors'); axes[1].legend(fontsize=6)
        for key in ('relaxed_f1', 'relaxed_iou', 'cldice'):
            axes[2].plot(ep, [m['macro'][key] for m in M], label=f'macro_{key}')
            axes[2].plot(ep, [m['merged'][key] for m in M], '--', label=f'merged_{key}')
        axes[2].set_title(f'Binary metrics (thr={history["thresh"]})'); axes[2].legend(fontsize=6)
        for key in ('chamfer_distance', 'hausdorff95_distance'):
            axes[3].plot(ep, [m['merged'][key] for m in M], label=key)
        axes[3].set_title('Curve metrics (px, merged)'); axes[3].legend()
    for a in axes:
        a.set_xlabel('Epoch')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'history.png'), dpi=120); plt.close()


def _plot_validation_curves(history, save_dir):
    M = history['metrics']
    if not M:
        return
    names = ['relaxed_f1', 'cldice', 'relaxed_iou', 'bf2', 'connectivity', 'topo_lc']
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in names:
        ax.plot(range(1, len(M) + 1), [m['macro'][n] for m in M], label=f'macro_{n}')
    ax.set_title('Validation score vs epoch (macro across supported classes)')
    ax.set_xlabel('Epoch'); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'validation_curves.png'), dpi=120); plt.close()


def _plot_class_bars(history, save_dir):
    if not history['metrics']:
        return
    last = history['metrics'][-1]
    ck = sorted(k for k in last if k.startswith('class_'))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ck)); metrics = ['relaxed_f1', 'bf2', 'cldice', 'connectivity', 'topo_lc']
    w = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        axes[0].bar(x + i * w, [last[c][m] for c in ck], width=w, label=m)
    axes[0].set_xticks(x + w * (len(metrics) - 1) / 2); axes[0].set_xticklabels(ck); axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7); axes[0].set_title('Scores by class (last epoch)')
    axes[1].bar(x, [last[c]['chamfer_distance'] for c in ck], color='indianred')
    axes[1].set_xticks(x); axes[1].set_xticklabels(ck); axes[1].set_title('Chamfer by class (px, last epoch)')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'topology_metrics_by_class.png'), dpi=120); plt.close()


@torch.no_grad()
def threshold_sensitivity_diagnostic(model, loader, loss_fn, thresholds, save_dir):
    """Exact (tolerance-free) merged precision/recall/F1/IoU vs threshold; every count is consistent."""
    model.eval()
    tp = {t: 0.0 for t in thresholds}; fp = {t: 0.0 for t in thresholds}; fn = {t: 0.0 for t in thresholds}
    for ms, dem, dist_gt, mask, skeleton, _ in tqdm(loader, desc='Threshold sweep', leave=False):
        mask = mask.to(DEVICE)
        ms, dem, dist_gt, skeleton = ms.to(DEVICE), dem.to(DEVICE), dist_gt.to(DEVICE), skeleton.to(DEVICE)
        _, _, logits, dist_gt, _, valid = run_step(ms, dem, dist_gt, mask, skeleton, model, loss_fn)
        merged_pred = torch.clamp(torch.sigmoid(logits).sum(dim=1, keepdim=True), 0.0, 1.0) * valid
        tgt_bin = (torch.clamp(dist_gt.sum(dim=1, keepdim=True), 0.0, 1.0) > 0.01).float() * valid
        for t in thresholds:
            pb = (merged_pred >= t).float()
            tp[t] += (pb * tgt_bin).sum().item()
            fp[t] += (pb * (1.0 - tgt_bin)).sum().item()
            fn[t] += (tgt_bin * (1.0 - pb)).sum().item()
    P, R, F, I = [], [], [], []
    for t in thresholds:
        p = tp[t] / max(tp[t] + fp[t], 1.0); r = tp[t] / max(tp[t] + fn[t], 1.0)
        P.append(p); R.append(r); F.append(2 * p * r / max(p + r, 1e-8)); I.append(tp[t] / max(tp[t] + fp[t] + fn[t], 1.0))
    best_t = thresholds[int(np.argmax(F))]
    fig, ax = plt.subplots(figsize=(8, 5))
    for y, l in ((P, 'Precision'), (R, 'Recall'), (F, 'F1'), (I, 'IoU')):
        ax.plot(thresholds, y, marker='o', markersize=3, label=l)
    ax.axvline(best_t, color='gray', linestyle='--')
    ax.set_xlabel('Threshold'); ax.set_title(f'Threshold sensitivity (merged, exact) — best F1 @ {best_t:.2f}'); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'threshold_sensitivity.png'), dpi=120); plt.close()
    return best_t


def _build_epoch_row(epoch, lrs, train_time, train_samples, train_loss, train_breakdown, n_skipped,
                     val_time, val_samples, val_loss, val_breakdown, val_metrics, scaler_scale=None):
    row = dict(epoch=epoch)
    for i, lr in enumerate(lrs):
        row[f'lr_group{i}'] = lr
    row.update(train_time_sec=train_time, train_samples_sec=train_samples / max(train_time, 1e-6),
               train_loss=train_loss, train_skipped_batches=n_skipped)
    if scaler_scale is not None:
        row['amp_scale'] = scaler_scale
    for k, v in train_breakdown.items():
        row[f'train_{k}'] = v
    row['val_time_sec'] = val_time
    row['val_loss'] = val_loss if val_loss is not None else ''
    for k, v in val_breakdown.items():
        row[f'val_{k}'] = v
    if val_metrics is not None:
        for group in ('macro', 'merged'):
            for k, v in val_metrics[group].items():
                row[f'val_{group}_{k}'] = v
    if torch.cuda.is_available():
        row['gpu_peak_mem_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return row


def _write_epoch_csv(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# =========================================================================================
#  ONE TRAINING RUN
# =========================================================================================
_CRITERIA = {   # name -> (getter on val metrics, higher_is_better)
    'f1':      (lambda r: r['macro']['relaxed_f1'],             True),
    'rmse':    (lambda r: r['macro']['active_rmse'],            False),
    'chamfer': (lambda r: r['merged']['chamfer_distance'],      False),
    'hd95':    (lambda r: r['merged']['hausdorff95_distance'],  False),
}


def train_run(cfg: dict) -> dict:
    """Train one variant with one seed.  Everything goes to cfg['run_dir'].  Idempotent: returns the stored
    summary if done.json exists (unless cfg['force']); resumes from last.pt if a previous attempt crashed."""
    run_dir = cfg['run_dir']
    os.makedirs(run_dir, exist_ok=True)
    done_path = os.path.join(run_dir, 'done.json')
    if os.path.exists(done_path) and not cfg.get('force', False):
        print(f'[{run_dir}] already finished — skipping (delete done.json or set force=True to retrain).')
        return json.load(open(done_path))

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(_jsonable(cfg), f, indent=2)

    set_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    num_classes = cfg.get('num_classes') or discover_num_classes(cfg['base_dir'], cfg['scenes'])
    print(f'\nRun {run_dir} | variant={cfg["variant"]} seed={cfg["seed"]} device={DEVICE} classes={num_classes}')
    train_loader, val_loader, test_loader = get_loaders(cfg, num_classes)
    if train_loader is None:
        raise RuntimeError('No training data found — check base_dir/scenes.')

    model = build_model(cfg['variant'], num_classes, pretrained=cfg['pretrained']).to(DEVICE)
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    enc_ids = model.encoder_param_ids()
    enc_params   = [p for p in model.parameters() if id(p) in enc_ids and p.requires_grad]
    other_params = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
    groups = [{'params': other_params, 'lr': cfg['lr']}]
    if enc_params:
        groups.insert(0, {'params': enc_params, 'lr': cfg['lr'] * cfg.get('encoder_lr_mult', 1.0)})
    optimizer = AdamW(groups, weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] / 100)
    scaler    = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    ema       = ModelEMA(model, decay=cfg['ema_decay']) if cfg['use_ema'] else None
    loss_fn   = loss_from_cfg(cfg)
    metrics   = metrics_from_cfg(cfg, num_classes)

    history = dict(train_loss=[], val_loss=[], metrics=[], thresh=cfg['seg_thresh'])
    best = {c: dict(value=None, epoch=None) for c in cfg['ckpt_criteria']}
    start_epoch = 1
    csv_path  = os.path.join(run_dir, 'epoch_stats.csv')
    last_path = os.path.join(run_dir, 'last.pt')

    if os.path.exists(last_path):
        st = torch.load(last_path, map_location='cpu')
        model.load_state_dict(st['model'])
        if ema is not None and st.get('ema') is not None:
            ema.ema.load_state_dict(st['ema'])
        optimizer.load_state_dict(st['optimizer']); scheduler.load_state_dict(st['scheduler'])
        if scaler is not None and st.get('scaler'):
            scaler.load_state_dict(st['scaler'])
        history, best, start_epoch = st['history'], st['best'], st['epoch'] + 1
        print(f'  RESUMED from {last_path} at epoch {start_epoch}')

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        print(f"\n{'=' * 60}  Epoch {epoch}/{cfg['epochs']}  [{os.path.basename(run_dir)}]")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_bd, train_time, train_samples, n_skipped = train_epoch(
            train_loader, model, optimizer, loss_fn, scaler, epoch, ema=ema)
        history['train_loss'].append(train_loss)
        print(f'  train_loss={train_loss:.4f}' + (f'   (skipped {n_skipped} non-finite batches)' if n_skipped else ''))

        eval_model = ema.ema if ema else model
        val_loss, val_bd, val_metrics, val_time, val_samples = None, {}, None, 0.0, 0
        if val_loader is not None:
            val_loss, val_bd, val_metrics, val_time, val_samples = eval_epoch(
                val_loader, eval_model, loss_fn, metrics, epoch, tag='Val')
            history['val_loss'].append(val_loss)
            history['metrics'].append(val_metrics)

            for name in cfg['ckpt_criteria']:
                getter, higher = _CRITERIA[name]
                v = float(getter(val_metrics))
                if v != v:                              # NaN (e.g. no curve pairs) never becomes "best"
                    continue
                cur = best[name]['value']
                if cur is None or (v > cur if higher else v < cur):
                    best[name] = dict(value=v, epoch=epoch)
                    torch.save(eval_model.state_dict(), os.path.join(run_dir, f'best_{name}.pth'))
                    print(f'  ✓ new best {name:<8} = {v:.4f}')
            with open(os.path.join(run_dir, 'best.json'), 'w') as f:
                json.dump(best, f, indent=2)

            mode = cfg.get('keep_epoch_ckpts', 'last_k')
            if mode in ('all', 'last_k'):
                torch.save(eval_model.state_dict(), os.path.join(run_dir, f'epoch_{epoch:03d}.pth'))
                if mode == 'last_k':
                    old = os.path.join(run_dir, f'epoch_{epoch - cfg.get("keep_last_k", 10):03d}.pth')
                    if os.path.exists(old):
                        os.remove(old)

            if cfg.get('vis_every', 10) and (epoch % cfg['vis_every'] == 0 or epoch == cfg['epochs']):
                visualise(eval_model, val_loader, loss_fn, cfg['seg_thresh'],
                          os.path.join(run_dir, f'vis_epoch{epoch:03d}.png'), epoch)

        lrs = [g['lr'] for g in optimizer.param_groups]
        _write_epoch_csv(csv_path, _build_epoch_row(
            epoch, lrs, train_time, train_samples, train_loss, train_bd, n_skipped,
            val_time, val_samples, val_loss, val_bd, val_metrics,
            scaler_scale=(scaler.get_scale() if scaler is not None else None)))
        scheduler.step()

        if cfg.get('save_last_every', 0) and epoch % cfg['save_last_every'] == 0 and epoch < cfg['epochs']:
            torch.save(dict(epoch=epoch, model=model.state_dict(), ema=ema.ema.state_dict() if ema else None,
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            scaler=scaler.state_dict() if scaler else None, history=history, best=best),
                       last_path)

    torch.save((ema.ema if ema else model).state_dict(), os.path.join(run_dir, 'final.pth'))
    if os.path.exists(last_path):
        os.remove(last_path)

    _plot_history(history, run_dir)
    _plot_validation_curves(history, run_dir)
    _plot_class_bars(history, run_dir)
    if val_loader is not None:
        threshold_sensitivity_diagnostic(ema.ema if ema else model, val_loader, loss_fn, cfg['threshold_sweep'], run_dir)

    soup_report = {}
    if cfg.get('make_soups', True) and val_loader is not None:
        print('\n── Model soups ──')
        soup_report = build_soups(cfg, num_classes, val_loader)

    summary = dict(run_dir=run_dir, variant=cfg['variant'], seed=cfg['seed'], num_classes=num_classes,
                   best=best, soup=soup_report, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(done_path, 'w') as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f'\nRun finished → {run_dir}')
    return summary


# =========================================================================================
#  CLI
# =========================================================================================
def _parse_value(v: str):
    try:
        return json.loads(v)
    except Exception:
        return v


def main():
    ap = argparse.ArgumentParser(description='Train one HedgeFormer variant (use run_all.py for everything).')
    ap.add_argument('--variant', choices=list(VARIANTS))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--run_dir')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='any other CFG key, JSON values')
    a = ap.parse_args()
    cfg = copy.deepcopy(CFG)
    for k in ('variant', 'seed', 'run_dir', 'epochs'):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.run_dir is None and (a.variant or a.seed is not None):
        cfg['run_dir'] = os.path.join('runs', f'{cfg["variant"]}_s{cfg["seed"]}')
    for kv in a.set:
        k, v = kv.split('=', 1)
        cfg[k] = _parse_value(v)
    train_run(cfg)


if __name__ == '__main__':
    main()
