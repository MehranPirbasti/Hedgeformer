"""
validation.py -- evaluation of trained checkpoints on the held-out test split.

Library functions (used by run_all.py):
    get_eval_loader / collect_predictions / metrics_at_threshold / run_threshold_sweep /
    ordinal_from_cache / evaluate_weights
Standalone:  python validation.py     (evaluates CFG['models'], writes one folder per model + a comparison)

All metric code lives in metrics.py; nothing is re-implemented here.
"""
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from dataset import discover_num_classes
from model import build_model, load_checkpoint
from metrics import SegMetrics, OrdinalKm, METRIC_KEYS, BASE_KEYS
import main as M
from main import CFG as TRAIN_CFG, DEVICE, set_seed, run_step, loss_from_cfg, make_loader


def _log(msg: str):
    print(msg, flush=True)


# =========================================================================================
#  plotting style
# =========================================================================================
def _configure_style():
    preferred = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in preferred if f in available), 'DejaVu Sans')
    plt.rcParams.update({
        'font.family': family, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
        'legend.frameon': False, 'figure.titlesize': 13, 'figure.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#333333',
        'axes.linewidth': 0.9, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
        'lines.linewidth': 1.6, 'lines.markersize': 4, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })


_configure_style()

PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']
HEATMAP_CMAP = 'viridis'


def _palette(n: int) -> list:
    if n <= len(PALETTE):
        return PALETTE[:n]
    cmap = matplotlib.colormaps['tab20']
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _save_fig(fig, save_path: str, also_pdf: bool = True):
    fig.savefig(save_path, dpi=300)
    if also_pdf:
        fig.savefig(os.path.splitext(save_path)[0] + '.pdf')
    plt.close(fig)


METRIC_LABELS = {
    'active_mse': 'Active-Region MSE', 'active_mae': 'Active-Region MAE', 'active_rmse': 'Active-Region RMSE',
    'relaxed_precision': 'Precision (relaxed)', 'relaxed_recall': 'Recall (relaxed)',
    'relaxed_f1': 'F1 (relaxed)', 'relaxed_iou': 'IoU / quality (relaxed)',
    'bf2_precision': 'Boundary precision', 'bf2_recall': 'Boundary recall', 'bf2': 'Boundary F1',
    'cldice': 'clDice',
    'connectivity_correctness': 'Connectivity - correctness', 'connectivity_completeness': 'Connectivity - completeness',
    'connectivity': 'Connectivity (F1)', 'topo_lc': 'Topological length consistency',
    'chamfer_distance': 'Chamfer (px)', 'hausdorff95_distance': 'HD95 (px)', 'curve_coverage': 'Curve coverage',
    'chamfer_distance_matched': 'Chamfer, matched components (px)',
    'hausdorff95_distance_matched': 'HD95, matched components (px)',
}
UNBOUNDED_METRICS = {'chamfer_distance', 'hausdorff95_distance', 'chamfer_distance_matched',
                     'hausdorff95_distance_matched', 'active_mse', 'active_mae', 'active_rmse'}


def _metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key.replace('_', ' ').title())


def _class_names(num_classes: int, cfg: dict = None) -> list:
    names = (cfg or {}).get('class_names')
    if names and len(names) >= num_classes:
        return list(names[:num_classes])
    return [f'Class {i + 1}' for i in range(num_classes)]


# =========================================================================================
#  standalone config (run_all.py builds its own per-run configs)
# =========================================================================================
CFG = dict(
    base_dir=TRAIN_CFG['base_dir'], scenes=TRAIN_CFG['scenes'],
    img_size=256, batch_size=4, num_workers=0, seed=42,
    seg_thresh=0.30, gt_thresh=0.01, metric_tolerance=3, bf2_tolerance=2,
    threshold_sweep=TRAIN_CFG['threshold_sweep'],
    eval_non_aug_only=True, eval_empty_frac=1.0, train_empty_frac=0.25,
    jitter_cfg=None, val_topology=True,
    num_vis_samples=8, output_dir='validation_results',
    class_names=None,          # e.g. ['1a','1b','1c','2','3','4','6']
    class_mid=None,            # metres per class (for ordinal height error), e.g. [0.75,1.25,1.75,2.5,3.5,5,7]
    models=[
        dict(name='TTT_s42', variant='TTT', weight_path='runs/TTT_s42/best_f1.pth'),
    ],
)


# =========================================================================================
#  data / predictions
# =========================================================================================
def get_eval_loader(cfg: dict, split: str, num_classes: int, batch_size=None):
    """Deterministic (unshuffled) loader over un-augmented rows -> identical patch order for every model,
    which is what makes per-patch paired tests possible."""
    return make_loader(cfg, split, num_classes, batch_size=batch_size, shuffle=False)


def scene_pixel_sizes(cfg: dict) -> dict:
    """Pixel size (m) per scene from the NPZ metadata written by dataprep.py."""
    out = {}
    for sc in cfg['scenes']:
        d = os.path.join(cfg['base_dir'], sc, 'npz')
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith('.npz'))
        if files:
            meta = np.load(os.path.join(d, files[0]), allow_pickle=True)['meta'].item()
            out[sc] = float(meta.get('pixel_size_x', 1.0))
    return out


def scene_of(name: str, scenes: list) -> str:
    for s in scenes:
        if name.startswith(s + '_') or name == s:
            return s
    return scenes[0]


@torch.no_grad()
def collect_predictions(model, loader, loss_fn, cfg: dict = None) -> dict:
    """Run the model once and keep logits/targets on the CPU in compact dtypes
    (logits fp16, targets fp16, skeleton/valid uint8)."""
    model.eval()
    L, T, S, V, names = [], [], [], [], []
    for ms, dem, dist_gt, mask, skeleton, nm in tqdm(loader, desc='Inference', leave=False):
        ms, dem, dist_gt = ms.to(DEVICE), dem.to(DEVICE), dist_gt.to(DEVICE)
        mask, skeleton = mask.to(DEVICE), skeleton.to(DEVICE)
        _, _, logits, gt, skel, valid = run_step(ms, dem, dist_gt, mask, skeleton, model, loss_fn)
        L.append(logits.half().cpu()); T.append(gt.half().cpu())
        S.append((skel > 0.5).to(torch.uint8).cpu()); V.append((valid > 0.5).to(torch.uint8).cpu())
        names.extend(list(nm))
    return dict(logits=torch.cat(L), target=torch.cat(T), skeleton=torch.cat(S), valid=torch.cat(V), names=names)


def iter_cache(cache: dict, bs: int = 8):
    n = cache['logits'].shape[0]
    for i in range(0, n, bs):
        sl = slice(i, i + bs)
        yield (cache['logits'][sl].float(), cache['target'][sl].float(), cache['skeleton'][sl].float(),
               cache['valid'][sl].float(), cache['names'][i:i + bs])


def metrics_at_threshold(cache: dict, num_classes: int, thresh: float, cfg: dict,
                         topology: bool = True, per_patch: bool = False):
    eng = SegMetrics(num_classes, thresh=thresh, tolerance=cfg['metric_tolerance'],
                     bf2_tolerance=cfg['bf2_tolerance'], gt_thresh=cfg.get('gt_thresh', 0.01),
                     topology=topology, keep_per_patch=per_patch)
    for lg, tg, sk, vd, nm in iter_cache(cache):
        eng.update(lg.to(DEVICE), tg.to(DEVICE), sk.to(DEVICE), vd.to(DEVICE), nm)
    res = eng.compute()
    return (res, eng.per_patch()) if per_patch else res


def ordinal_from_cache(cache: dict, num_classes: int, thresh: float, cfg: dict) -> dict:
    px = scene_pixel_sizes(cfg)
    eng = OrdinalKm(num_classes, thresh, cfg.get('class_mid'))
    for lg, tg, sk, vd, nm in iter_cache(cache):
        sizes = [px.get(scene_of(n, cfg['scenes']), 1.0) for n in nm]
        eng.update(lg.to(DEVICE), sk.to(DEVICE), vd.to(DEVICE), sizes)
    return eng.compute()


def run_threshold_sweep(cache: dict, num_classes: int, cfg: dict):
    """Sweep with the fast (non-topological) metrics; the operating-point row is recomputed with the same code,
    so the value at seg_thresh in the sweep equals the value in the summary."""
    rows_mm, per_class_rows = [], {f'class_{i + 1}': [] for i in range(num_classes)}
    for i, t in enumerate(cfg['threshold_sweep'], start=1):
        t0 = time.time()
        res = metrics_at_threshold(cache, num_classes, t, cfg, topology=False)
        _log(f'    threshold {i}/{len(cfg["threshold_sweep"])} (t={t:.2f}) {time.time() - t0:.1f}s')
        row = {'threshold': t}
        for k in BASE_KEYS:
            row[f'merged__{k}'] = res['merged'][k]
            row[f'macro__{k}'] = res['macro'][k]
        rows_mm.append(row)
        for c in range(num_classes):
            crow = {'threshold': t}
            crow.update({k: res[f'class_{c + 1}'][k] for k in BASE_KEYS})
            per_class_rows[f'class_{c + 1}'].append(crow)
    return pd.DataFrame(rows_mm), {ck: pd.DataFrame(r) for ck, r in per_class_rows.items()}


# =========================================================================================
#  plots
# =========================================================================================
def plot_threshold_sweep_merged_macro(df, save_path, metric_keys=BASE_KEYS):
    n = len(metric_keys); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, k in zip(axes, metric_keys):
        ax.plot(df['threshold'], df[f'merged__{k}'], color=PALETTE[0], marker='o', markersize=3, label='Merged')
        ax.plot(df['threshold'], df[f'macro__{k}'], color=PALETTE[1], marker='s', markersize=3, ls='--', label='Macro')
        ax.set_title(_metric_label(k), fontsize=9.5, pad=6); ax.set_xlim(0, 1)
        if k not in UNBOUNDED_METRICS:
            ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Decision threshold', fontsize=8)
    for ax in axes[n:]:
        ax.axis('off')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05 / nrows))
    fig.suptitle('Threshold sensitivity', y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save_fig(fig, save_path)


def plot_threshold_sweep_per_class(df_per_class, save_path, metric='relaxed_f1', class_names=None):
    keys = list(df_per_class.keys())
    names = class_names or [k.replace('_', ' ').title() for k in keys]
    colors = _palette(len(keys)); markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (ck, df) in enumerate(df_per_class.items()):
        ax.plot(df['threshold'], df[metric], color=colors[i], marker=markers[i % 8], label=names[i] if i < len(names) else ck)
    ax.set_xlabel('Decision threshold'); ax.set_ylabel(_metric_label(metric)); ax.set_xlim(0, 1)
    ax.set_title(f'Per-class threshold sensitivity - {_metric_label(metric)}')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), title='Class')
    fig.tight_layout(); _save_fig(fig, save_path)


def plot_per_class_bars(summary, num_classes, save_path, class_names=None, cfg=None):
    keys = [f'class_{i + 1}' for i in range(num_classes)]
    names = class_names or _class_names(num_classes, cfg)
    colors = _palette(num_classes)
    n = len(METRIC_KEYS); ncols = 5; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.7 * nrows), squeeze=False)
    axes = axes.reshape(-1); x = np.arange(num_classes)
    for ax, k in zip(axes, METRIC_KEYS):
        vals = np.array([summary[c][k] for c in keys], dtype=float)
        bars = ax.bar(x, np.nan_to_num(vals), color=colors, edgecolor='white', width=0.68)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), 'n/a' if v != v else f'{v:.2f}',
                    ha='center', va='bottom', fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, fontsize=7, ha='right')
        ax.set_title(_metric_label(k), fontsize=9, pad=5)
        if k not in UNBOUNDED_METRICS:
            ax.set_ylim(0, 1.12)
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Per-class performance at the operating threshold', y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.99)); _save_fig(fig, save_path)


def plot_confusion_km(cm_km, save_path, class_names=None):
    cm = np.asarray(cm_km, dtype=float); N = cm.shape[0]
    names = class_names or [f'C{i + 1}' for i in range(N)]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(N):
        for j in range(N + 1):
            ax.text(j, i, f'{cm[i, j]:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > cm.max() * 0.55 else 'black')
    ax.set_xticks(range(N + 1)); ax.set_xticklabels(names + ['missed'], rotation=40, ha='right')
    ax.set_yticks(range(N)); ax.set_yticklabels(names)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('Reference class'); ax.grid(False)
    ax.set_title('Height-class confusion, km of hedge length')
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); _save_fig(fig, save_path)


def _norm_for_display(t: torch.Tensor) -> np.ndarray:
    arr = t.cpu().numpy(); out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        lo, hi = np.percentile(arr[c], 2), np.percentile(arr[c], 98)
        out[c] = np.clip((arr[c] - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0
    return out.transpose(1, 2, 0)


@torch.no_grad()
def visualize_samples_grid(model, vis_loader, loss_fn, thresh, num_classes, num_samples, save_path, class_names=None):
    model.eval()
    names = class_names or [f'Class {c + 1}' for c in range(num_classes)]
    colors = _palette(num_classes)
    it = iter(vis_loader); n_cols = 8
    num_samples = min(num_samples, len(vis_loader))
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(2.6 * n_cols, 2.6 * num_samples), squeeze=False)
    col_titles = ['MS (bands 1-3)', 'DEM', 'GT heatmap', 'Pred. heatmap', 'GT binary', 'Pred. binary',
                  'GT multiclass', 'Pred. multiclass']
    for row in range(num_samples):
        ms, dem, dist_gt, mask, skeleton, _ = next(it)
        ms, dem, dist_gt, mask, skeleton = ms.to(DEVICE), dem.to(DEVICE), dist_gt.to(DEVICE), mask.to(DEVICE), skeleton.to(DEVICE)
        _, _, logits, gt_r, _, _ = run_step(ms, dem, dist_gt, mask, skeleton, model, loss_fn)
        pred_all = torch.sigmoid(logits[0]).cpu().numpy(); gt_all = gt_r[0].cpu().numpy()
        H, W = pred_all.shape[-2:]
        pred_map = np.clip(pred_all.sum(0), 0, 1); gt_map = np.clip(gt_all.sum(0), 0, 1)
        gt_rgb = np.ones((H, W, 3), np.float32); pred_rgb = np.ones((H, W, 3), np.float32)
        rgb = [matplotlib.colors.to_rgb(colors[c % len(colors)]) for c in range(num_classes)]
        for c in range(num_classes):
            gt_rgb[gt_all[c] > 0.0] = rgb[c]; pred_rgb[pred_all[c] >= thresh] = rgb[c]
        imgs = [_norm_for_display(ms[0, :3]), dem.cpu().numpy()[0, 0], gt_map, pred_map,
                (gt_map > 0).astype(np.float32), (pred_map >= thresh).astype(np.float32), gt_rgb, pred_rgb]
        cmaps = [None, 'terrain', HEATMAP_CMAP, HEATMAP_CMAP, 'gray', 'gray', None, None]
        for col, (img, cm) in enumerate(zip(imgs, cmaps)):
            ax = axes[row, col]
            if cm is None and img.ndim == 3:
                ax.imshow(img)
            elif col == 1:
                ax.imshow(img, cmap=cm)
            else:
                ax.imshow(img, cmap=cm, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if row == 0:
                ax.set_title(f'{chr(97 + col)}. {col_titles[col]}', fontsize=9, loc='left')
    handles = [Patch(facecolor=colors[c % len(colors)], label=names[c]) for c in range(num_classes)]
    fig.legend(handles=handles, loc='lower center', ncol=min(num_classes, 8), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Qualitative prediction overview', y=1.01)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99)); _save_fig(fig, save_path)


# =========================================================================================
#  excel
# =========================================================================================
def _autosize_and_style(ws, df, header_fill='1F4E78'):
    hf = Font(bold=True, color='FFFFFF', size=10)
    bg = PatternFill(start_color=header_fill, end_color=header_fill, fill_type='solid')
    for ci, col in enumerate(df.columns, start=1):
        c = ws.cell(row=1, column=ci); c.font = hf; c.fill = bg; c.alignment = Alignment(horizontal='center')
        mx = max([len(str(col))] + [len(f'{v:.4f}' if isinstance(v, float) else str(v)) for v in df[col]]) if len(df) else len(str(col))
        ws.column_dimensions[get_column_letter(ci)].width = min(max(mx + 2, 10), 40)
        if pd.api.types.is_float_dtype(df[col]):
            for r in range(2, len(df) + 2):
                ws.cell(row=r, column=ci).number_format = '0.0000'
    ws.freeze_panes = 'A2'


def save_excel(save_path, summary, num_classes, df_sweep_mm, df_sweep_pc, eval_thresh, class_names=None, ordinal=None):
    names = class_names or [f'Class {i + 1}' for i in range(num_classes)]
    with pd.ExcelWriter(save_path, engine='openpyxl') as w:
        groups = ['merged', 'macro'] + [f'class_{i + 1}' for i in range(num_classes)]
        labels = ['Merged (pixel-level)', 'Macro (supported classes)'] + names
        rows = []
        for g, lab in zip(groups, labels):
            row = {'group': lab}
            row.update({_metric_label(k): v for k, v in summary[g].items() if k in METRIC_LABELS})
            for k in ('gt_px', 'pred_px', 'gt_line_px', 'n_patches_gt'):
                row[k] = summary[g].get(k)
            rows.append(row)
        df = pd.DataFrame(rows); df.insert(1, 'Evaluation threshold', eval_thresh)
        df.to_excel(w, sheet_name='Summary', index=False); _autosize_and_style(w.sheets['Summary'], df)
        df_sweep_mm.to_excel(w, sheet_name='Threshold sweep', index=False)
        _autosize_and_style(w.sheets['Threshold sweep'], df_sweep_mm)
        for ck, d in df_sweep_pc.items():
            idx = int(ck.split('_')[1]) - 1
            sn = (names[idx] if idx < len(names) else ck)[:31]
            d.to_excel(w, sheet_name=sn, index=False); _autosize_and_style(w.sheets[sn], d)
        if ordinal:
            od = pd.DataFrame([{k: v for k, v in ordinal.items() if not isinstance(v, list)}])
            od.to_excel(w, sheet_name='Ordinal', index=False); _autosize_and_style(w.sheets['Ordinal'], od)
            cm = pd.DataFrame(ordinal['confusion_km'], index=names, columns=names + ['missed'])
            cm.to_excel(w, sheet_name='Confusion km')


# =========================================================================================
#  evaluate one set of weights (used by run_all.py and by the standalone main)
# =========================================================================================
def evaluate_weights(cfg: dict, variant: str, weight_path: str, num_classes: int, test_loader, out_dir: str,
                     loss_fn=None, vis_loader=None, make_plots: bool = True, do_sweep: bool = True,
                     class_names=None) -> dict:
    """Full evaluation of ONE checkpoint.  Writes into out_dir:
       metrics.json (summary, ordinal), per_patch.npz (for paired tests), confusion + threshold-sweep figures,
       metrics.xlsx.  Returns the dict written to metrics.json plus the prediction cache."""
    os.makedirs(out_dir, exist_ok=True)
    loss_fn = loss_fn or loss_from_cfg(TRAIN_CFG)
    class_names = class_names or _class_names(num_classes, cfg)
    t_start = time.time()

    model = build_model(variant, num_classes, pretrained=False).to(DEVICE)
    load_checkpoint(model, weight_path, map_location=DEVICE)          # strict
    model.eval()

    cache = collect_predictions(model, test_loader, loss_fn, cfg)
    _log(f'  [{os.path.basename(out_dir)}] {cache["logits"].shape[0]} test patches predicted')
    summary, (pp_groups, pp_names) = metrics_at_threshold(cache, num_classes, cfg['seg_thresh'], cfg,
                                                          topology=True, per_patch=True)
    ordinal = ordinal_from_cache(cache, num_classes, cfg['seg_thresh'], cfg)

    flat = {'names': np.array(pp_names)}
    for g, d in pp_groups.items():
        for k, v in d.items():
            flat[f'{g}__{k}'] = v
    np.savez(os.path.join(out_dir, 'per_patch.npz'), **flat)

    df_mm = df_pc = None
    if do_sweep:
        df_mm, df_pc = run_threshold_sweep(cache, num_classes, cfg)
    if make_plots:
        plot_per_class_bars(summary, num_classes, os.path.join(out_dir, 'per_class_bars.png'), class_names, cfg)
        plot_confusion_km(ordinal['confusion_km'], os.path.join(out_dir, 'confusion_km.png'), class_names)
        if do_sweep:
            plot_threshold_sweep_merged_macro(df_mm, os.path.join(out_dir, 'threshold_sweep_merged_macro.png'))
            plot_threshold_sweep_per_class(df_pc, os.path.join(out_dir, 'threshold_sweep_per_class_f1.png'),
                                           class_names=class_names)
        if vis_loader is not None and cfg.get('num_vis_samples', 0):
            visualize_samples_grid(model, vis_loader, loss_fn, cfg['seg_thresh'], num_classes,
                                   cfg['num_vis_samples'], os.path.join(out_dir, 'samples_overview.png'), class_names)
    if do_sweep:
        save_excel(os.path.join(out_dir, 'metrics.xlsx'), summary, num_classes, df_mm, df_pc,
                   cfg['seg_thresh'], class_names, ordinal)

    out = dict(variant=variant, weight_path=weight_path, threshold=cfg['seg_thresh'], n_patches=len(cache['names']),
               summary=M._jsonable(summary), ordinal=M._jsonable(ordinal),
               seconds=time.time() - t_start)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    out['_cache'] = cache
    return out


def plot_model_comparison(rows: list, save_path: str, metric: str = 'relaxed_f1'):
    df = pd.DataFrame(rows); colors = _palette(len(df))
    fig, ax = plt.subplots(figsize=(max(5.5, 1.4 * len(df)), 4.4))
    x = np.arange(len(df)); w = 0.35
    for off, key, col, lab in ((-w / 2, f'merged__{metric}', colors[0], 'Merged'),
                               (w / 2, f'macro__{metric}', colors[1 % len(colors)], 'Macro')):
        bars = ax.bar(x + off, df[key], w, label=lab, color=col, edgecolor='white')
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f'{r.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(df['model']); ax.set_ylabel(_metric_label(metric))
    if metric not in UNBOUNDED_METRICS:
        ax.set_ylim(0, 1.12)
    ax.set_title(f'Model comparison - {_metric_label(metric)}'); ax.legend()
    fig.tight_layout(); _save_fig(fig, save_path)


# =========================================================================================
#  standalone entry
# =========================================================================================
def main():
    cfg = CFG
    set_seed(cfg['seed'])
    os.makedirs(cfg['output_dir'], exist_ok=True)
    num_classes = discover_num_classes(cfg['base_dir'], cfg['scenes'])
    test_loader = get_eval_loader(cfg, 'test', num_classes)
    vis_loader = make_loader(cfg, 'test', num_classes, batch_size=1, shuffle=True)
    loss_fn = loss_from_cfg(TRAIN_CFG)
    rows = []
    for mc in cfg['models']:
        out = evaluate_weights(cfg, mc['variant'], mc['weight_path'], num_classes, test_loader,
                               os.path.join(cfg['output_dir'], mc['name']), loss_fn, vis_loader)
        row = {'model': mc['name'], 'eval_threshold': cfg['seg_thresh']}
        for g in ('macro', 'merged'):
            row.update({f'{g}__{k}': v for k, v in out['summary'][g].items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(os.path.join(cfg['output_dir'], 'all_models_comparison.xlsx'), engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Model comparison', index=False)
        _autosize_and_style(w.sheets['Model comparison'], df)
    plot_model_comparison(rows, os.path.join(cfg['output_dir'], 'model_comparison_f1.png'))
    _log('done.')


if __name__ == '__main__':
    main()
