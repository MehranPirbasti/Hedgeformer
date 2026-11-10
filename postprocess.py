"""
postprocess.py -- gallery images (and a soup safety net) for runs that run_all.py has already finished.

It NEVER trains anything and only ADDS files.  Safe to run many times: galleries that already exist are
skipped (use --force to redraw them), and runs that are not finished yet (no done.json) are ignored.

For every finished run under <out>/runs/<run_id>/ and every weight in --weights it writes
    <out>/eval/<run_id>/<weight>/gallery_<weight>.png
showing the SAME fixed set of --n (default 16) LABELLED test patches, in the SAME order, for every run and
every weight, so the panels can be compared across models.  The patch list is saved once to
    <out>/eval/gallery_patches.json

The patches are drawn once with a fixed seed from the pooled test split: un-augmented rows that contain a hedge
(is_empty == 0), preferring rows that are not mostly nodata (is_masked == 0).

If soup_greedy.pth / soup_uniform.pth are missing for a run, they are built from the checkpoints that are
already on disk for that run (same code as the end of training).  Normally they exist already, in
<out>/runs/<run_id>/ next to best_f1.pth.

IMPORTANT: model.py must be configured exactly as it was when the runs were trained (FUSION_MIXER, DECODER,
HEDGE_ENCODER).  Checkpoints are loaded strictly, so a mismatch is reported per run/weight and skipped.

Usage (from the folder that contains main.py, model.py, validation.py ...):
    python postprocess.py                                   # everything, default settings
    python postprocess.py --runs TTT_s42 --weights best_f1 --n 4     # quick one-run test first
    python postprocess.py --force                           # redraw existing galleries
"""
import argparse
import gc
import glob
import json
import os
import time
import traceback

import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import main as M
import validation as V
from model import build_model, load_checkpoint
from dataset import HedgeDistanceDataset

GALLERY_SEED = 20240101   # fixed and independent of every training seed -> same patches for every run/weight


def _log(msg=''):
    print(msg, flush=True)


def _install_light_save(dpi: int):
    """validation.visualize_samples_grid saves at 300 dpi AND writes a PDF.  For a 16-row gallery that is a
    ~6000 x 12000 px image per file; use a moderate dpi and PNG only instead."""
    def _save_fig_light(fig, save_path, also_pdf=False):
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    V._save_fig = _save_fig_light


# =========================================================================================
#  the fixed patch set (built once, reused for every run)
# =========================================================================================
def build_gallery_loader(cfg: dict, n: int, seed: int):
    dfs = []
    for scene in cfg['scenes']:
        try:
            dfs.append(M._load_csv(os.path.join(cfg['base_dir'], scene), 'test'))
        except FileNotFoundError as exc:
            _log(f'  WARNING: {exc} - skipping scene "{scene}"')
    if not dfs:
        raise RuntimeError('no test_grouped.csv found - check base_dir/scenes in the run configs')
    df = pd.concat(dfs, axis=0, ignore_index=True)
    if 'Augmentation' in df.columns:
        df = df[df['Augmentation'] == 'non']
    df = df[df['is_empty'] == 0]
    if 'is_masked' in df.columns and (df['is_masked'] == 0).sum() >= n:
        df = df[df['is_masked'] == 0]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError('no labelled (is_empty==0) test patches found - cannot build the gallery')
    if len(df) < n:
        _log(f'  WARNING: only {len(df)} labelled test patches available, requested {n} - using all of them')
        n = len(df)
    picked = df.sample(n=n, random_state=seed).reset_index(drop=True)
    ids = [f'{r.Sample_ID}_{r.SubPatch}_{getattr(r, "Augmentation", "non")}' for r in picked.itertuples()]

    ds = HedgeDistanceDataset(base_dir=cfg['base_dir'], dataframe=picked, num_classes=cfg['num_classes'],
                              split='test', img_size=cfg['img_size'], jitter_cfg=None)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    return loader, n, ids


# =========================================================================================
#  discover finished runs from disk (independent of run_all.py's GROUPS)
# =========================================================================================
def discover_runs(out_root: str, only_runs=None):
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(out_root, 'runs', '*'))):
        run_id = os.path.basename(run_dir)
        if only_runs and run_id not in only_runs:
            continue
        done_path = os.path.join(run_dir, 'done.json')
        cfg_path = os.path.join(run_dir, 'config.json')
        if not (os.path.exists(done_path) and os.path.exists(cfg_path)):
            continue
        done = json.load(open(done_path))
        cfg = json.load(open(cfg_path))
        runs.append(dict(run_id=run_id, run_dir=run_dir, variant=done['variant'],
                         num_classes=done['num_classes'], cfg=cfg))
    return runs


def soups_missing(run) -> bool:
    return not (os.path.exists(os.path.join(run['run_dir'], 'soup_greedy.pth')) and
                os.path.exists(os.path.join(run['run_dir'], 'soup_uniform.pth')))


def build_missing_soups(run, workers: int):
    _log(f'  soup files missing for {run["run_id"]} - building them from the checkpoints on disk ...')
    cfg = dict(run['cfg']); cfg['variant'] = run['variant']; cfg['num_workers'] = workers
    val_loader = V.get_eval_loader(cfg, 'val', run['num_classes'])
    M.build_soups(cfg, run['num_classes'], val_loader, run_dir=run['run_dir'])


# =========================================================================================
#  main
# =========================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments', help='same output root run_all.py used')
    ap.add_argument('--weights', nargs='+', default=['best_f1', 'soup_greedy', 'soup_uniform'])
    ap.add_argument('--n', type=int, default=16, help='patches per gallery')
    ap.add_argument('--gallery-seed', type=int, default=GALLERY_SEED)
    ap.add_argument('--runs', nargs='*', default=None, help='only these run ids (default: all finished runs)')
    ap.add_argument('--dpi', type=int, default=120, help='gallery resolution (PNG only)')
    ap.add_argument('--workers', type=int, default=0, help='DataLoader workers if soups must be built')
    ap.add_argument('--force', action='store_true', help='redraw galleries that already exist')
    ap.add_argument('--no-soups', action='store_true', help='do not build missing soups; just skip that weight')
    a = ap.parse_args()

    runs = discover_runs(a.out, set(a.runs) if a.runs else None)
    if not runs:
        _log(f'no finished runs under {a.out}/runs (a run needs done.json and config.json)'); return
    _log(f'{len(runs)} finished run(s): {[r["run_id"] for r in runs]}')

    _install_light_save(a.dpi)

    base_cfg = dict(runs[0]['cfg']); base_cfg['num_classes'] = runs[0]['num_classes']
    gallery_loader, n_used, patch_ids = build_gallery_loader(base_cfg, a.n, a.gallery_seed)
    os.makedirs(os.path.join(a.out, 'eval'), exist_ok=True)
    with open(os.path.join(a.out, 'eval', 'gallery_patches.json'), 'w') as f:
        json.dump(dict(seed=a.gallery_seed, n=n_used, patches_top_to_bottom=patch_ids), f, indent=2)
    _log(f'gallery uses {n_used} fixed labelled patches (saved to {a.out}/eval/gallery_patches.json):')
    for i, pid in enumerate(patch_ids, 1):
        _log(f'   row {i:>2}: {pid}')

    loss_fn = M.loss_from_cfg(base_cfg)
    class_names = base_cfg.get('class_names')
    made, skipped, failures = 0, 0, []

    for run in runs:
        run_id, variant, nc = run['run_id'], run['variant'], run['num_classes']
        cfg = dict(run['cfg']); cfg['num_classes'] = nc
        _log(f'\n=== {run_id} ({variant}) ===')

        wants_soup = any(w.startswith('soup_') for w in a.weights)
        if wants_soup and not a.no_soups and soups_missing(run):
            try:
                build_missing_soups(run, a.workers)
            except Exception:
                _log(f'  !! building soups for {run_id} failed:\n{traceback.format_exc()}')
                failures.append(f'{run_id}/soups')

        for w in a.weights:
            wp = os.path.join(run['run_dir'], f'{w}.pth')
            if not os.path.exists(wp):
                _log(f'  [{w}] no such weights file - skipped'); skipped += 1; continue
            out_dir = os.path.join(a.out, 'eval', run_id, w)
            gallery_path = os.path.join(out_dir, f'gallery_{w}.png')
            if os.path.exists(gallery_path) and not a.force:
                _log(f'  [{w}] gallery exists - skipped'); skipped += 1; continue
            try:
                os.makedirs(out_dir, exist_ok=True)
                t0 = time.time()
                model = build_model(variant, nc, pretrained=False).to(M.DEVICE)
                load_checkpoint(model, wp, map_location=M.DEVICE)      # strict
                model.eval()
                V.visualize_samples_grid(model, gallery_loader, loss_fn, cfg['seg_thresh'], nc, n_used,
                                         gallery_path, class_names=class_names)
                _log(f'  [{w}] -> {gallery_path}  ({time.time() - t0:.1f}s)')
                made += 1
            except Exception:
                _log(f'  !! [{w}] FAILED for {run_id}  (if this is a size/key mismatch, model.py is not set '
                     f'like it was during training):\n{traceback.format_exc()}')
                failures.append(f'{run_id}/{w}')
            finally:
                model = None
                plt.close('all')
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    _log(f'\nDone: {made} gallery image(s) written, {skipped} skipped, {len(failures)} failed.')
    if failures:
        _log(f'Failed: {failures}')
    _log(f'Images are in {a.out}/eval/<run_id>/<weight>/gallery_<weight>.png')


if __name__ == '__main__':
    main()
