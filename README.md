# HedgeFormer

[![Weights DOI](https://img.shields.io/badge/weights-10.5281%2Fzenodo.24106458-blue)](https://doi.org/10.5281/zenodo.24106458)
[![Code licence: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)

Test-time-training cross-modal fusion of Pléiades Neo imagery and a lidar-derived surface height layer for hedgerow height mapping.

This repository holds the code, run configurations, training logs, evaluation outputs, per-patch scores, split files and class-map galleries behind the paper

> M. Alizadeh Pirbasti, E. Hemmati, N. Alizadeh, G. McArdle and V. Akbari, "HedgeFormer: Test-Time-Training Cross-Modal Fusion of Pléiades Neo Imagery and a Digital Surface Model for Hedgerow Height Mapping", submitted to *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* (2026).

The Pléiades Neo imagery and the training tiles are **not** in this repository (Airbus licence, see [Data and licences](#data-and-licences)). Everything else needed to check the reported numbers is.

## Contents

- [Method in brief](#method-in-brief)
- [Repository layout](#repository-layout)
- [Where the paper's numbers come from](#where-the-papers-numbers-come-from)
- [Height classes](#height-classes)
- [Galleries and the Airbus licence](#galleries-and-the-airbus-licence)
- [Trained weights](#trained-weights)
- [Reproducing](#reproducing)
- [Data and licences](#data-and-licences)
- [Citation](#citation)

## Method in brief

Two MiT-B3 encoders read a 1 m height layer (first-return DSM minus DTM, Environment Agency composite) and the six Pléiades Neo bands. At three scales a bidirectional test-time-training (TTT) cross-attention block exchanges information between the streams: fast weights are re-derived from the context modality of every patch by one normalised gradient step, then read out with the queries of the other modality. A SegFormer-style decoder predicts, for each of the seven UKCEH height classes, a proximity field around the hedge centreline. Supervision comes entirely from the UKCEH Land Cover Plus: Hedgerows product; no pixel was labelled by hand.

Under one protocol (identical encoders, decoder, objective, split and three seeds: 42, 1337, 2024) the TTT operator is compared with concatenation, spatial-reduction cross-attention (SRA), early fusion, late fusion, DSM-only and spectral-only models, plus six objective ablations of TTT: **27 training runs, 81 evaluated weight sets** (best-F1 checkpoint, greedy soup and uniform soup of every run).

## Repository layout

```
main.py                   training of one run (CFG dict at the top; run_all.py overrides it); also builds the
                          greedy and uniform model soups at the end of a run
model.py                  encoders, the seven fusion variants, the TTT block, decoder, build_model / load_checkpoint
dataset.py                tile/sub-patch dataset, per-scene standardisation, augmentation
metrics.py                relaxed P/R/F1, IoU, boundary F, clDice, connectivity, Chamfer/HD95,
                          km-weighted ordinal confusion along the reference centreline
validation.py             test-set evaluation of a weight set: metrics.json/.xlsx, per_patch.npz, plots
postprocess.py            the 16-patch galleries of every finished run and weight set (and missing soups)
analysis.py               perturbation study, fast-weight probe, length estimators, paired statistics, cost
inference_predicting.py   predict on new tiles from a checkpoint
dataprep.py               tile preparation: clips the UKCEH lines, rasterises line/buffer/proximity targets,
                          writes the .npz tiles, band statistics and the split CSVs
run_all.py                the whole study end to end (job table, stages, self-tests, --smoke)

data/splits/<scene>/      train/val/test_grouped.csv, band_statistics.csv, clipped_lines.gpkg   (see data/README.md)

results/runs/<run>/       config.json, epoch_stats.csv, best.json, soup_report.json, done.json and training plots
results/eval/<run>/<w>/   metrics.json, metrics.xlsx, per_patch.npz, confusion_km, per_class_bars,
                          threshold sweeps and gallery_<w>.png, for w in best_f1, soup_greedy, soup_uniform
results/eval/             gallery_patches.json (the 16 gallery patches), GALLERY_STRIP_LOG.json
results/analysis/         perturb/ (perturbation runs), probe/ (fast-weight displacement per patch, TTT),
                          estimators/ (network-length estimators and synthetic calibration),
                          ceiling.json, ordinal_vs_ceiling.csv, complexity.csv
results/report/           aggregated tables: all_runs.csv, variants_mean_std.csv, paired_stats.csv,
                          perturbation_results.csv, loss_ablation_deltas_vs_full.csv, summary.xlsx
scripts/refpack/          reference package used to re-derive the manuscript numbers (s1 to s9)
```

Run names are `<variant>_s<seed>`. Variants: `TTT`, `Simple` (concatenation), `SRA`, `Early`, `Late`, `DEM_only`, `MS_only`; objective ablations: `TTT_abl-<term>_s42` with `<term>` one of `no_cldice`, `conventional_cldice`, `no_margin`, `no_merge_loss`, `no_pu`, `no_skeleton_boost`.

`results/` has the same structure that `run_all.py` writes to its output root (`experiments/` by default), so the scripts can be pointed at it directly. Checkpoints (`*.pth`) are not included. Two runs lack some training plots that were not written at training time: `DEM_only_s42` has no `history.png`, and `DEM_only_s1337` has no `threshold_sensitivity.png`, `topology_metrics_by_class.png` or `validation_curves.png`; their `epoch_stats.csv` are complete.

## Where the paper's numbers come from

| Paper | File |
|---|---|
| Table I (scenes) | `data/splits/<scene>/band_statistics.csv`, DIMAP metadata of the products (not redistributed) |
| Table II (tiles, km per split) | `data/splits/<scene>/*_grouped.csv`, `clipped_lines.gpkg` |
| Tables III, IV, V (main, per-class, ordinal) | `results/eval/<run>/best_f1/metrics.json`; aggregated in `results/report/variants_mean_std.csv` |
| Table VI (lateral offsets on the 16 gallery patches) | panels e, f, g of `results/eval/<run>/best_f1/gallery_best_f1.png` |
| Table VII (paired tests) | `results/report/paired_stats.csv`, from `per_patch.npz` |
| Table VIII (perturbations) | `results/report/perturbation_results.csv`, `results/analysis/perturb/` |
| Table IX (objective ablation) | `results/eval/TTT_abl-*_s42/`, `results/report/loss_ablation_deltas_vs_full.csv` |
| Tables X, XI (checkpoints, soups) | `results/runs/<run>/best.json`, `soup_report.json`; `results/eval/<run>/soup_*/` |
| Table XII (cost) | `results/analysis/complexity.csv` |
| Table XIII (network length) | `results/analysis/estimators/<run>.json`, `synthetic_calibration.json` |
| Fig. 6 (confusion in km) | `results/eval/TTT_s*/best_f1/confusion_km.*` |
| Fig. 8 (training dynamics) | `results/runs/<run>/epoch_stats.csv` |
| Fig. 11 (spatial map of scores) | `results/eval/<run>/best_f1/per_patch.npz` |
| Fig. 12 (gallery) | `results/eval/<run>/best_f1/gallery_best_f1.png`, patches 3, 12 and 16 |
| Fig. 13 (fast-weight probe) | `results/analysis/probe/TTT_s*.csv` |

`per_patch.npz` holds, for each of the 690 test sub-patches (330 in Scene 1, 360 in Scene 2), the relaxed F1, relaxed IoU, clDice, Chamfer and HD95 per class and merged; it contains scores only, no imagery. The 16 gallery patches are a fixed random draw (seed 20240101) of labelled test patches, listed top to bottom in `results/eval/gallery_patches.json`.

## Height classes

The seven classes are the `hghtcls` values of the UKCEH product. The code numbers them in sorted order, and the plots and result files use these numbers:

| In files and plots | `class_1` | `class_2` | `class_3` | `class_4` | `class_5` | `class_6` | `class_7` |
|---|---|---|---|---|---|---|---|
| UKCEH height class | 1a | 1b | 1c | 2 | 3 | 4 | 6 |

## Galleries and the Airbus licence

The galleries are drawn by `postprocess.py` (with `validation.visualize_samples_grid`) and originally had eight columns: (a) Pléiades Neo bands 1 to 3, (b) DSM, (c, d) reference and predicted proximity, (e, f) reference and predicted binary, (g, h) reference and predicted class map. The Pléiades Neo imagery is licensed from Airbus and may not be redistributed, so **column (a) was removed from all 81 galleries in this repository**; the panel letters b to h were left as drawn. The remaining columns come from the Open Government Licence height layer, the UKCEH reference and the model output. `results/eval/GALLERY_STRIP_LOG.json` records the operation (original and new size of each file). The paper's Fig. 12 shows imagery excerpts of three patches under the notice © Airbus DS 2022 and 2024, all rights reserved; those excerpts are not reproduced here.

No other file in this repository contains Pléiades Neo pixels. `band_statistics.csv` holds only the scene-wide mean, standard deviation, minimum and maximum of each band.

## Trained weights

The best-F1 checkpoints of the three TTT seeds are archived on Zenodo: **[doi:10.5281/zenodo.24106458](https://doi.org/10.5281/zenodo.24106458)** (version 1; [doi:10.5281/zenodo.24106457](https://doi.org/10.5281/zenodo.24106457) always points to the latest version)

| File | Run | Size |
|---|---|---|
| `best_f1_TTT_s42.pth` | `results/runs/TTT_s42/` | 365.9 MB |
| `best_f1_TTT_s1337.pth` | `results/runs/TTT_s1337/` | 365.9 MB |
| `best_f1_TTT_s2024.pth` | `results/runs/TTT_s2024/` | 365.9 MB |

Each is the `best_f1.pth` of its run (EMA weights, selected on validation macro F1), stored as a plain PyTorch state dict:

```python
from model import build_model, load_checkpoint
model = build_model('TTT', num_class=7, pretrained=False)
load_checkpoint(model, 'best_f1_TTT_s42.pth')   # strict load
model.eval()
out = model(dem, ms)          # dem: (B,1,H,W), ms: (B,6,H,W), standardised as in dataset.py
probs = out['seg_logits'].sigmoid()             # (B,7,H,W) proximity per height class
```

The other 24 runs can be retrained with `run_all.py`; their configurations and logs are in `results/runs/`.

## Reproducing

Python 3.11.15 and PyTorch 2.10.0 were used, on one RTX 4060 laptop GPU (8 GB); see `requirements.txt`. The recorded training and validation time of the 27 runs adds up to about 893 hours (`train_time_sec` and `val_time_sec` in `epoch_stats.csv`).

The data directory (`base_dir`) holds, per scene, the tile `.npz` files written by `dataprep.py`, `band_statistics.csv` and the three `*_grouped.csv` split files. The `.npz` tiles contain Pléiades Neo pixels and are not distributed; a licence holder can rebuild them from the two scenes and the height layer with `dataprep.py`.

The paths set in the code (`CFG` in `main.py` and `validation.py`, `BASE` in `run_all.py`, the bottom of `dataprep.py` and of `inference_predicting.py`) and the `base_dir` recorded in `results/runs/<run>/config.json` are the authors' local paths. Set them to your own directories first.

```
python run_all.py --list                          # the job table and the enabled stages
python run_all.py --smoke                         # CPU smoke test on synthetic data (minutes, no data needed)
python run_all.py                                 # the full study (27 runs)
python run_all.py --groups fusion_core --seeds 42 # a subset
python postprocess.py --out experiments           # galleries of the finished runs
python validation.py                              # evaluate the weight sets listed in its CFG
python inference_predicting.py                    # predict on new tiles
```

## Data and licences

| Source | Licence | In this repository |
|---|---|---|
| Pléiades Neo scenes (Airbus DS, 16 Oct 2022 and 8 May 2024) | Airbus licence, no redistribution | no: removed from the galleries, not in any file |
| Environment Agency England 1 m Composite DTM/DSM, GEE asset `UK/EA/ENGLAND_1M_TERRAIN/2022` | Open Government Licence v3.0 | only through the DSM panels of the galleries |
| UKCEH Land Cover Plus: Hedgerows 2016–2021 (England) | UKCEH licence (EIDC) | `clipped_lines.gpkg`: the reference centrelines clipped to the two scenes (461.4 km and 400.2 km); see `data/README.md` |

Environment Agency copyright and/or database right 2022. © UKCEH. © Airbus DS 2022 and 2024, all rights reserved.

The source code is released under the MIT licence (`LICENSE`). The data and derived outputs keep the licences of their sources.

## Citation

Until the paper is published, cite the repository (GitHub's "Cite this repository" button reads `CITATION.cff`). If you use the trained weights, also cite the Zenodo record, doi:10.5281/zenodo.24106458.

## Acknowledgment

The Pléiades Neo imagery was provided by Airbus Defence and Space under licence. The hedgerow reference is the UKCEH Land Cover Plus: Hedgerows product. This work was supported by Research Ireland under Grant 18/CRT/6183.
