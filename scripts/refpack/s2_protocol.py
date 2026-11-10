"""
Section 2 of the work plan: a controlled experimental protocol.

Three problems are fixed here.

  2.1  Unequal training length. Every run gets the same epoch budget and the
       same checkpoint selection criterion.
  2.2  Unequal data volume. Turned into a deliberate data-efficiency ablation
       with stratified subsets that every variant and every seed shares.
  2.3  A single seed. Three seeds per cell, with the aggregation in
       refpack.stats.

Plus 2.4, the parameter and inference-cost table the manuscript is missing.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import (DATA_FRACTIONS, EPOCHS, N_CLASSES, PATCH, RUN_DIR,
                     SEEDS, SELECT_CKPT_ON, SPLIT_DIR, VARIANTS)


# --------------------------------------------------------------------------
# 2.2  stratified subsets, built once and reused by every run
# --------------------------------------------------------------------------
def patch_dominant_class(npz_path: str) -> int:
    """Class with the most centreline pixels in a patch, 0 if the patch is
    empty. Used only as a stratification key."""
    d = np.load(npz_path, allow_pickle=True)
    line = d["label_line"]
    counts = [(int((line == c).sum()), c) for c in range(1, N_CLASSES + 1)]
    n, c = max(counts)
    return c if n > 0 else 0


def build_stratified_subsets(train_df: pd.DataFrame,
                             base_dir: str,
                             fractions: Sequence[float] = DATA_FRACTIONS,
                             strat_seed: int = 0,
                             cache_key_col: str = "Rel_Path") -> Dict[float, List[int]]:
    """
    Draws nested stratified subsets of the training rows.

    Two properties matter and both are easy to get wrong.

    Nested: the 25 percent subset is contained in the 50 percent subset, which
    is contained in the full set. Without nesting, a difference between
    fractions could come from which patches were drawn rather than from how
    many, and the data-efficiency curve would not be interpretable.

    Stratified: the share of each height class is preserved. At 25 percent an
    unstratified draw would effectively delete the rare classes and the
    per-class results would collapse for a reason that has nothing to do with
    the fusion operator.
    """
    strat = []
    for rel in train_df[cache_key_col]:
        p = os.path.join(base_dir, str(rel).replace("\\", "/"))
        try:
            strat.append(patch_dominant_class(p))
        except Exception:
            strat.append(0)
    train_df = train_df.copy()
    train_df["_strat"] = strat

    rng = np.random.default_rng(strat_seed)
    # a single shuffled order per stratum gives nesting for free: take the
    # first k of the same permutation for every fraction
    order: Dict[int, np.ndarray] = {}
    for key, grp in train_df.groupby("_strat"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        order[key] = idx

    subsets: Dict[float, List[int]] = {}
    for frac in sorted(fractions):
        chosen = []
        for key, idx in order.items():
            k = int(round(len(idx) * frac))
            chosen.extend(idx[:k].tolist())
        subsets[frac] = sorted(chosen)

    # verify nesting and stratification, because a silent failure here would
    # invalidate the headline figure of the paper
    fr = sorted(subsets)
    for a, b in zip(fr, fr[1:]):
        assert set(subsets[a]).issubset(set(subsets[b])), "subsets are not nested"
    full = Counter(train_df["_strat"])
    for frac, ids in subsets.items():
        got = Counter(train_df.loc[ids, "_strat"])
        for key in full:
            want = full[key] * frac
            assert abs(got[key] - want) <= 1.5, f"stratum {key} off at frac {frac}"

    return subsets


def write_subsets(subsets: Dict[float, List[int]], train_df: pd.DataFrame,
                  out_dir: str = SPLIT_DIR) -> Dict[float, str]:
    paths = {}
    for frac, ids in subsets.items():
        rows = train_df.loc[ids]
        path = os.path.join(out_dir, f"train_frac{frac:.2f}.csv")
        rows.drop(columns=[c for c in rows.columns if c.startswith("_")]).to_csv(
            path, index=False)
        paths[frac] = path
        print(f"fraction {frac:.2f}: {len(rows):,} rows -> {path}")
    return paths


# --------------------------------------------------------------------------
# 2.1 + 2.3  the run matrix
# --------------------------------------------------------------------------
def run_matrix(fractions: Sequence[float] = DATA_FRACTIONS,
               variants: Sequence[str] = VARIANTS,
               seeds: Sequence[int] = SEEDS,
               epochs: int = EPOCHS) -> List[Dict]:
    """
    Every cell of the experiment. Note what is held fixed: epochs, the
    validation split, the checkpoint criterion, the learning-rate schedule.
    The only things that vary are the fusion operator, the training fraction
    and the seed.
    """
    rows = []
    for frac in fractions:
        for variant in variants:
            for seed in seeds:
                tag = f"{variant}_f{frac:.2f}_s{seed}"
                rows.append({
                    "tag": tag,
                    "attention_type": variant,
                    "train_csv": os.path.join(SPLIT_DIR, f"train_frac{frac:.2f}.csv"),
                    "data_fraction": frac,
                    "seed": seed,
                    "epochs": epochs,
                    "select_ckpt_on": SELECT_CKPT_ON,
                    "out_dir": os.path.join(RUN_DIR, tag),
                    "log_csv": os.path.join(RUN_DIR, tag, "epoch_stats.csv"),
                })
    return rows


def minimal_run_matrix() -> List[Dict]:
    """If the compute budget will not stretch to twenty-seven runs, drop the
    middle fraction. Eighteen runs still support the data-efficiency claim."""
    return run_matrix(fractions=[0.25, 1.00])


def emit_run_script(rows: List[Dict], path: str, python: str = "python") -> str:
    """Writes a shell script so the whole matrix can be launched and resumed."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for r in rows:
        lines += [
            f'if [ ! -f "{r["out_dir"]}/DONE" ]; then',
            f'  echo "=== {r["tag"]}"',
            f'  mkdir -p "{r["out_dir"]}"',
            f'  {python} -m refpack.train_one \\',
            f'    --attention-type {r["attention_type"]} \\',
            f'    --train-csv "{r["train_csv"]}" \\',
            f'    --seed {r["seed"]} --epochs {r["epochs"]} \\',
            f'    --select-on {r["select_ckpt_on"]} \\',
            f'    --out-dir "{r["out_dir"]}"',
            f'  touch "{r["out_dir"]}/DONE"',
            "fi", "",
        ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    os.chmod(path, 0o755)
    return path


# --------------------------------------------------------------------------
# 2.4  parameter counts and inference cost
# --------------------------------------------------------------------------
def parameter_report(num_class: int = N_CLASSES, weights=None) -> Dict[str, Dict]:
    """
    Fills the three FILL entries in the complexity table, and checks the claim
    the paper makes about the comparison being controlled: the encoder and
    decoder must be bit-for-bit identical in size across variants.
    """
    import torch
    from model import HedgeFormer

    report = {}
    for variant in VARIANTS:
        m = HedgeFormer(num_class=num_class, attention_type=variant)
        count = lambda mod: sum(p.numel() for p in mod.parameters())
        enc = count(m.encoder_dem) + count(m.encoder_ms)
        fuse = sum(count(f) for f in m.fuse_modules)
        dec = count(m.decoder) + count(m.seg_head)
        total = sum(p.numel() for p in m.parameters())
        report[variant] = {
            "total": total,
            "total_M": round(total / 1e6, 2),
            "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
            "encoders": enc, "encoders_M": round(enc / 1e6, 2),
            "fusion": fuse, "fusion_M": round(fuse / 1e6, 3),
            "decoder_head": dec, "decoder_head_M": round(dec / 1e6, 3),
            "per_scale_fusion": [count(f) for f in m.fuse_modules],
        }
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    encs = {r["encoders"] for r in report.values()}
    decs = {r["decoder_head"] for r in report.values()}
    report["_controlled_comparison"] = {
        "encoder_identical": len(encs) == 1,
        "decoder_identical": len(decs) == 1,
        "note": ("If either is False the paper cannot claim that differences "
                 "are attributable to the fusion operator alone."),
    }
    return report


def inference_cost(variant: str = "TTT", num_class: int = N_CLASSES,
                   batch: int = 8, reps: int = 50, device: str = "cuda") -> Dict:
    """
    Latency, throughput and peak memory at inference, none of which the
    manuscript currently reports. Also converts to seconds per square
    kilometre, which is the number that matters for the stated motivation of
    national-scale monitoring.
    """
    import torch
    from model import HedgeFormer

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    m = HedgeFormer(num_class=num_class, attention_type=variant).to(dev).eval()
    dtm = torch.randn(batch, 1, PATCH, PATCH, device=dev)
    ms = torch.randn(batch, 6, PATCH, PATCH, device=dev)

    with torch.no_grad():
        for _ in range(5):                      # warm up
            m(dtm, ms)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        import time
        t0 = time.perf_counter()
        for _ in range(reps):
            m(dtm, ms)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps

    per_patch_s = dt / batch
    # tiles overlap by 64 px of 256, so each square metre is visited
    # (256 / 192)^2 times by the sliding window
    stride = PATCH - 64
    redundancy = (PATCH / stride) ** 2
    px_per_km2 = 1_000_000.0                     # at 1 m
    patches_per_km2 = px_per_km2 / (PATCH * PATCH) * redundancy

    return {
        "variant": variant,
        "batch": batch,
        "latency_per_batch_s": dt,
        "latency_per_patch_ms": per_patch_s * 1000.0,
        "throughput_patch_per_s": 1.0 / per_patch_s,
        "peak_memory_GiB": (torch.cuda.max_memory_allocated() / 2**30
                            if dev.type == "cuda" else float("nan")),
        "tile_redundancy_factor": redundancy,
        "patches_per_km2": patches_per_km2,
        "seconds_per_km2": patches_per_km2 * per_patch_s,
        "device": str(dev),
    }


def flops_report(variant: str = "TTT", num_class: int = N_CLASSES) -> Dict:
    """Optional, needs fvcore or ptflops. Reviewers increasingly ask for it."""
    import torch
    from model import HedgeFormer
    m = HedgeFormer(num_class=num_class, attention_type=variant).eval()
    dtm = torch.randn(1, 1, PATCH, PATCH)
    ms = torch.randn(1, 6, PATCH, PATCH)
    try:
        from fvcore.nn import FlopCountAnalysis
        f = FlopCountAnalysis(m, (dtm, ms))
        f.unsupported_ops_warnings(False)
        return {"variant": variant, "gflops": f.total() / 1e9,
                "note": ("fvcore does not count the fast-weight update as a "
                         "standard op; treat TTT numbers as a lower bound.")}
    except ImportError:
        return {"variant": variant, "gflops": None,
                "note": "pip install fvcore to populate this row"}


def main():
    out = {"parameters": parameter_report()}
    try:
        out["inference_cost"] = {v: inference_cost(v) for v in VARIANTS}
    except Exception as exc:
        out["inference_cost"] = {"error": str(exc)}
    path = os.path.join(RUN_DIR, "s2_complexity.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["parameters"], indent=2))
    print("written:", path)


if __name__ == "__main__":
    main()
