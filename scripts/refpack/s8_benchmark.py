"""
Section 8 of the work plan: publish a benchmark without publishing pixels.

Airbus permits release of derived feature tensors. That makes a genuine
benchmark possible: anyone can drop their own fusion operator onto the exact
encoder features this study used and be compared directly, with no licence to
buy and no imagery redistributed.

As far as we can tell, a feature-level benchmark released specifically to work
around an imagery licence has no precedent in remote sensing, so the release
itself is a contribution and should be described as one in the paper.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import CLASS_CODE, CLASS_LO_M, DIST_CAP_M, N_CLASSES, PIXEL_SIZE_M


# ==========================================================================
# export
# ==========================================================================
@torch.no_grad()
def export_features(model, loader, out_dir: str, device,
                    dtype=np.float16, max_batches: Optional[int] = None,
                    store_targets: bool = True) -> Dict:
    """
    Writes the three encoder pyramid stages for both modalities, per patch.

    Stored at strides 4, 8 and 16 with channel widths 64, 128 and 320, which
    is exactly what FusionEncoder consumes. A third party can therefore load
    these, run any fusion module they like, feed the existing decoder, and
    obtain a number comparable to ours.

    Half precision is used because it halves the archive and the fusion
    operators all run under autocast anyway. Record the choice in the
    datasheet so nobody is surprised by a fourth-decimal difference.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    manifest = {"patches": [], "dtype": str(np.dtype(dtype)),
                "stages": [], "n_classes": N_CLASSES}

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ms, dem, target = batch[0].to(device), batch[1].to(device), batch[2]
        mask, skel, sample_ids = batch[3], batch[4], batch[5]

        feats_dem = [f for f in model.encoder_dem(dem) if f.shape[1] > 1]
        feats_ms = [f for f in model.encoder_ms(ms) if f.shape[1] > 6]

        if not manifest["stages"]:
            manifest["stages"] = [
                {"index": s, "channels": int(feats_dem[s].shape[1]),
                 "height": int(feats_dem[s].shape[2]),
                 "width": int(feats_dem[s].shape[3])}
                for s in range(3)
            ]

        for b in range(dem.shape[0]):
            sid = sample_ids[b] if isinstance(sample_ids[b], str) else str(sample_ids[b])
            payload = {}
            for s in range(3):
                payload[f"dem_s{s+1}"] = feats_dem[s][b].cpu().numpy().astype(dtype)
                payload[f"ms_s{s+1}"] = feats_ms[s][b].cpu().numpy().astype(dtype)
            if store_targets:
                payload["target"] = target[b].numpy().astype(np.float16)
                payload["nodata_mask"] = mask[b].numpy().astype(np.uint8)
                payload["skeleton"] = skel[b].numpy().astype(np.uint8)

            path = os.path.join(out_dir, f"{sid}.npz")
            np.savez_compressed(path, **payload)
            manifest["patches"].append({
                "id": sid, "file": os.path.basename(path),
                "sha256": _sha256(path),
            })

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ==========================================================================
# the invertibility argument
# ==========================================================================
@torch.no_grad()
def invertibility_evidence(model, loader, device, n_batches: int = 8,
                           steps: int = 300, lr: float = 0.05) -> Dict:
    """
    Evidence that the released tensors cannot be turned back into imagery.

    A licence officer and a reviewer will both ask. Rather than asserting it,
    we run the attack: freeze the encoder, optimise an input image to match
    the released features, and report how close the reconstruction gets. The
    encoder downsamples by four and mixes channels at every block, so the
    inverse problem is badly underdetermined and the reconstruction should
    stay far from the original in any perceptual sense.

    Report the resulting PSNR and correlation in the data availability
    section. If the reconstruction is good, do not publish the tensors.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    results = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ms = batch[0].to(device)
        target_feats = [f.detach() for f in model.encoder_ms(ms) if f.shape[1] > 6][:3]

        x = torch.randn_like(ms, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                f = [g for g in model.encoder_ms(x) if g.shape[1] > 6][:3]
                loss = sum(torch.nn.functional.mse_loss(a, b)
                           for a, b in zip(f, target_feats))
                loss.backward()
                opt.step()

        with torch.no_grad():
            err = (x - ms)
            mse = float((err ** 2).mean())
            rng = float(ms.max() - ms.min()) + 1e-8
            psnr = float(10.0 * np.log10(rng ** 2 / max(mse, 1e-12)))
            a = x.flatten().cpu().numpy()
            b = ms.flatten().cpu().numpy()
            corr = float(np.corrcoef(a, b)[0, 1])
        results.append({"mse": mse, "psnr_db": psnr, "pearson_r": corr})

    psnrs = [r["psnr_db"] for r in results]
    corrs = [r["pearson_r"] for r in results]
    return {
        "n_attacks": len(results),
        "optimisation_steps": steps,
        "mean_psnr_db": float(np.mean(psnrs)) if psnrs else float("nan"),
        "mean_pearson_r": float(np.mean(corrs)) if corrs else float("nan"),
        "verdict": ("safe to release" if psnrs and np.mean(psnrs) < 15.0
                    else "reconstruction is too good, do not release"),
        "method": ("White-box feature inversion by gradient descent on the "
                   "input with the encoder frozen, which is the strongest "
                   "attack a recipient of the tensors could mount."),
    }


# ==========================================================================
# datasheet and split manifest
# ==========================================================================
def write_datasheet(out_dir: str, extra: Optional[Dict] = None) -> str:
    """
    A datasheet in the sense of Gebru et al. Benchmarks without one get used
    incorrectly, and for a positive-unlabelled reference the failure mode is
    predictable: somebody will treat unlabelled as negative and report
    nonsense precision.
    """
    sheet = {
        "name": "HedgeFormer feature-level hedgerow height benchmark",
        "task": ("Per-pixel regression of a bounded proximity field for seven "
                 "ordinal hedgerow canopy-height classes."),
        "why_features_and_not_imagery": (
            "The source imagery is Pleiades Neo, licensed commercially from "
            "Airbus Defence and Space and not redistributable. Airbus permits "
            "release of derived feature tensors, so the benchmark is "
            "distributed at the feature level."),
        "contents": {
            "per_patch": ["dem_s1..s3", "ms_s1..s3", "target", "skeleton",
                          "nodata_mask"],
            "stages": "strides 4, 8 and 16; channel widths 64, 128 and 320",
            "precision": "float16 for features, float16 for targets",
        },
        "label_space": [
            {"channel": i + 1, "ukceh_class": CLASS_CODE[i],
             "lower_bound_m": CLASS_LO_M[i]} for i in range(N_CLASSES)
        ],
        "target_construction": {
            "distance_cap_m": DIST_CAP_M, "pixel_size_m": PIXEL_SIZE_M,
            "definition": ("Euclidean distance to the nearest labelled "
                           "centreline, inverted, capped and normalised to "
                           "[0, 1], restricted to the class buffer."),
        },
        "critical_caveats": [
            "Labels are POSITIVE AND UNLABELLED, not positive and negative. "
            "Every mapped hedge is a hedge; many real hedges are unmapped. "
            "Treating blank pixels as negatives will produce a precision "
            "figure that means nothing.",
            "Reference geometry follows parcel boundaries rather than surveyed "
            "hedge centrelines, so positional agreement of several metres is "
            "expected and absolute distance metrics have a floor set by the "
            "label, not the model.",
            "The reference height classification agrees exactly with field "
            "survey for roughly one third of surveyed length. Absolute scores "
            "must be read against that ceiling.",
            "Woodland, urban areas and rough grassland are masked out of the "
            "reference, so the unlabelled region is not a random sample.",
        ],
        "recommended_protocol": {
            "split": "splits.json, disjoint spatial blocks",
            "threshold": "0.30, selected on validation and held fixed",
            "primary_metrics": ["macro relaxed F1", "macro clDice",
                                "matched Chamfer", "matched Frechet",
                                "quadratic weighted kappa", "MAE in metres"],
            "report_alongside": ("the instance match rate, since matched-only "
                                 "geometry improves trivially for a model that "
                                 "detects fewer and easier instances"),
        },
        "attribution_required": "Imagery (c) Airbus DS 2022",
        "licence_notes": ("Elevation data under the Open Government Licence. "
                          "Reference hedgerow product licensed from UKCEH. "
                          "Feature tensors released with Airbus permission."),
    }
    if extra:
        sheet.update(extra)
    path = os.path.join(out_dir, "DATASHEET.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(sheet, fh, indent=2)
    return path


def write_split_manifest(out_dir: str, splits: Dict[str, List[str]]) -> str:
    """Exact patch identifiers per split, so results are reproducible to the
    patch rather than to the description of a split."""
    path = os.path.join(out_dir, "splits.json")
    with open(path, "w") as fh:
        json.dump({k: sorted(v) for k, v in splits.items()}, fh, indent=2)
    return path


# ==========================================================================
# reference implementation for benchmark users
# ==========================================================================
REFERENCE_USAGE = '''
# Minimal example for a third party using the released tensors.
#
#   1. load one patch
#   2. run your own fusion operator on the six feature maps
#   3. feed the shared decoder
#   4. score with the released evaluation code
#
# No imagery is needed at any point.

import numpy as np, torch
from model import SegFormerDecoder, _Head

d = np.load("patch_0001.npz")
dem = [torch.from_numpy(d[f"dem_s{s}"]).float()[None] for s in (1, 2, 3)]
ms  = [torch.from_numpy(d[f"ms_s{s}"]).float()[None]  for s in (1, 2, 3)]
target = torch.from_numpy(d["target"]).float()[None]

fused = [my_fusion_operator(a, b) for a, b in zip(dem, ms)]   # your method here

decoder = SegFormerDecoder([64, 128, 320], 128)
head = _Head(128, 7)
logits = head(torch.nn.functional.interpolate(
    decoder(fused), size=target.shape[-2:], mode="bilinear", align_corners=False))
pred = torch.sigmoid(logits)
'''


def write_usage_example(out_dir: str) -> str:
    path = os.path.join(out_dir, "example_usage.py")
    with open(path, "w") as fh:
        fh.write(REFERENCE_USAGE)
    return path
