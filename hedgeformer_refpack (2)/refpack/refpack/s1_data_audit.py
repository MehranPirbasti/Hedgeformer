"""
Section 1 of the work plan: lock down the data facts.

Nothing in this module trains anything. It answers, from the rasters and the
patch archive themselves, the questions a reviewer will ask about provenance,
geometry and class support.

Run:
    python -m refpack.s1_data_audit --dem Scene_1_Height.tif \
        --ms scene1.tif --patches ./data/Scene1/npz --out ./refpack_out/reports
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Dict

import numpy as np

from .config import (BUFFER_M, CLASS_CODE, DIST_CAP_M, N_CLASSES, PIXEL_SIZE_M,
                     REPORT_DIR)


# --------------------------------------------------------------------------
# 1.2  is the elevation layer terrain, surface, or normalised height?
# --------------------------------------------------------------------------
def classify_elevation_layer(path: str, sample_px: int = 4_000_000) -> Dict:
    """
    Decides, from the value distribution alone, which of the three candidate
    products the elevation raster actually is.

    The decision rule is blunt on purpose. Absolute elevation around
    St Albans sits roughly between 40 m and 150 m above ordnance datum.
    A normalised canopy height model sits near zero with a thin positive tail.
    """
    import rasterio
    with rasterio.open(path) as ds:
        arr = ds.read(1, out_shape=(1, min(ds.height, 2048), min(ds.width, 2048)),
                      masked=True).astype("float64")
        profile = {
            "crs": str(ds.crs), "res": ds.res, "width": ds.width,
            "height": ds.height, "dtype": ds.dtypes[0], "nodata": ds.nodata,
            "bounds": list(ds.bounds),
        }
    v = np.asarray(arr.compressed() if hasattr(arr, "compressed") else arr).ravel()
    v = v[np.isfinite(v)]
    if v.size > sample_px:
        v = np.random.default_rng(0).choice(v, sample_px, replace=False)

    q = np.percentile(v, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    frac_near_zero = float(np.mean(np.abs(v) < 0.5))
    frac_negative = float(np.mean(v < -0.1))

    if q[4] > 25.0:
        verdict = "absolute elevation (DSM or DTM)"
        note = ("The median is far above zero, so this is an absolute surface. "
                "The network must then learn to subtract terrain implicitly, "
                "which has to be justified in the paper.")
    elif frac_near_zero > 0.5 and q[8] < 40.0:
        verdict = "normalised canopy height (CHM)"
        note = ("Values cluster at zero with a positive tail, which is a "
                "normalised height layer. State how it was produced.")
    else:
        verdict = "ambiguous"
        note = "Neither rule fired cleanly. Inspect the histogram by hand."

    # a DTM and a DSM are hard to separate from values alone; the giveaway is
    # local roughness over vegetated areas
    return {
        "path": path, "profile": profile, "verdict": verdict, "note": note,
        "percentiles": {f"p{p}": float(x) for p, x in
                        zip([0.1, 1, 5, 25, 50, 75, 95, 99, 99.9], q)},
        "fraction_within_half_metre_of_zero": frac_near_zero,
        "fraction_negative": frac_negative,
        "native_resolution_m": profile["res"],
        "resolution_matches_1m": bool(abs(profile["res"][0] - 1.0) < 0.05),
    }


def gee_provenance_template(out_path: str) -> str:
    """
    Writes the form that has to come back filled in from whoever ran the
    Earth Engine export. The public Earth Engine catalogue has no 1 m
    elevation product for England, so either the Environment Agency data was
    uploaded as a private asset or a different product was used. The paper
    cannot describe the data correctly until this is answered.
    """
    template = {
        "asset_id_or_collection": "",
        "is_private_upload": None,
        "original_source_if_upload": "",
        "export_scale_m": None,
        "export_crs": "",
        "reducer_or_mosaic_rule": "",
        "date_filter_start": "",
        "date_filter_end": "",
        "acquisition_season": "",
        "_why_this_matters": (
            "If export scale differs from the native resolution of the source, "
            "Earth Engine resampled the data and the effective resolution is "
            "not what the paper claims."
        ),
    }
    with open(out_path, "w") as fh:
        json.dump(template, fh, indent=2)
    return out_path


# --------------------------------------------------------------------------
# 1.3  phenological offset between the two acquisitions
# --------------------------------------------------------------------------
def phenology_offset_report(lidar_month: int, image_month: int,
                            lidar_year: int, image_year: int) -> Dict:
    """
    Turns the central physical claim of the paper into a number instead of an
    assertion. Leaf-on for temperate deciduous hedgerows in southern England
    runs roughly May to October.
    """
    leaf_on = set(range(5, 11))
    return {
        "lidar": {"year": lidar_year, "month": lidar_month,
                  "state": "leaf-on" if lidar_month in leaf_on else "leaf-off"},
        "imagery": {"year": image_year, "month": image_month,
                    "state": "leaf-on" if image_month in leaf_on else "leaf-off"},
        "months_apart": abs((image_year * 12 + image_month)
                            - (lidar_year * 12 + lidar_month)),
        "states_disagree": (lidar_month in leaf_on) != (image_month in leaf_on),
        "recommended_evidence": (
            "Report mean NDVI from a Sentinel-2 time series for the LiDAR "
            "survey month and for the Pleiades Neo acquisition date over the "
            "same hedgerow mask. The difference between the two is the "
            "quantitative form of the phenological offset."
        ),
    }


# --------------------------------------------------------------------------
# 1.4  co-registration check
# --------------------------------------------------------------------------
def coregistration_check(dem_path: str, ms_path: str,
                         max_shift_px: int = 6) -> Dict:
    """
    Estimates residual misregistration by phase correlation between an
    edge-magnitude image of the elevation layer and of the near-infrared band.

    For a target two to four pixels wide, a one-pixel shift is material, and a
    reviewer will ask for this number.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(dem_path) as dds:
        dem = dds.read(1).astype("float32")
        ref_transform, ref_crs = dds.transform, dds.crs
        H, W = dds.height, dds.width

    with rasterio.open(ms_path) as mds:
        band = min(6, mds.count)          # near infrared in the Pleiades Neo stack
        nir = np.zeros((H, W), dtype="float32")
        reproject(source=rasterio.band(mds, band), destination=nir,
                  src_transform=mds.transform, src_crs=mds.crs,
                  dst_transform=ref_transform, dst_crs=ref_crs,
                  resampling=Resampling.bilinear)

    def edges(a):
        a = np.nan_to_num(a.astype("float32"))
        gy, gx = np.gradient(a)
        e = np.hypot(gx, gy)
        e -= e.mean()
        s = e.std()
        return e / (s + 1e-8)

    # work on a central window to keep the FFT cheap and avoid scene borders
    h = min(2048, H); w = min(2048, W)
    r0 = (H - h) // 2; c0 = (W - w) // 2
    A = edges(dem[r0:r0 + h, c0:c0 + w])
    B = edges(nir[r0:r0 + h, c0:c0 + w])

    Fa = np.fft.rfft2(A)
    Fb = np.fft.rfft2(B)
    cross = Fa * np.conj(Fb)
    cross /= np.abs(cross) + 1e-8                  # phase correlation
    corr = np.fft.irfft2(cross, s=A.shape)
    corr = np.fft.fftshift(corr)

    cy, cx = np.array(corr.shape) // 2
    win = corr[cy - max_shift_px:cy + max_shift_px + 1,
               cx - max_shift_px:cx + max_shift_px + 1]
    dy, dx = np.unravel_index(np.argmax(win), win.shape)
    shift_y = int(dy - max_shift_px)
    shift_x = int(dx - max_shift_px)
    peak = float(win.max() / (np.abs(win).mean() + 1e-12))

    return {
        "shift_rows_px": shift_y,
        "shift_cols_px": shift_x,
        "shift_magnitude_m": float(np.hypot(shift_y, shift_x) * PIXEL_SIZE_M),
        "peak_to_mean_ratio": peak,
        "acceptable": bool(np.hypot(shift_y, shift_x) <= 1.0),
        "note": ("Reported against the elevation grid, which is the reference "
                 "grid used everywhere in the pipeline. The multispectral "
                 "stack is resampled onto it bilinearly in "
                 "inference_patching.read_ms_on_dem_grid, so the same "
                 "resampling should be stated in the paper."),
    }


# --------------------------------------------------------------------------
# 1.5  class support, in pixels and in kilometres
# --------------------------------------------------------------------------
def class_support(npz_dir: str, split_ids: set | None = None) -> Dict:
    """
    Counts, per height class, the number of buffered pixels and the length of
    labelled centreline in kilometres. Per-class F1 is uninterpretable without
    this table and it is currently missing from the manuscript.
    """
    files = sorted(glob(os.path.join(npz_dir, "*.npz")))
    line_px = np.zeros(N_CLASSES, dtype=np.int64)
    buf_px = np.zeros(N_CLASSES, dtype=np.int64)
    n_patches_with = np.zeros(N_CLASSES, dtype=np.int64)
    n_used = 0

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if split_ids is not None and stem not in split_ids:
            continue
        d = np.load(f, allow_pickle=True)
        line = d["label_line"]
        buf = d["label_buffer"]
        n_used += 1
        for c in range(1, N_CLASSES + 1):
            nl = int((line == c).sum())
            nb = int((buf == c).sum())
            line_px[c - 1] += nl
            buf_px[c - 1] += nb
            if nl > 0:
                n_patches_with[c - 1] += 1

    # a single-pixel-wide rasterised line: pixel count is a biased length
    # estimate, so the Kulpa-corrected value from s7 should be preferred once
    # the skeleton is available. Here we give the raw figure and flag it.
    km = line_px * PIXEL_SIZE_M / 1000.0
    total_km = float(km.sum())

    return {
        "n_patches_scanned": n_used,
        "per_class": [
            {
                "channel": i + 1,
                "ukceh_class": CLASS_CODE[i],
                "centreline_px": int(line_px[i]),
                "buffered_px": int(buf_px[i]),
                "length_km_raw": float(km[i]),
                "share_of_labelled_length": float(km[i] / total_km) if total_km else 0.0,
                "patches_containing_class": int(n_patches_with[i]),
            }
            for i in range(N_CLASSES)
        ],
        "total_labelled_length_km_raw": total_km,
        "imbalance_ratio_max_over_min": float(
            km.max() / max(km[km > 0].min(), 1e-9)) if (km > 0).any() else float("nan"),
        "note": ("length_km_raw counts centreline pixels directly and therefore "
                 "overestimates by roughly 5 to 10 percent on diagonal runs. "
                 "Use refpack.s7_length.kulpa_length for the reported figure."),
    }


def patch_inventory(npz_dir: str, splits: Dict[str, set]) -> Dict:
    """Fills the \\NPATCHTRAIN, \\NPATCHVAL, \\NPATCHTEST placeholders."""
    all_files = {os.path.splitext(os.path.basename(f))[0]
                 for f in glob(os.path.join(npz_dir, "*.npz"))}
    out = {"total_npz": len(all_files)}
    for name, ids in splits.items():
        out[name] = len(all_files & set(ids))
    out["unassigned"] = len(all_files - set().union(*splits.values())) if splits else len(all_files)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True)
    ap.add_argument("--ms", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--out", default=REPORT_DIR)
    ap.add_argument("--lidar-month", type=int, default=None)
    ap.add_argument("--lidar-year", type=int, default=None)
    ap.add_argument("--image-month", type=int, default=None)
    ap.add_argument("--image-year", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = {
        "elevation_layer": classify_elevation_layer(args.dem),
        "coregistration": coregistration_check(args.dem, args.ms),
        "class_support": class_support(args.patches),
        "target_construction": {
            "distance_cap_m": DIST_CAP_M, "buffer_m": BUFFER_M,
            "pixel_size_m": PIXEL_SIZE_M,
        },
    }
    if args.lidar_month and args.image_month:
        report["phenology"] = phenology_offset_report(
            args.lidar_month, args.image_month, args.lidar_year, args.image_year)

    gee_provenance_template(os.path.join(args.out, "gee_provenance_TO_FILL.json"))
    path = os.path.join(args.out, "s1_data_audit.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["elevation_layer"]["verdict"], indent=2))
    print("written:", path)


if __name__ == "__main__":
    main()
