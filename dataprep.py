import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio import Affine
from rasterio.warp import reproject as rio_reproject, Resampling as WarpResampling
import tempfile
import fiona
from shapely.geometry import shape, box, mapping, LineString, MultiLineString
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid
import pyproj
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

AUGMENTATIONS = ["non", "rot90", "rot180", "rot270", "vflip", "hflip"]


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class MultispectralProcessor:

    def __init__(
        self,
        dem_path: str,
        multispectral_path: str,
        gpkg_path: str,
        output_dir: str,
        region_id: str,
        patch_size: int = 1024,
        overlap: float = 0.0,
        force_full: bool = True,
        label_field: str = "hghtcls",
        nodata_threshold: float = 0.85,
        buffer_meters: float = 10.0,
        distance_cap_meters: float = 10.0,
    ):
        self.dem_path            = dem_path
        self.multispectral_path  = multispectral_path
        self.gpkg_path           = gpkg_path
        self.output_dir          = output_dir
        self.region_id           = region_id
        self.patch_size          = patch_size
        self.overlap             = overlap
        self.force_full          = force_full
        self.label_field         = label_field
        self.nodata_threshold    = nodata_threshold
        self.buffer_meters       = buffer_meters
        self.distance_cap_meters = distance_cap_meters

        # populated at runtime
        self.base_meta    = {}
        self.crs_wkt      = ""
        self.crs_obj      = None
        self.global_h     = 0
        self.global_w     = 0
        self.pixel_size_x = 1.0
        self.pixel_size_y = 1.0
        self.pixel_size_m = 1.0
        self.hghtcls_map  = {}
        self.hghtcls_key  = ""

        self.npz_dir      = os.path.join(output_dir, region_id, "npz")
        self.clipped_gpkg = os.path.join(output_dir, region_id, "clipped_lines.gpkg")
        os.makedirs(self.npz_dir, exist_ok=True)

    # ── 1. Reference grid from DEM ────────────────────────────────────────────
    def _read_base_transform(self):
        with rasterio.open(self.dem_path) as ds:
            self.base_meta    = ds.meta.copy()
            self.crs_wkt      = ds.crs.to_wkt()
            self.crs_obj      = ds.crs
            self.global_h     = ds.height
            self.global_w     = ds.width
            self.pixel_size_x = abs(ds.transform.a)
            self.pixel_size_y = abs(ds.transform.e)
            self.pixel_size_m = self.pixel_size_x
        print(
            f"   > Reference grid (DEM) : {self.global_w} × {self.global_h} px"
            f"  |  pixel = {self.pixel_size_x:.3f} × {self.pixel_size_y:.3f} m"
            f"  |  CRS : {self.crs_obj.to_epsg() or self.crs_wkt[:60]}"
        )

    # ── 2. Read DEM ───────────────────────────────────────────────────────────
    def _read_dem(self) -> np.ndarray:
        """
        Return (1, H, W) float32.
        Nodata pixels (by nodata value or NaN) are set to 0.0.
        """
        with rasterio.open(self.dem_path) as ds:
            arr    = ds.read(1).astype(np.float32)
            nodata = ds.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        # replace any NaN (from nodata or bilinear edge artifacts) with 0
        arr = np.where(np.isnan(arr), 0.0, arr)
        return arr[np.newaxis, ...]   # (1, H, W) float32, no NaNs

    # ── 3. Read + reproject multispectral ─────────────────────────────────────
    def _read_multispectral(self) -> np.ndarray:
        """
        Warp all bands onto the DEM reference grid.
        Returns (C, H, W) uint16.

        Out-of-bounds pixels after warping are filled by rasterio with 0
        (numpy default for the pre-allocated array).  The actual image may
        also contain edge pixels filled with 1 depending on the source nodata.
        Both are caught in _build_nodata_mask.
        """
        target_transform = self.base_meta["transform"]

        with rasterio.open(self.multispectral_path) as src:
            n_bands = src.count
            arr = np.zeros(
                (n_bands, self.global_h, self.global_w), dtype=np.uint16
            )
            for band_idx in range(1, n_bands + 1):
                rio_reproject(
                    source        = rasterio.band(src, band_idx),
                    destination   = arr[band_idx - 1],
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = target_transform,
                    dst_crs       = self.crs_obj,
                    resampling    = WarpResampling.bilinear,
                )
        print(f"   > Multispectral loaded : {n_bands} bands, dtype uint16")
        return arr   # (C, H, W) uint16

    # ── 4. Nodata mask ────────────────────────────────────────────────────────
    def _build_nodata_mask(self, ms_arr: np.ndarray) -> np.ndarray:
        """
        (H, W) bool mask : True = invalid/nodata, False = valid.

        Two distinct fill regions exist after warping onto the DEM grid:
          - Pixels where ALL bands == 0  (numpy pre-fill for out-of-bounds)
          - Pixels where ALL bands == 1  (source image nodata value)
        Both must be masked out.  Any real pixel will have at least one band
        with a value other than 0 or 1 in uint16 DN space.
        """
        all_zero = np.all(ms_arr == 0, axis=0)
        all_one  = np.all(ms_arr == 1, axis=0)
        nodata_mask = all_zero | all_one   # (H, W) bool

        invalid_px = int(nodata_mask.sum())
        total_px   = int(nodata_mask.size)
        print(
            f"   > Nodata mask          : {invalid_px:,} invalid px"
            f"  /  {total_px:,} total"
            f"  ({100 * invalid_px / total_px:.1f} % nodata)"
            f"  [{int(all_zero.sum()):,} zero-fill  +  {int(all_one.sum()):,} one-fill]"
        )
        return nodata_mask   # bool, True = nodata

    # ── 5. Per-band statistics (whole scene, valid pixels only) ───────────────
    def _compute_band_statistics(
        self,
        ms_arr: np.ndarray,       # (C, H, W) uint16
        dem_arr: np.ndarray,      # (1, H, W) float32, no NaNs
        nodata_mask: np.ndarray,  # (H, W) bool, True = nodata
    ) -> pd.DataFrame:
        """
        Compute per-band mean, std, min, max over the entire scene for both
        multispectral and DEM, excluding nodata pixels.

        Saves band_statistics.csv with a 'source' column so dataset.py can
        filter by source when loading normalisation stats for each modality.
        """
        valid_mask = ~nodata_mask   # (H, W) bool

        records = []

        for i in range(ms_arr.shape[0]):
            band_data = ms_arr[i][valid_mask].astype(np.float64)
            records.append({
                "source"        : "multispectral",
                "band"          : i + 1,
                "mean"          : float(band_data.mean()),
                "std"           : float(band_data.std()),
                "min"           : float(band_data.min()),
                "max"           : float(band_data.max()),
                "valid_px_count": int(band_data.size),
            })

        for i in range(dem_arr.shape[0]):
            band_data = dem_arr[i][valid_mask].astype(np.float64)
            records.append({
                "source"        : "dem",
                "band"          : i + 1,
                "mean"          : float(band_data.mean()),
                "std"           : float(band_data.std()),
                "min"           : float(band_data.min()),
                "max"           : float(band_data.max()),
                "valid_px_count": int(band_data.size),
            })

        df = pd.DataFrame(records)
        print("\n   > Per-band statistics (valid pixels only):")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        stats_path = os.path.join(self.output_dir, self.region_id, "band_statistics.csv")
        df.to_csv(stats_path, index=False)
        print(f"   > Saved → {stats_path}\n")
        return df

    # ── 6. Clip GeoPackage to DEM extent ──────────────────────────────────────
    def _clip_and_save_gpkg(self):
        if os.path.exists(self.clipped_gpkg):
            print("   > Clipped GeoPackage already on disk – skipping clip.")
            return

        print("   > Clipping GeoPackage to DEM extent (first layer only) ...")
        print("     This may take a while for a large file – please be patient.")

        if not os.path.exists(self.gpkg_path):
            raise FileNotFoundError(f"GPKG not found: {self.gpkg_path}")

        try:
            layers = fiona.listlayers(self.gpkg_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to list layers in GPKG: {exc}") from exc

        if not layers:
            raise RuntimeError(f"No layers found in GPKG: {self.gpkg_path}")

        src_layer = layers[0]
        print(f"   > Using layer : '{src_layer}'")

        with rasterio.open(self.dem_path) as ds:
            dem_bounds = ds.bounds
            dem_bbox   = box(*dem_bounds)

        dem_crs = pyproj.CRS(self.crs_wkt)

        with fiona.open(self.gpkg_path, layer=src_layer) as src:
            src_crs_wkt = src.crs_wkt or None
            schema      = src.schema.copy()
            out_geom    = schema.get("geometry", "MultiLineString")
            props       = schema.get("properties", {}).copy()

            if src_crs_wkt:
                src_crs        = pyproj.CRS(src_crs_wkt)
                need_reproject = not src_crs.equals(dem_crs)
            else:
                print("   > WARNING: source GPKG has no CRS – assuming it matches DEM.")
                src_crs        = dem_crs
                need_reproject = False

            if need_reproject:
                to_src_crs = pyproj.Transformer.from_crs(
                    dem_crs, src_crs, always_xy=True
                ).transform
                to_dem_crs = pyproj.Transformer.from_crs(
                    src_crs, dem_crs, always_xy=True
                ).transform
                bbox_in_src   = shp_transform(to_src_crs, dem_bbox)
                filter_bounds = bbox_in_src.bounds
            else:
                to_dem_crs    = None
                filter_bounds = tuple(dem_bounds)

            features = []
            for feat in tqdm(src.filter(bbox=filter_bounds), desc="   Clipping features"):
                raw_geom = feat.get("geometry")
                if not raw_geom:
                    continue
                try:
                    geom = shape(raw_geom)
                except Exception:
                    continue
                if geom is None or geom.is_empty:
                    continue

                if to_dem_crs:
                    geom = shp_transform(to_dem_crs, geom)

                if not geom.intersects(dem_bbox):
                    continue
                geom = geom.intersection(dem_bbox)
                if geom.is_empty:
                    continue

                if not geom.is_valid:
                    try:
                        geom = make_valid(geom)
                    except Exception:
                        pass

                if out_geom.lower().startswith("multilinestring"):
                    if isinstance(geom, LineString):
                        geom = MultiLineString([geom])
                    elif isinstance(geom, MultiLineString):
                        pass
                    else:
                        lines = []
                        for g in getattr(geom, "geoms", [geom]):
                            if isinstance(g, LineString):
                                lines.append(g)
                            elif isinstance(g, MultiLineString):
                                lines.extend(list(g.geoms))
                        if not lines:
                            continue
                        geom = MultiLineString(lines)
                elif out_geom.lower().startswith("linestring"):
                    if isinstance(geom, MultiLineString):
                        parts = list(geom.geoms)
                        if not parts:
                            continue
                        geom = parts[0]
                    elif not isinstance(geom, LineString):
                        continue

                features.append({
                    "type"      : "Feature",
                    "id"        : feat.get("id"),
                    "properties": dict(feat.get("properties", {})),
                    "geometry"  : mapping(geom),
                })

        if not features:
            raise RuntimeError(
                "No features remained after clipping. "
                "Check that the GPKG overlaps the DEM extent."
            )

        print(f"   > {len(features)} polyline feature(s) inside DEM extent")

        out_dir = os.path.dirname(self.clipped_gpkg)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".gpkg", dir=out_dir)
        os.close(tmp_fd)
        os.remove(tmp_path)

        out_schema = {"geometry": out_geom, "properties": props}
        layer_name = os.path.splitext(os.path.basename(self.clipped_gpkg))[0]

        try:
            with fiona.open(
                tmp_path, "w",
                driver = "GPKG",
                layer  = layer_name,
                crs    = dem_crs.to_wkt(),
                schema = out_schema,
            ) as dst:
                for feat in tqdm(features, desc="   Saving clipped GPKG"):
                    dst.write(feat)

            if os.path.exists(self.clipped_gpkg):
                os.remove(self.clipped_gpkg)
            os.replace(tmp_path, self.clipped_gpkg)

        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Failed to write clipped GPKG: {exc}") from exc

        print(f"   > Saved → {self.clipped_gpkg}")

    # ── 7. Auto-discover hghtcls string values ────────────────────────────────
    def _build_hghtcls_map(self):
        values = set()
        with fiona.open(self.clipped_gpkg, layer=0) as src:
            schema_props = list(src.schema["properties"].keys())
            key = next(
                (k for k in schema_props if k.lower() == self.label_field.lower()),
                None,
            )
            if key is None:
                raise ValueError(
                    f"Field '{self.label_field}' not found in GeoPackage. "
                    f"Available columns: {schema_props}"
                )
            self.hghtcls_key = key

            for feat in src:
                val = feat["properties"].get(key)
                if val is not None:
                    values.add(str(val).strip())

        self.hghtcls_map = {v: i + 1 for i, v in enumerate(sorted(values))}
        print(f"   > hghtcls mapping (string → integer):")
        for k, v in self.hghtcls_map.items():
            print(f"        {v:>3}  →  '{k}'")

    # ── 8. Rasterize polylines ────────────────────────────────────────────────
    def _rasterize_labels(self) -> tuple:
        """
        Returns three arrays on the full DEM grid:
          label_line     : (H, W) uint16  — class index burned along exact polyline pixels
          label_buffer   : (H, W) uint16  — class index burned into buffered polygon
          label_distance : (H, W) float32 — distance heat-map capped at distance_cap_meters
        """
        print("   > Rasterizing polylines (line / buffer / distance) ...")

        line_shapes   = []
        buffer_shapes = []
        binary_shapes = []

        with fiona.open(self.clipped_gpkg, layer=0) as src:
            for feat in tqdm(src, desc="   Building shapes"):
                geom    = shape(feat["geometry"])
                val_str = str(feat["properties"].get(self.hghtcls_key, "")).strip()
                int_val = self.hghtcls_map.get(val_str, 0)
                if int_val == 0:
                    continue

                line_shapes.append((geom, int_val))
                buffer_shapes.append((geom.buffer(self.buffer_meters), int_val))
                binary_shapes.append((geom, 1))

        tr = self.base_meta["transform"]
        sh = (self.global_h, self.global_w)

        if line_shapes:
            label_line = rasterize(
                line_shapes, out_shape=sh, transform=tr, fill=0, dtype=np.uint16,
            )
        else:
            label_line = np.zeros(sh, dtype=np.uint16)

        if buffer_shapes:
            label_buffer = rasterize(
                buffer_shapes, out_shape=sh, transform=tr, fill=0, dtype=np.uint16,
            )
        else:
            label_buffer = np.zeros(sh, dtype=np.uint16)

        if binary_shapes:
            binary_line = rasterize(
                binary_shapes, out_shape=sh, transform=tr, fill=0, dtype=np.uint16,
            )
            pixel_dist = distance_transform_edt(
                1 - binary_line,
                sampling=(self.pixel_size_y, self.pixel_size_x),
            )
            label_distance = np.where(
                pixel_dist <= self.distance_cap_meters,
                self.distance_cap_meters - pixel_dist,
                0.0,
            ).astype(np.float32)
        else:
            label_distance = np.zeros(sh, dtype=np.float32)

        print(
            f"   > Line non-zero px     : {int((label_line > 0).sum()):,}\n"
            f"   > Buffer non-zero px   : {int((label_buffer > 0).sum()):,}\n"
            f"   > Distance non-zero px : {int((label_distance > 0).sum()):,}"
        )
        return label_line, label_buffer, label_distance

    # ── 9. Main patching loop ─────────────────────────────────────────────────
    def execute_patching(self):
        print(f"\n{'='*60}")
        print(f"  Region : {self.region_id}")
        print(f"{'='*60}")

        # Step 1 – reference grid
        self._read_base_transform()

        # Step 2 – clip GeoPackage
        self._clip_and_save_gpkg()

        # Step 3 – hghtcls map
        self._build_hghtcls_map()

        # Step 4 – rasterize labels on full grid
        label_line, label_buffer, label_distance = self._rasterize_labels()

        # Step 5 – load DEM (before stats)
        print("   > Reading DEM ...")
        dem_arr = self._read_dem()           # (1, H, W) float32, no NaNs

        # Step 6 – load multispectral
        print("   > Reading Multispectral ...")
        ms_arr = self._read_multispectral()  # (C, H, W) uint16

        # Step 7 – nodata mask: all-zero OR all-one across bands
        print("   > Building nodata mask ...")
        nodata_mask = self._build_nodata_mask(ms_arr)   # (H, W) bool, True = nodata

        # Step 8 – zero out labels in nodata region
        label_line    [nodata_mask] = 0
        label_buffer  [nodata_mask] = 0
        label_distance[nodata_mask] = 0.0

        # Step 9 – whole-scene per-band statistics (valid pixels only)
        print("   > Computing per-band statistics ...")
        self._compute_band_statistics(ms_arr, dem_arr, nodata_mask)

        # Step 10 – patch grid
        step = int(self.patch_size * (1 - self.overlap))
        rows = list(range(0, self.global_h - self.patch_size + 1, step))
        cols = list(range(0, self.global_w - self.patch_size + 1, step))

        if self.force_full:
            last_r = self.global_h - self.patch_size
            last_c = self.global_w - self.patch_size
            if last_r >= 0 and (not rows or rows[-1] != last_r):
                rows.append(last_r)
            if last_c >= 0 and (not cols or cols[-1] != last_c):
                cols.append(last_c)

        rows = sorted(set(r for r in rows if 0 <= r <= self.global_h - self.patch_size))
        cols = sorted(set(c for c in cols if 0 <= c <= self.global_w - self.patch_size))

        total        = len(rows) * len(cols)
        base_tr      = self.base_meta["transform"]
        hghtcls_map_str = str(self.hghtcls_map)

        print(f"   > Writing up to {total} patches ({len(rows)} rows × {len(cols)} cols) ...")
        skipped = 0

        with tqdm(total=total) as pbar:
            for r in rows:
                for c in cols:
                    ps = self.patch_size

                    patch_mask   = nodata_mask[r:r+ps, c:c+ps]
                    nodata_ratio = float(patch_mask.mean())
                    if nodata_ratio > self.nodata_threshold:
                        skipped += 1
                        pbar.update(1)
                        continue

                    fname    = f"{self.region_id}_R{r:05d}_C{c:05d}.npz"
                    out_path = os.path.join(self.npz_dir, fname)
                    patch_tr = base_tr * Affine.translation(c, r)

                    meta_dict = {
                        "row"          : r,
                        "col"          : c,
                        "height"       : ps,
                        "width"        : ps,
                        "transform"    : tuple(patch_tr)[:6],
                        "crs"          : self.crs_wkt,
                        "region_id"    : self.region_id,
                        "hghtcls_map"  : hghtcls_map_str,
                        "buffer_m"     : self.buffer_meters,
                        "dist_cap_m"   : self.distance_cap_meters,
                        "pixel_size_x" : self.pixel_size_x,
                        "pixel_size_y" : self.pixel_size_y,
                    }

                    np.savez_compressed(
                        out_path,
                        multispectral  = ms_arr[:,        r:r+ps, c:c+ps],   # (C, ps, ps) uint16
                        dem            = dem_arr[:,        r:r+ps, c:c+ps],   # (1, ps, ps) float32
                        label_line     = label_line[       r:r+ps, c:c+ps],   # (ps, ps)    uint16
                        label_buffer   = label_buffer[     r:r+ps, c:c+ps],   # (ps, ps)    uint16
                        label_distance = label_distance[   r:r+ps, c:c+ps],   # (ps, ps)    float32
                        mask           = patch_mask,                            # (ps, ps)    bool
                        meta           = np.array(meta_dict, dtype=object),
                    )
                    pbar.update(1)

        saved = total - skipped
        print(
            f"   > Done.  Saved {saved} patches, "
            f"skipped {skipped} (>{self.nodata_threshold*100:.0f}% nodata)."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class DatasetSimulator:
    """
    Reads NPZ patches, extracts 5 sub-patches each, explodes with 6
    augmentations and writes train / val / test CSVs.

    is_empty  = 1  iff label_line, label_buffer AND label_distance are all zero
    is_masked = 1  iff > 50 % of the sub-patch pixels are nodata
    """

    def __init__(
        self,
        root_dir:   str,
        region_id:  str,
        patch_size: int = 1024,
        sub_size:   int = 512,
    ):
        self.root_dir   = root_dir
        self.region_id  = region_id
        self.npz_dir    = os.path.join(root_dir, region_id, "npz")
        self.patch_size = patch_size
        self.sub_size   = sub_size

        half = patch_size // 2
        q    = patch_size // 4
        s    = sub_size

        self.sub_coords = {
            "tl": (0,    s,      0,      s    ),
            "tr": (0,    s,      half,   half+s),
            "bl": (half, half+s, 0,      s    ),
            "br": (half, half+s, half,   half+s),
            "ce": (q,    q+s,    q,      q+s  ),
        }

    def analyze_and_generate(self):
        files = sorted(f for f in os.listdir(self.npz_dir) if f.endswith(".npz"))
        if not files:
            print("   > No NPZ files found – nothing to do.")
            return

        print(f"\n   > Analysing {len(files)} NPZ patches ...")
        records = []

        for fname in tqdm(files):
            fpath = os.path.join(self.npz_dir, fname)
            try:
                data           = np.load(fpath, allow_pickle=True)
                lbl_line_full  = data["label_line"]
                lbl_buf_full   = data["label_buffer"]
                lbl_dist_full  = data["label_distance"]
                mask_full      = data["mask"]
            except Exception as exc:
                print(f"   > Skipping {fname}: {exc}")
                continue

            file_has_labels = int(
                np.any(lbl_line_full > 0) or
                np.any(lbl_buf_full  > 0) or
                np.any(lbl_dist_full > 0)
            )
            rel_path = os.path.join(self.region_id, "npz", fname)

            for sub_name, (r0, r1, c0, c1) in self.sub_coords.items():
                sub_line = lbl_line_full[ r0:r1, c0:c1]
                sub_buf  = lbl_buf_full[  r0:r1, c0:c1]
                sub_dist = lbl_dist_full[ r0:r1, c0:c1]
                sub_msk  = mask_full[     r0:r1, c0:c1]

                any_label = (
                    np.any(sub_line > 0) or
                    np.any(sub_buf  > 0) or
                    np.any(sub_dist > 0)
                )

                records.append({
                    "Sample_ID"      : fname[:-4],
                    "SubPatch"       : sub_name,
                    "Rel_Path"       : rel_path,
                    "File_Has_Labels": file_has_labels,
                    "is_empty"       : int(not any_label),
                    "is_masked"      : int(sub_msk.mean() > 0.5),
                })

        df = pd.DataFrame(records)
        self._split_and_save(df)

    def _split_and_save(self, df: pd.DataFrame):
        if df.empty:
            print("   > Warning: DataFrame is empty – no CSVs written.")
            return

        labeled_ids = df[df["File_Has_Labels"] == 1]["Sample_ID"].unique()
        empty_ids   = df[df["File_Has_Labels"] == 0]["Sample_ID"].unique()

        if len(labeled_ids) >= 3:
            train_lbl, tv      = train_test_split(labeled_ids, test_size=0.30, random_state=42)
            val_lbl,  test_lbl = train_test_split(tv,          test_size=0.50, random_state=42)
        elif len(labeled_ids) == 2:
            train_lbl, val_lbl, test_lbl = labeled_ids[:1], labeled_ids[1:], []
        elif len(labeled_ids) == 1:
            train_lbl, val_lbl, test_lbl = labeled_ids, [], []
        else:
            train_lbl, val_lbl, test_lbl = [], [], []

        train_ids = np.concatenate([train_lbl, empty_ids])

        splits = {
            "train": df[df["Sample_ID"].isin(train_ids)],
            "val"  : df[df["Sample_ID"].isin(val_lbl)],
            "test" : df[df["Sample_ID"].isin(test_lbl)],
        }

        out_base = os.path.join(self.root_dir, self.region_id)
        print("   > Saving CSVs (with augmentation explosion) ...")

        for split_name, split_df in splits.items():
            if split_df.empty:
                print(f"      {split_name}: 0 rows (skipped)")
                continue

            aug_rows = []
            for _, row in split_df.iterrows():
                for aug in AUGMENTATIONS:
                    r               = row.to_dict()
                    r["Augmentation"] = aug
                    aug_rows.append(r)

            out_csv = os.path.join(out_base, f"{split_name}_grouped.csv")
            pd.DataFrame(aug_rows).to_csv(out_csv, index=False)
            print(
                f"      {split_name}: {len(aug_rows):>7} rows"
                f"  ({len(split_df)} sub-patches × {len(AUGMENTATIONS)} augs)"
                f"  →  {out_csv}"
            )

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── paths ─────────────────────────────────────────────────────────────────
    DEM_PATH    = r"E:\Hedge\Classification\Classification\Scene_2_Height.tif"
    MS_PATH     = r"E:\Hedge\Classification\Classification\scene2.tif"
    GPKG_PATH   = r"E:\Hedge\Classification\Classification\lcplus-hedges_5527321.gpkg"
    OUTPUT_DIR  = r"E:\Hedge\Classification\Classification\Scene2"
    REGION_ID   = "Scene2"

    # ── patch settings ────────────────────────────────────────────────────────
    PATCH_SIZE  = 512
    SUB_SIZE    = 256
    OVERLAP     = 0.0

    # ── label settings ────────────────────────────────────────────────────────
    LABEL_FIELD       = "hghtcls"   # column name in the GeoPackage
    BUFFER_METERS     = 5.0        # buffer radius for label_buffer
    DISTANCE_CAP_M    = 5.0        # max metres stored in label_distance
    NODATA_THRESHOLD  = 0.85        # skip patch if > 85 % pixels are nodata

    # ── run ───────────────────────────────────────────────────────────────────
    processor = MultispectralProcessor(
        dem_path           = DEM_PATH,
        multispectral_path = MS_PATH,
        gpkg_path          = GPKG_PATH,
        output_dir         = OUTPUT_DIR,
        region_id          = REGION_ID,
        patch_size         = PATCH_SIZE,
        overlap            = OVERLAP,
        force_full         = True,
        label_field        = LABEL_FIELD,
        nodata_threshold   = NODATA_THRESHOLD,
        buffer_meters      = BUFFER_METERS,
        distance_cap_meters= DISTANCE_CAP_M,
    )
    processor.execute_patching()

    simulator = DatasetSimulator(
        root_dir   = OUTPUT_DIR,
        region_id  = REGION_ID,
        patch_size = PATCH_SIZE,
        sub_size   = SUB_SIZE,
    )
    simulator.analyze_and_generate()
