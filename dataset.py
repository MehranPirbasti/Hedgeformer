import ast
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_SUB_COORDS = {
    'tl': (0,   256, 0,   256),
    'tr': (0,   256, 256, 512),
    'bl': (256, 512, 0,   256),
    'br': (256, 512, 256, 512),
    'ce': (128, 384, 128, 384),
}

def load_band_statistics(stats_path: str) -> tuple:
    df = pd.read_csv(stats_path)

    ms_df  = df[df["source"] == "multispectral"].sort_values("band")
    dem_df = df[df["source"] == "dem"].sort_values("band")

    ms_mean  = ms_df["mean"].to_numpy(dtype=np.float32)
    ms_std   = ms_df["std"].to_numpy(dtype=np.float32)
    dem_mean = dem_df["mean"].to_numpy(dtype=np.float32)
    dem_std  = dem_df["std"].to_numpy(dtype=np.float32)

    return ms_mean, ms_std, dem_mean, dem_std


def discover_class_maps(base_dir: str, scenes: list) -> dict:
    """Reads the string->int height-class map that dataprep.py stored in the NPZ metadata of each scene."""
    maps = {}
    for scene in scenes:
        npz_dir = os.path.join(base_dir, scene, "npz")
        if not os.path.isdir(npz_dir):
            continue
        files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
        if not files:
            continue
        meta = np.load(os.path.join(npz_dir, files[0]), allow_pickle=True)["meta"].item()
        raw = meta.get("hghtcls_map")
        if raw is not None:
            maps[scene] = ast.literal_eval(raw) if isinstance(raw, str) else dict(raw)
    return maps


def discover_num_classes(base_dir: str, scenes: list, sample_files_per_scene: int = 5) -> int:
    """
    Number of height classes = max integer id in the hghtcls_map stored in the NPZ metadata (this cannot
    miss a rare class the way sampling a few patches' label_buffer could).  Falls back to scanning
    label_buffer of a few patches if the metadata is absent.  Prints the map of every scene and refuses to
    continue if scenes disagree, because the integer ids are built per scene by sorting the class strings.
    """
    maps = discover_class_maps(base_dir, scenes)
    if maps:
        ref_scene, ref = next(iter(maps.items()))
        for sc, m in maps.items():
            print(f"   > class map [{sc}]: {m}")
            if m != ref:
                raise RuntimeError(f"hghtcls_map differs between scenes ({ref_scene} vs {sc}): {ref} vs {m}. "
                                   "Class ids would not mean the same height class in both scenes.")
        return int(max(ref.values()))
    max_cls = 0
    for scene in scenes:
        npz_dir = os.path.join(base_dir, scene, "npz")
        if not os.path.isdir(npz_dir):
            continue
        files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
        for fname in files[:sample_files_per_scene]:
            data = np.load(os.path.join(npz_dir, fname), allow_pickle=True)
            buf = data["label_buffer"]
            if buf.size:
                max_cls = max(max_cls, int(buf.max()))
    if max_cls == 0:
        raise RuntimeError(
            f"discover_num_classes found no nonzero label_buffer values in scenes {scenes}. "
            "Check base_dir/scenes paths."
        )
    return max_cls


class SensorJitter:
    def __init__(
        self,
        p: float = 0.5,
        bias_range: float = 50.0,
        gain_range: float = 0.08,
        noise_std: float = 15.0,
        dropout_p: float = 0.001,
        dem_noise_std: float = 3.0,
        swap_p: float = 0.0,
    ):
        self.p           = p
        self.bias_range  = bias_range
        self.gain_range  = gain_range
        self.noise_std   = noise_std
        self.dropout_p   = dropout_p
        self.swap_p      = swap_p
        self.dem_noise_std = dem_noise_std

    def __call__(self, ms: np.ndarray) -> np.ndarray:
        """ms : (C, H, W) float32 in raw DN space."""
        if np.random.rand() > self.p:
            return ms

        C, H, W = ms.shape

        if np.random.rand() < 0.7:
            bias = np.random.uniform(-self.bias_range, self.bias_range, (C, 1, 1)).astype(np.float32)
            ms   = ms + bias

        if np.random.rand() < 0.7:
            gain = np.random.uniform(1.0 - self.gain_range, 1.0 + self.gain_range, (C, 1, 1)).astype(np.float32)
            ms   = ms * gain

        if np.random.rand() < 0.8:
            noise = np.random.normal(0.0, self.noise_std, ms.shape).astype(np.float32)
            ms    = ms + noise

        if np.random.rand() < 0.4:
            dead = np.random.rand(H, W) < self.dropout_p
            ms[:, dead] = 0.0

        if C >= 2 and np.random.rand() < self.swap_p:
            i, j       = np.random.choice(C, size=2, replace=False)
            ms[[i, j]] = ms[[j, i]]

        return np.clip(ms, 0.0, 65535.0)

    def jitter_dem(self, dem: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return dem

        noise = np.random.normal(0.0, self.dem_noise_std, dem.shape).astype(np.float32)
        return dem + noise


def _apply_csv_aug(
    ms: np.ndarray,          # (C, H, W)
    dem: np.ndarray,         # (1, H, W)
    dist: np.ndarray,        # (H, W)  merged capped distance (meters)
    mask: np.ndarray,        # (H, W)
    buffer: np.ndarray,      # (H, W)  class id per pixel (buffered polygons), 0 = background
    label_line: np.ndarray,  # (H, W)  class id per pixel (exact line pixels), 0 = background
    aug: str,
) -> tuple:
    """
    NOTE: the original version of this function rotated the skeleton array by
    k=1 in the 'rot270' branch instead of k=3 - a real bug, fixed here while
    threading `buffer` and `label_line` through the same transforms.
    """
    if aug == 'non':
        return ms, dem, dist, mask, buffer, label_line

    k_map = {'rot90': 1, 'rot180': 2, 'rot270': 3}
    if aug in k_map:
        k          = k_map[aug]
        ms         = np.rot90(ms,         k=k, axes=(1, 2)).copy()
        dem        = np.rot90(dem,        k=k, axes=(1, 2)).copy()
        dist       = np.rot90(dist,       k=k).copy()
        mask       = np.rot90(mask,       k=k).copy()
        buffer     = np.rot90(buffer,     k=k).copy()
        label_line = np.rot90(label_line, k=k).copy()
    elif aug == 'vflip':
        ms         = ms[:,  ::-1, :].copy()
        dem        = dem[:, ::-1, :].copy()
        dist       = dist[::-1, :].copy()
        mask       = mask[::-1, :].copy()
        buffer     = buffer[::-1, :].copy()
        label_line = label_line[::-1, :].copy()
    elif aug == 'hflip':
        ms         = ms[:,  :, ::-1].copy()
        dem        = dem[:, :, ::-1].copy()
        dist       = dist[:, ::-1].copy()
        mask       = mask[:, ::-1].copy()
        buffer     = buffer[:, ::-1].copy()
        label_line = label_line[:, ::-1].copy()
    return ms, dem, dist, mask, buffer, label_line


class HedgeDistanceDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        dataframe: pd.DataFrame,
        num_classes: int,
        stats_path: str = None,
        split: str = 'train',
        img_size: int = 512,
        jitter_cfg: dict = None,
    ):
        self.base_dir    = base_dir
        self.df          = dataframe.reset_index(drop=True)
        self.split       = split
        self.img_size    = img_size
        self.num_classes = num_classes

        self._stats_cache         = {}
        self._fallback_stats_path = stats_path

        jcfg = jitter_cfg or {}
        self.jitter = SensorJitter(
            p          = jcfg.get('p',           0.5)  if split == 'train' else 0.0,
            bias_range = jcfg.get('bias_range',  50.0),
            gain_range = jcfg.get('gain_range',  0.08),
            noise_std  = jcfg.get('noise_std',   15.0),
            dropout_p  = jcfg.get('dropout_p',   0.001),
            swap_p     = jcfg.get('swap_p',       0.0),
            dem_noise_std = jcfg.get('dem_noise_std', 3.0),   # BUGFIX: was never forwarded (default 3.0 used)
        )

    def __len__(self) -> int:
        return len(self.df)

    def _scene_stats(self, scene_id: str) -> dict:
        if scene_id not in self._stats_cache:
            path = os.path.join(self.base_dir, scene_id, "band_statistics.csv")
            if not os.path.exists(path) and self._fallback_stats_path:
                path = self._fallback_stats_path
            ms_mean, ms_std, dem_mean, dem_std = load_band_statistics(path)
            self._stats_cache[scene_id] = dict(
                ms_mean  = ms_mean.reshape(-1, 1, 1).astype(np.float32),
                ms_std   = ms_std.reshape(-1, 1, 1).astype(np.float32),
                dem_mean = dem_mean.reshape(-1, 1, 1).astype(np.float32),
                dem_std  = dem_std.reshape(-1, 1, 1).astype(np.float32),
            )
            print(f"   > Loaded band statistics for scene '{scene_id}' from {path}")
        return self._stats_cache[scene_id]

    def __getitem__(self, idx: int):
        row       = self.df.iloc[idx]
        rel_path  = row['Rel_Path'].replace('\\', '/')
        sub_name  = row['SubPatch']
        aug       = row.get('Augmentation', 'non')
        sample_id = f"{row['Sample_ID']}_{sub_name}_{aug}"

        scene_id = rel_path.split('/')[0]
        stats    = self._scene_stats(scene_id)

        npz_path = os.path.join(self.base_dir, rel_path)
        data     = np.load(npz_path, allow_pickle=True)

        ms_full         = data['multispectral'].astype(np.float32)
        dem_full        = data['dem'].astype(np.float32)
        dist_full       = data['label_distance'].astype(np.float32)   # (H, W) merged capped dist (m)
        mask_full       = data['mask'].astype(np.float32)             # (H, W)
        buffer_full     = data['label_buffer'].astype(np.float32)     # (H, W) class id, 0 = bg
        label_line_full = data['label_line'].astype(np.float32)       # (H, W) class id, 0 = bg

        valid_full = 1.0 - mask_full   # (H, W)
        ms_full         *= valid_full
        dem_full        *= valid_full
        dist_full       *= valid_full          # NOTE: original code did `valid_full[0]` here - a bug
                                                # (only row 0 of the mask got applied). Fixed.
        buffer_full     *= valid_full
        label_line_full *= valid_full

        r0, r1, c0, c1 = _SUB_COORDS.get(sub_name, _SUB_COORDS['ce'])
        ms         = ms_full[:,  r0:r1, c0:c1]
        dem        = dem_full[:, r0:r1, c0:c1]
        dist       = dist_full[  r0:r1, c0:c1]
        mask       = mask_full[  r0:r1, c0:c1]
        buffer     = buffer_full[r0:r1, c0:c1]
        label_line = label_line_full[r0:r1, c0:c1]

        ms, dem, dist, mask, buffer, label_line = _apply_csv_aug(
            ms, dem, dist, mask, buffer, label_line, aug
        )

        ms  = self.jitter(ms)
        dem = self.jitter.jitter_dem(dem)
        ms_norm  = (ms  - stats['ms_mean'])  / (stats['ms_std']  + 1e-6)   # (C, H, W)
        dem_norm = (dem - stats['dem_mean']) / (stats['dem_std'] + 1e-6)   # (1, H, W)

        dist_cap = float(data['meta'].item().get('dist_cap_m', 10.0))

        dist_stack = np.zeros((self.num_classes,) + dist.shape, dtype=np.float32)
        skel_stack = np.zeros((self.num_classes,) + dist.shape, dtype=np.float32)
        for c in range(1, self.num_classes + 1):
            dist_stack[c - 1] = dist * (buffer == c)
            skel_stack[c - 1] = (label_line == c).astype(np.float32)

        dist_norm = np.clip(dist_stack / dist_cap, 0.0, 1.0)   # (N, H, W)

        return (
            torch.from_numpy(ms_norm),
            torch.from_numpy(dem_norm),
            torch.from_numpy(dist_norm),
            torch.from_numpy(mask),
            torch.from_numpy(skel_stack),
            sample_id,
        )