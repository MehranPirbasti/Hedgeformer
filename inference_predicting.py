
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, load_checkpoint


def infer_num_classes(state_dict):
    for key in ("seg_head.head.3.weight", "seg_head_dem.head.3.weight"):   # second one: the 'Late' variant
        if key in state_dict:
            return state_dict[key].shape[0]
    raise KeyError("Could not infer num_classes from the checkpoint — pass num_classes explicitly instead.")


def predict_tiles(work_dir, weights_path, variant, batch_size=8, device=None, num_classes=None):
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    tiles_dir = os.path.join(work_dir, "tiles")
    preds_dir = os.path.join(work_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    state_dict = torch.load(weights_path, map_location=device)
    if num_classes is None:
        num_classes = infer_num_classes(state_dict)
    print(f"num_classes: {num_classes}")

    # variant MUST be the architecture the checkpoint was trained with (Simple | SRA | TTT | Early | Late |
    # DEM_only | MS_only).  Loading is strict: the old code built a TTT model with strict=False, so a checkpoint of
    # another variant silently left the mismatched layers randomly initialised.
    model = build_model(variant, num_classes, pretrained=False).to(device)
    load_checkpoint(model, state_dict, map_location=device)
    model.eval()

    tile_paths = sorted(glob.glob(os.path.join(tiles_dir, "tile_*.npz")))
    print(f"found {len(tile_paths)} tiles")

    autocast_enabled = device.type == "cuda"
    with torch.no_grad():
        for batch_start in range(0, len(tile_paths), batch_size):
            batch_paths = tile_paths[batch_start:batch_start + batch_size]
            ms_list, dem_list, rc_list = [], [], []
            for p in batch_paths:
                data = np.load(p, allow_pickle=True)
                ms_list.append(data["ms"])
                dem_list.append(data["dem"])
                rc_list.append((int(data["row"]), int(data["col"])))

            ms_t = torch.from_numpy(np.stack(ms_list, axis=0)).to(device)
            dem_t = torch.from_numpy(np.stack(dem_list, axis=0)).to(device)

            with torch.autocast(device_type="cuda", enabled=autocast_enabled):
                out = model(dem_t, ms_t)
            logits = out["seg_logits"]  # (B, num_classes, h, w)

            patch_size = ms_t.shape[-1]
            if logits.shape[-2:] != (patch_size, patch_size):
                logits = F.interpolate(
                    logits, size=(patch_size, patch_size), mode="bilinear", align_corners=False
                )

            prob = torch.sigmoid(logits).float().cpu().numpy()  # (B, C, h, w)

            for (r, c), p_arr in zip(rc_list, prob):
                out_path = os.path.join(preds_dir, f"pred_r{r:05d}_c{c:05d}.npz")
                np.savez(out_path, prob=p_arr, row=r, col=c, num_classes=num_classes)

            done = min(batch_start + batch_size, len(tile_paths))
            print(f"  tiles {done}/{len(tile_paths)}", end="\r")
    print()


if __name__ == "__main__":
    predict_tiles(
        work_dir="E:/Hedge/Classification/Classification/Codes/work_scene2",
        # weights_path="C:/Users/fs-home 3/Desktop/best_frechet_116.8427_ema.pth",
        # weights_path="C:/Users/fs-home 3/Desktop/best_f1_0.4380_ema.pth",
        weights_path=r"E:\Hedge\Classification\Classification\Codes\checkpoints_hedge_Simple\best_f1_0.4377_ema.pth",
        variant="Simple",          # the checkpoint above was trained as 'Simple' (concat fusion)
        batch_size=8,
        device="cuda",
    )