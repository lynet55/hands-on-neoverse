"""
Stage 1 of the keyframe-grid figure pipeline.

For a single clip / stream and a fixed list of keyframes, run BOTH NeoVerse
segmentation variants (Naive Gaussian = DPT/hand_pred_head, Gaussian Params =
rasterized per-Gaussian mask logits) and dump per-frame artifacts to a shared
directory:

    <out_dir>/raw_<fi>.png        raw input frame
    <out_dir>/naive_<fi>.png      Naive Gaussian overlay
    <out_dir>/gsparam_<fi>.png    Gaussian Params overlay
    <out_dir>/gt_<fi>.png         Ground-truth overlay
    <out_dir>/metadata.json       clip / stream / frames / per-frame mIoU / npz path

Stage 2 (render_skeleton_frames.py, in the hamer repo) reads metadata.json and
adds skeleton_<fi>.png for the same frames; Stage 3 (compose_grid.py) assembles
the final grid figure. This script reuses the inference/overlay helpers from
eval_segmentation_gs_mask.py so the masks match the team's video eval exactly.

Usage:
    python make_grid_neoverse.py \
        --npz /work/courses/3dv/team32/training_data_modal/clip-001100.npz \
        --stream stream1201-1 --frames 39 42 45 48 \
        --out_dir outputs/grid_clip-001100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from diffsynth.models import ModelManager
from eval_segmentation_gs_mask import (
    load_hand_head,
    load_gs_mask_head,
    predict_naive_labels,
    predict_gs_params_labels,
    build_gt_label,
    overlay_label,
    compute_iou,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--stream", default=None,
                        help="Stream name, e.g. stream1201-1. Defaults to first found.")
    parser.add_argument("--frames", type=int, nargs="+", required=True,
                        help="Absolute frame indices to dump (the keyframe rows).")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reconstructor_path",
                        default="models/NeoVerse/reconstructor.ckpt")
    parser.add_argument("--hand_head_path",
                        default="models/NeoVerse/hand_seg_model_opt_best.ckpt",
                        help="Naive Gaussian: DPT/hand_pred_head weights.")
    parser.add_argument("--gs_mask_head_path",
                        default="models/NeoVerse/gs_mask_model_best.ckpt",
                        help="Gaussian Params: nested {gs_head, gs_head_dynamic} state dict.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("gs_mask eval needs a CUDA GPU - the rasterizer is CUDA-only.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load base reconstructor + both heads ----
    print("Loading reconstructor ...")
    model_manager = ModelManager()
    model_manager.load_model(args.reconstructor_path, device=device,
                             torch_dtype=torch.bfloat16)
    reconstructor = model_manager.fetch_model("reconstructor")
    if reconstructor is None:
        raise RuntimeError(
            f"ModelManager could not load a reconstructor from {args.reconstructor_path}.")
    load_hand_head(reconstructor, args.hand_head_path, device)
    load_gs_mask_head(reconstructor, args.gs_mask_head_path, device)

    # ---- load the chosen frames ----
    npz = np.load(args.npz)
    stream_keys = [k for k in npz.keys() if k.startswith("images_")]
    stream = args.stream if args.stream else stream_keys[0].replace("images_", "")
    images_np = npz[f"images_{stream}"]           # (T, H, W, 3) uint8
    frame_indices = list(args.frames)
    T = images_np.shape[0]
    for fi in frame_indices:
        if not (0 <= fi < T):
            raise ValueError(f"frame {fi} out of range (clip has {T} frames)")
    print(f"clip={Path(args.npz).stem} stream={stream} frames={frame_indices}")

    imgs_tensor = torch.stack([
        F.to_tensor(Image.fromarray(images_np[i])) for i in frame_indices
    ], dim=0)                                     # (S, 3, H, W)
    S = imgs_tensor.shape[0]

    # ---- predictions for both variants ----
    naive_labels = predict_naive_labels(reconstructor, imgs_tensor, device)
    gsparam_labels = predict_gs_params_labels(reconstructor, imgs_tensor, device)
    gt_labels = np.stack([build_gt_label(npz, stream, i) for i in frame_indices])

    # ---- dump per-frame PNGs + collect per-frame mIoU ----
    per_frame = {}
    for s, fi in enumerate(frame_indices):
        pil = Image.fromarray(images_np[fi])
        pil.save(out_dir / f"raw_{fi}.png")
        overlay_label(pil, naive_labels[s]).save(out_dir / f"naive_{fi}.png")
        overlay_label(pil, gsparam_labels[s]).save(out_dir / f"gsparam_{fi}.png")
        overlay_label(pil, gt_labels[s]).save(out_dir / f"gt_{fi}.png")

        naive_miou = float(np.nanmean(compute_iou(naive_labels[s], gt_labels[s])))
        gsparam_miou = float(np.nanmean(compute_iou(gsparam_labels[s], gt_labels[s])))
        per_frame[str(fi)] = {"naive_miou": naive_miou, "gsparam_miou": gsparam_miou}
        print(f"  frame {fi}: naive mIoU={naive_miou:.3f}  gsparam mIoU={gsparam_miou:.3f}")

    meta = {
        "npz": str(Path(args.npz).resolve()),
        "clip": Path(args.npz).stem,
        "stream": stream,
        "frames": frame_indices,
        "height": int(images_np.shape[1]),
        "width": int(images_np.shape[2]),
        "per_frame": per_frame,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_dir/'metadata.json'} and {4*S} frame PNGs to {out_dir}")


if __name__ == "__main__":
    main()
