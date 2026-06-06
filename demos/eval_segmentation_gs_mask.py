"""
Evaluate the gs_mask segmentation checkpoint on a val-set NPZ clip.

Unlike ``eval_segmentation.py`` (which scores the 2D ``hand_pred_head`` via
``predictions["seg_labels"]``), the gs_mask model carries its segmentation in the
*per-Gaussian* mask logits. So the prediction here is the **rasterized mask
channels** at the input cameras, argmaxed over the 4 classes — exactly the
``gs_mask`` path in ``evals/benchmark_rec.py``.

Loading is also different: the gs_mask checkpoint is NOT a hand head. It is a
nested ``{"gs_head": <sd>, "gs_head_dynamic": <sd>}`` state dict that is overlaid
onto a base reconstructor (see ``benchmark_rec.load_gs_mask_head``). The base
reconstructor's ``gs_head`` already owns 4 mask channels (GaussianSplatRenderer
bakes ``num_mask_classes=4`` in unconditionally); the original ``reconstructor.ckpt``
just leaves them at init, and this checkpoint fills them with trained weights.

Outputs:
  - Per-class IoU and mIoU printed to stdout
  - Side-by-side video: input | prediction (rasterized mask) | ground truth

Usage:
    python demos/eval_segmentation_gs_mask.py \
        --npz /work/courses/3dv/team32/training_data_modal/clip-001053.npz \
        --reconstructor_path models/NeoVerse/reconstructor.ckpt \
        --gs_mask_head_path models/NeoVerse/gs_mask_model_run20260510-175056_epoch006.ckpt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from NeoVerse.diffsynth.models import ModelManager
from NeoVerse.diffsynth import save_video
from NeoVerse.diffsynth.utils.auxiliary import homo_matrix_inverse

# Reuse the gs_mask loader (nested gs_head/gs_head_dynamic overlay) and the
# visualization / IoU helpers so this script stays in lockstep with the benchmark.
from evals.benchmark_rec import load_gs_mask_head
from demos.eval_segmentation import (
    CLASS_NAMES,
    build_gt_label,
    overlay_label,
    add_caption,
    add_legend,
    compute_iou,
)


@torch.no_grad()
def render_mask_labels(reconstructor, predictions, width, height):
    """Rasterize the per-Gaussian mask logits at the input cameras → pred labels [S, H, W].

    ``width``/``height`` must be the input frames' resolution: ``rendered_intrinsics``
    are defined at that size and the rasterizer projects them onto the canvas without
    rescaling, so a mismatched size would crop rather than resize.
    """
    gaussians = predictions["splats"]
    input_c2w = predictions["rendered_extrinsics"][0]
    input_intrs = predictions["rendered_intrinsics"][0]
    input_ts = predictions["rendered_timestamps"][0]
    input_w2c = homo_matrix_inverse(input_c2w)

    _, _, _, masks = reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[input_w2c],
        render_Ks=[input_intrs],
        render_timestamps=[input_ts],
        sh_degree=0,
        width=width,
        height=height,
        render_classes=[0, 1, 2, 3],
    )
    if masks is None:
        raise RuntimeError(
            "Rasterizer returned no mask logits — the loaded gs_head has no trained mask "
            "channels. Check --gs_mask_head_path points to a gs_mask checkpoint."
        )
    # masks: [1, S, H, W, C] → argmax over classes → [S, H, W]
    return masks[0].float().argmax(dim=-1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", nargs="+",
                        default=["diffsynth/data/training_data/clip-001053.npz"])
    parser.add_argument("--reconstructor_path",
                        default="models/NeoVerse/reconstructor.ckpt",
                        help="Base reconstructor the gs_mask head is overlaid onto.")
    parser.add_argument("--gs_mask_head_path",
                        default="models/NeoVerse/gs_mask_model_run20260510-175056_epoch006.ckpt",
                        help="gs_mask checkpoint: nested {gs_head, gs_head_dynamic} state dict.")
    parser.add_argument("--output", default="outputs/eval_seg_gs_mask.mp4")
    parser.add_argument("--stream", default=None,
                        help="Which stream to use, e.g. 'stream1201-1'. Defaults to first found.")
    parser.add_argument("--stride", type=int, default=3,
                        help="Frame stride (matches training default of 3)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            "gs_mask eval needs a CUDA GPU — the Gaussian-splat rasterizer is CUDA-only."
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---- load base reconstructor, then overlay the gs_mask head ----
    print("Loading reconstructor ...")
    model_manager = ModelManager()
    model_manager.load_model(args.reconstructor_path, device=device,
                             torch_dtype=torch.bfloat16)
    reconstructor = model_manager.fetch_model("reconstructor")
    if reconstructor is None:
        raise RuntimeError(
            f"ModelManager could not load a reconstructor from {args.reconstructor_path}."
        )

    # Overlays gs_head / gs_head_dynamic (kept in fp32) and calls .to(device).eval().
    load_gs_mask_head(reconstructor, args.gs_mask_head_path, device)

    all_clip_ious = []

    for npz_path in args.npz:
        clip_name = Path(npz_path).stem
        out_path = Path(args.output).parent / f"eval_seg_gs_mask_{clip_name}.mp4"
        print(f"\n{'='*50}")
        print(f"Clip: {clip_name}")

        # ---- load NPZ ----
        npz = np.load(npz_path)
        stream_keys = [k for k in npz.keys() if k.startswith("images_")]
        stream = args.stream if args.stream else stream_keys[0].replace("images_", "")

        images_np = npz[f"images_{stream}"]        # (T, H, W, 3) uint8
        T = images_np.shape[0]
        frame_indices = list(range(0, T, args.stride))
        print(f"  stream={stream}  frames={T}  evaluating={len(frame_indices)}")

        # ---- inference ----
        imgs_tensor = torch.stack([
            F.to_tensor(Image.fromarray(images_np[i]))
            for i in frame_indices
        ], dim=0)                                  # (S, 3, H, W)

        S = imgs_tensor.shape[0]
        views = {
            "img":       imgs_tensor.unsqueeze(0).to(device),
            "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, S), dtype=torch.bool, device=device),
            "timestamp": torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0),
        }

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predictions = reconstructor(views, is_inference=True, use_motion=False)

        # ---- prediction = rasterized per-Gaussian mask channels ----
        H, W = images_np.shape[1], images_np.shape[2]
        pred_labels = render_mask_labels(reconstructor, predictions, width=W, height=H)

        # ---- GT labels ----
        gt_labels = np.stack([build_gt_label(npz, stream, i) for i in frame_indices])

        # ---- IoU ----
        clip_ious = [compute_iou(pred_labels[s], gt_labels[s]) for s in range(S)]
        mean_ious = np.nanmean(clip_ious, axis=0)
        miou = np.nanmean(mean_ious)
        all_clip_ious.append(mean_ious)

        print("  Per-class IoU:")
        for c, name in enumerate(CLASS_NAMES):
            print(f"    {name:12s}: {mean_ious[c]:.4f}")
        print(f"    {'mIoU':12s}: {miou:.4f}")

        # ---- render video ----
        out_frames = []
        for s, fi in enumerate(frame_indices):
            pil = Image.fromarray(images_np[fi])
            pred_overlay = overlay_label(pil, pred_labels[s])
            gt_overlay   = overlay_label(pil, gt_labels[s])

            frame_miou = np.nanmean(compute_iou(pred_labels[s], gt_labels[s]))
            col1 = add_legend(add_caption(pil,          f"{clip_name}  frame {fi}"))
            col2 = add_caption(pred_overlay, f"gs_mask  mIoU={frame_miou:.2f}")
            col3 = add_caption(gt_overlay,   "Ground Truth")

            combined = Image.new("RGB", (W * 3, H))
            combined.paste(col1, (0,   0))
            combined.paste(col2, (W,   0))
            combined.paste(col3, (W*2, 0))
            out_frames.append(combined)

        save_video(out_frames, str(out_path), fps=10)
        print(f"  Saved {out_path}")

        del predictions, views, imgs_tensor
        torch.cuda.empty_cache()

    # ---- overall summary ----
    overall = np.nanmean(all_clip_ious, axis=0)
    print(f"\n{'='*50}")
    print("OVERALL (mean across all clips):")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {overall[c]:.4f}")
    print(f"  {'mIoU':12s}: {np.nanmean(overall):.4f}")


if __name__ == "__main__":
    main()
