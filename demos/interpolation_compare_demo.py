"""Side-by-side *interpolation* demo for the NeoVerse reconstructor.

Where ``reconstruction_compare_demo.py`` re-renders the input frames statically
(``use_motion=False``) — and therefore cannot see the velocity field at all —
this tool evaluates the thing the velocity regularization actually changes:
**held-out in-between frames**.

For a window of ``2K-1`` consecutive frames it keeps the even frames as
keyframes (the model sees these) and *hides* the odd frames. Each model is then
asked to reconstruct the hidden frames by velocity-transitioning its keyframe
Gaussians to the mid-interval timestamp (``use_motion=True``). We lay the result
next to the real hidden frame:

    ground truth (hidden frame) | original | velocity-regularized

Each rendered cell is annotated with full-frame PSNR and **background PSNR**
(the region the velocity loss targets — lower background drift should show up
here first). Mirrors the data/split logic of ``benchmark.py``'s interpolation
mode, trimmed to a visual, single-window tool.

Examples
--------
    # First test clip, first window, both checkpoints:
    python interpolation_compare_demo.py \
        --velreg-path models/NeoVerse/velocity_regularization_best.ckpt

    # A specific clip / stream / window, K=3 keyframes (=> 5-frame window):
    python interpolation_compare_demo.py --clip clip-001160 \
        --stream stream1201-1 --window-index 2 --num-keyframes 3
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.SimpleHandObjectSegmentationDataset import STREAMS

# Reuse the demo's model-loading + split + metric helpers so the two tools stay
# in lockstep (same checkpoint handling, same test split, same PSNR/SSIM).
from demos.reconstruction_compare_demo import (
    get_test_clips,
    load_reconstructor,
    psnr,
    _ssim_fn,
    _to_img,
)
# The actual keyframe-only forward + camera-interpolated render lives in benchmark.py
# so the demo and the benchmark stay byte-for-byte identical in how they evaluate.
from evals.bechmark_interpolation import render_interpolated_targets


# ===================================================================== #
#                          TEST-SPLIT / DATA                            #
# ===================================================================== #


def load_interp_window(
    data_root: str,
    clip: str,
    stream: str,
    num_keyframes: int,
    frame_stride: int,
    window_index: int,
):
    """Load one interpolation window and return everything both the model and the
    figure need.

    Returns
    -------
    kf_images  : [K, 3, H, W]      keyframes shown to the model (even positions)
    tgt_images : [K-1, 3, H, W]    the hidden in-between frames (GT, odd positions)
    tgt_masks  : [K-1, 4, H, W]    GT seg masks for the hidden frames
    kf_masks   : [K, 4, H, W]      GT seg masks for the keyframes (right, left, object, bg)
    kf_idxs    : list[int]         original frame indices of the keyframes
    tgt_idxs   : list[int]         original frame indices of the hidden frames
    """
    block_len = 2 * num_keyframes - 1
    npz_path = Path(data_root) / f"{clip}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found.")
    npz = np.load(str(npz_path), mmap_mode="r")
    if f"images_{stream}" not in npz.files:
        avail = sorted({k.split("images_")[1] for k in npz.files if k.startswith("images_")})
        raise ValueError(f"Stream '{stream}' not in {clip}. Available: {avail}")

    n_frames = npz[f"images_{stream}"].shape[0]
    sampled = list(range(0, n_frames, frame_stride))
    blocks = [
        sampled[i : i + block_len]
        for i in range(0, len(sampled) - block_len + 1, block_len)
    ]
    if not blocks:
        raise ValueError(
            f"{clip}/{stream}: clip too short for a {block_len}-frame window "
            f"(num_keyframes={num_keyframes}, frame_stride={frame_stride})."
        )
    if window_index >= len(blocks):
        raise ValueError(
            f"window-index {window_index} out of range; {clip}/{stream} has {len(blocks)} windows."
        )
    block = blocks[window_index]

    def load(f):
        img = torch.tensor(npz[f"images_{stream}"][f], dtype=torch.float32).permute(2, 0, 1) / 255.0
        right = torch.tensor(npz[f"masks_{stream}_hand_RIGHT"][f] > 0)
        left = torch.tensor(npz[f"masks_{stream}_hand_LEFT"][f] > 0)
        obj = torch.tensor(npz[f"masks_{stream}_object"][f] > 0)
        bg = ~(right | left | obj)
        mask = torch.stack([right, left, obj, bg], dim=0).float()
        return img, mask

    kf_pos = list(range(0, block_len, 2))   # K keyframes (even positions)
    tg_pos = list(range(1, block_len, 2))   # K-1 targets   (odd positions)

    kf_pairs = [load(block[p]) for p in kf_pos]
    kf_images = torch.stack([img for img, _ in kf_pairs])            # [K, 3, H, W]
    kf_masks = torch.stack([mask for _, mask in kf_pairs])           # [K, 4, H, W]
    tgt_pairs = [load(block[p]) for p in tg_pos]
    tgt_images = torch.stack([img for img, _ in tgt_pairs])          # [K-1, 3, H, W]
    tgt_masks = torch.stack([mask for _, mask in tgt_pairs])         # [K-1, 4, H, W]

    return (
        kf_images,
        tgt_images,
        tgt_masks,
        kf_masks,
        [block[p] for p in kf_pos],                   # kf_idxs
        [block[p] for p in tg_pos],                   # tgt_idxs
    )


# ===================================================================== #
#                         INFERENCE / METRICS                           #
# ===================================================================== #


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """pred/gt [3,H,W]; mask [1,H,W] in {0,1}. PSNR over masked pixels (NaN if empty)."""
    m = mask.expand_as(pred)
    denom = m.sum().item()
    if denom <= 0:
        return float("nan")
    mse = ((pred - gt) ** 2 * m).sum().item() / denom
    return 99.0 if mse == 0 else -10.0 * math.log10(mse)


def mean_in_region(vel_mag: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean velocity magnitude over a region. vel_mag [h,w], mask [h,w] bool (NaN if empty)."""
    if mask.sum() <= 0:
        return float("nan")
    return vel_mag[mask].mean().item()


def velocity_heatmap(vel_mag: torch.Tensor, vmax: float, size) -> torch.Tensor:
    """vel_mag [h,w] -> [3,H,W] turbo-colormapped RGB in [0,1], resized to ``size`` (H,W)."""
    v = vel_mag.detach().cpu().numpy()
    vmax = vmax if vmax and vmax > 0 else 1.0
    normed = np.clip(v / vmax, 0.0, 1.0)
    heat = plt.cm.turbo(normed)[..., :3]                          # (h, w, 3)
    heat = torch.from_numpy(heat).permute(2, 0, 1).float()        # [3, h, w]
    if heat.shape[-2:] != tuple(size):
        heat = F.interpolate(heat.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
    return heat


def blend_overlay(rgb: torch.Tensor, heat: torch.Tensor, alpha: float) -> torch.Tensor:
    """rgb/heat [3,H,W] in [0,1]; returns alpha-blended [3,H,W]."""
    return ((1.0 - alpha) * rgb + alpha * heat).clamp(0, 1)


# ===================================================================== #
#                          VISUALIZATION                                #
# ===================================================================== #


def build_interp_figure(gt_rgb, model_renders, tgt_idxs, title, out_path=None, return_fig=False):
    """gt_rgb [V,3,H,W] (hidden frames); model_renders: name -> {"rgb","psnr","psnr_bg","ssim"}.

    Rows = hidden in-between frames, columns = GT + one per model. Each model cell
    is annotated with full PSNR and background PSNR.

    Saves to ``out_path`` if given. If ``return_fig`` is True the Figure is returned
    (caller must close it) — used to log the figure to TensorBoard; otherwise it is
    closed here.
    """
    names = list(model_renders.keys())
    V = gt_rgb.shape[0]
    n_cols = 1 + len(names)
    H, W = gt_rgb.shape[2], gt_rgb.shape[3]
    cell = 2.6
    fig, axes = plt.subplots(
        V, n_cols, figsize=(cell * n_cols, cell * V * (H / W) + 0.7), squeeze=False
    )

    col_titles = ["GT (hidden frame)", *names]
    for r in range(V):
        axes[r][0].imshow(_to_img(gt_rgb[r]))
        axes[r][0].set_ylabel(f"frame {tgt_idxs[r]}", fontsize=9)
        for c, name in enumerate(names, start=1):
            mr = model_renders[name]
            axes[r][c].imshow(_to_img(mr["rgb"][r]))
            label = f"PSNR {mr['psnr'][r]:.2f}   bg {mr['psnr_bg'][r]:.2f}"
            if mr["ssim"] is not None:
                label += f"\nSSIM {mr['ssim'][r]:.3f}"
            axes[r][c].set_xlabel(label, fontsize=8)
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=11)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130)
    if return_fig:
        return fig
    plt.close(fig)


# ===================================================================== #
#                                 CLI                                   #
# ===================================================================== #


def parse_args():
    p = argparse.ArgumentParser(description="Side-by-side interpolation comparison demo.")
    p.add_argument("--data-root", type=str,
                   default="/work/courses/3dv/team32/training_data_modal",
                   help="Same data_root as training/benchmark so the test split matches.")
    p.add_argument("--val-fraction", type=float, default=0.1)

    p.add_argument("--clip", type=str, default=None,
                   help="Clip stem, e.g. clip-001160. Default: first test-split clip.")
    p.add_argument("--stream", type=str, default=None,
                   help=f"Stream name. Default: first of {STREAMS}.")
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--num-keyframes", type=int, default=3,
                   help="K: window is 2K-1 frames (K keyframes + K-1 hidden in-between targets).")
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--img-shape", type=int, nargs=2, default=(280, 280),
                   help="(H, W) — rasterizer render resolution.")

    p.add_argument("--original-path", type=str,
                   default="models/NeoVerse/reconstructor.ckpt")
    p.add_argument("--velreg-path", type=str,
                   default="models/NeoVerse/velocity_regularization_epoch1.ckpt")
    p.add_argument("--original", action=argparse.BooleanOptionalAction, default=True,
                   help="Include the original reconstructor.")
    p.add_argument("--velreg", action=argparse.BooleanOptionalAction, default=True,
                   help="Include the velocity-regularized reconstructor.")

    p.add_argument("--velocity-overlay", action=argparse.BooleanOptionalAction, default=False,
                   help="Overlay a heatmap of the predicted forward-velocity magnitude on each "
                        "rendered cell (turbo colormap; blue=still, red=fast). Use this to check "
                        "whether the model assigns near-zero velocity to the background.")
    p.add_argument("--velocity-alpha", type=float, default=0.5,
                   help="Blend weight of the velocity heatmap over the render (0=off, 1=heatmap only).")
    p.add_argument("--velocity-max", type=float, default=None,
                   help="Velocity magnitude mapped to the top of the colormap. Default: shared "
                        "95th percentile across the shown models (so columns are comparable).")
    p.add_argument("--velocity-direction", type=str, default="fwd", choices=["fwd", "bwd"],
                   help="Which velocity field to overlay: 'fwd' pushes the left keyframe forward, "
                        "'bwd' pushes the right keyframe backward. Compare the two to see whether "
                        "forward/backward disagree (the source of 'two hands' ghosting).")
    p.add_argument("--single-direction", action=argparse.BooleanOptionalAction, default=False,
                   help="Render each midpoint from the nearest keyframe only (rasterizer "
                        "bidirection=False) instead of blending forward+backward Gaussians. If the "
                        "doubling collapses to one instance, the ghosting is forward/backward "
                        "velocity disagreement.")

    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="outputs/interp_demo")
    return p.parse_args()


def main():
    args = parse_args()

    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            f"--device is '{args.device}' but a working CUDA GPU is required: the bf16 model and the "
            "Gaussian-splat rasterizer are CUDA-only. Run on a GPU node (check `nvidia-smi`)."
        )

    # --- resolve clip / stream from the test split ---
    test_clips = get_test_clips(args.data_root, args.val_fraction)
    clip = args.clip or test_clips[0]
    if args.clip is not None and args.clip not in test_clips:
        print(f"WARNING: {args.clip} is not in the test split ({len(test_clips)} clips). "
              "Using it anyway.", flush=True)
    stream = args.stream or STREAMS[0]
    print(f"Test clips: {len(test_clips)} | using clip={clip} stream={stream} "
          f"window={args.window_index} num_keyframes={args.num_keyframes}", flush=True)

    # --- which models to run ---
    runs = []
    if args.original:
        runs.append(("original", args.original_path))
    if args.velreg:
        if Path(args.velreg_path).exists():
            runs.append(("velocity_reg", args.velreg_path))
        else:
            print(f"WARNING: velreg checkpoint {args.velreg_path} not found; skipping.", flush=True)
    if not runs:
        raise RuntimeError("No models selected (use --original / --velreg).")

    # --- load the window ---
    kf_images, tgt_images, tgt_masks, _kf_masks, kf_idxs, tgt_idxs = load_interp_window(
        args.data_root, clip, stream, args.num_keyframes, args.frame_stride, args.window_index
    )
    H, W = args.img_shape
    print(f"Keyframes (seen):  frames {kf_idxs}\n"
          f"Targets   (hidden): frames {tgt_idxs}", flush=True)

    ssim_metric = _ssim_fn(args.device)

    # --- run each model (pass 1: render, metrics, collect raw velocity) ---
    model_renders: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    model_vel: dict[str, torch.Tensor | None] = {}   # name -> vel_mag [V, h, w] (render res) or None
    for name, path in runs:
        print(f"[{name}]", flush=True)
        reconstructor = load_reconstructor(path, args.device)
        pred_rgb, has_velocity, vel_fwd, vel_bwd = render_interpolated_targets(
            reconstructor, kf_images.unsqueeze(0), args.device, W, H,
            return_velocity=True, bidirection=not args.single_direction,
        )  # pred_rgb [V,3,h,w]; vel_fwd/vel_bwd [V,H_in,W_in] on device (or None)
        pred_rgb = pred_rgb.cpu()
        if not has_velocity:
            print("  WARNING: no 'velocity_fwd' — motion did not engage; in-between frames "
                  "are rendered statically.", flush=True)

        # Match GT (and masks) to the render resolution for fair metrics.
        gt = tgt_images
        masks = tgt_masks
        if pred_rgb.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False)
            masks = F.interpolate(masks, size=pred_rgb.shape[-2:], mode="nearest")

        # Velocity magnitudes resampled to the render resolution (for overlay + region stats).
        def _to_render(v):
            if v is None:
                return None
            return F.interpolate(
                v.unsqueeze(1), size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0].cpu()                                   # [V, h, w]
        vel_fwd_r, vel_bwd_r = _to_render(vel_fwd), _to_render(vel_bwd)
        # The direction the user asked to visualize drives the overlay.
        model_vel[name] = vel_fwd_r if args.velocity_direction == "fwd" else vel_bwd_r

        V = pred_rgb.shape[0]
        per_psnr, per_psnr_bg = [], []
        per_ssim = [] if ssim_metric is not None else None
        vel_stats = {"fwd": ([], []), "bwd": ([], [])}   # dir -> (per_bg, per_fg)
        for i in range(V):
            per_psnr.append(psnr(pred_rgb[i], gt[i]))
            bg_pix = masks[i, 3] > 0.5
            per_psnr_bg.append(masked_psnr(pred_rgb[i], gt[i], bg_pix[None].float()))
            for dname, vr in (("fwd", vel_fwd_r), ("bwd", vel_bwd_r)):
                if vr is not None:
                    vel_stats[dname][0].append(mean_in_region(vr[i], bg_pix))
                    vel_stats[dname][1].append(mean_in_region(vr[i], ~bg_pix))
            if ssim_metric is not None:
                ssim_metric.reset()
                ssim_metric.update(pred_rgb[i:i+1].to(args.device), gt[i:i+1].to(args.device))
                per_ssim.append(ssim_metric.compute().item())

        model_renders[name] = {
            "rgb": pred_rgb,
            "psnr": per_psnr,
            "psnr_bg": per_psnr_bg,
            "ssim": per_ssim,
        }
        bg_vals = [v for v in per_psnr_bg if v == v]
        summary[name] = {
            "mean_psnr": float(np.mean(per_psnr)),
            "mean_psnr_bg": float(np.mean(bg_vals)) if bg_vals else float("nan"),
            "mean_ssim": float(np.mean(per_ssim)) if per_ssim is not None else None,
            "per_frame_psnr": per_psnr,
            "per_frame_psnr_bg": per_psnr_bg,
        }
        for dname, (pbg, pfg) in vel_stats.items():
            summary[name][f"mean_vel_{dname}_bg"] = float(np.nanmean(pbg)) if pbg else None
            summary[name][f"mean_vel_{dname}_fg"] = float(np.nanmean(pfg)) if pfg else None
        del reconstructor
        torch.cuda.empty_cache()

    # --- pass 2: bake the velocity overlay onto the display renders (shared scale) ---
    out_dir = Path(args.output_dir) / f"{clip}_{stream}_w{args.window_index}"
    title = f"Interpolated in-between frames — {clip} / {stream} / window {args.window_index}"
    if args.velocity_overlay:
        all_vel = [v for v in model_vel.values() if v is not None]
        if all_vel:
            vmax = args.velocity_max
            if vmax is None:
                vmax = float(np.percentile(torch.cat([v.flatten() for v in all_vel]).numpy(), 95))
            for name in model_renders:
                v = model_vel.get(name)
                if v is None:
                    continue
                rgb = model_renders[name]["rgb"]
                overlaid = torch.stack([
                    blend_overlay(rgb[i], velocity_heatmap(v[i], vmax, rgb.shape[-2:]),
                                  args.velocity_alpha)
                    for i in range(rgb.shape[0])
                ])
                model_renders[name]["rgb"] = overlaid
            title += f"  |  {args.velocity_direction}-velocity overlay (turbo, vmax={vmax:.3g})"
        else:
            print("  velocity overlay requested but no model produced velocity; skipping.", flush=True)
    if args.single_direction:
        title += "  |  single-direction render"

    # --- figure (GT-resolution thumbnails for display) ---
    fig_path = out_dir / "interpolation_comparison.png"
    build_interp_figure(tgt_images, model_renders, tgt_idxs, title=title, out_path=fig_path)

    # --- summary ---
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"clip": clip, "stream": stream, "window_index": args.window_index,
                   "keyframe_idxs": kf_idxs, "target_idxs": tgt_idxs, "models": summary}, f, indent=2)

    print("\n=== Mean interpolation quality on hidden frames (higher = better) ===")
    for name, s in summary.items():
        ssim_str = f"  SSIM {s['mean_ssim']:.4f}" if s["mean_ssim"] is not None else ""
        print(f"  {name:>14}:  PSNR {s['mean_psnr']:.3f}   bg-PSNR {s['mean_psnr_bg']:.3f}{ssim_str}")
    if len(summary) > 1:
        best = max(summary, key=lambda n: summary[n]["mean_psnr_bg"])
        print(f"  -> best by background PSNR: {best}")

    # Velocity diagnostic: background should be ~0; a large bg value means the velocity
    # head is moving parts of the scene that don't actually move. We print both forward
    # and backward fields — if fg magnitudes differ a lot between fwd and bwd, the two
    # estimates disagree, which is what produces the "two hands" ghosting at the midpoint.
    if any(s.get("mean_vel_fwd_bg") is not None for s in summary.values()):
        print("\n=== Predicted velocity magnitude (background should be ~0) ===")
        for name, s in summary.items():
            if s.get("mean_vel_fwd_bg") is None:
                continue
            print(f"  {name}:")
            for d in ("fwd", "bwd"):
                vb, vf = s.get(f"mean_vel_{d}_bg"), s.get(f"mean_vel_{d}_fg")
                if vb is None:
                    continue
                ratio = f"  fg/bg {vf / vb:.1f}x" if vb and vb > 0 else ""
                print(f"      {d}:  bg {vb:.4f}   fg {vf:.4f}{ratio}")

    print(f"\nFigure: {fig_path}\nSummary: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
