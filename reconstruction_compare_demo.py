"""Side-by-side reconstruction demo for the NeoVerse reconstructor.

Renders a single window of frames from a *test-split* clip with one or two
reconstructors and lays the results next to the ground truth so you can see, at
a glance, which model gives the best reconstruction:

    ground truth | original reconstructor | velocity-regularized reconstructor

Each rendered cell is annotated with its per-frame PSNR/SSIM, and a summary of
the mean metrics per model is printed and saved.

This mirrors the data path (test split, window grouping) and the inference path
(``reconstructor(views, ...)`` -> Gaussian-splat re-render at the input cameras)
used by ``benchmark.py``, just trimmed down to a visual, single-clip tool.

Examples
--------
    # Compare both checkpoints on the first test clip, first window:
    python reconstruction_compare_demo.py \
        --velreg-path models/NeoVerse/velocity_regularization_best.ckpt

    # A specific clip / stream / window:
    python reconstruction_compare_demo.py --clip clip-0042 \
        --stream stream1201-1 --window-index 2

    # Only the original model:
    python reconstruction_compare_demo.py --no-velreg
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

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.data.SimpleHandObjectSegmentationDataset import STREAMS
from diffsynth.models.model_manager import ModelManager
from diffsynth.utils.auxiliary import homo_matrix_inverse


# ===================================================================== #
#                          TEST-SPLIT / DATA                            #
# ===================================================================== #


def get_test_clips(data_root: str, val_fraction: float) -> list[str]:
    """The held-out clips — the tail of the sorted glob, exactly as benchmark.py
    derives the val split (which is the split velocity_regularization.py leaves
    untrained when it consumes the rest of the clips)."""
    all_clips = sorted(p.stem for p in Path(data_root).glob("clip-*.npz"))
    if not all_clips:
        raise FileNotFoundError(f"No clip-*.npz under {data_root}")
    n_val = max(1, int(len(all_clips) * val_fraction))
    return all_clips[-n_val:]


def load_window(
    data_root: str,
    clip: str,
    stream: str,
    window_size: int,
    frame_stride: int,
    window_index: int,
):
    """Return (images [S,3,H,W] in [0,1], masks [S,4,H,W], frame_idxs).

    Frame selection matches ClipWindowDataset: stride-sample the clip, then take
    the ``window_index``-th contiguous block of ``window_size`` frames.
    """
    npz_path = Path(data_root) / f"{clip}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found.")
    npz = np.load(str(npz_path), mmap_mode="r")

    if f"images_{stream}" not in npz.files:
        avail = sorted({k.split("images_")[1] for k in npz.files if k.startswith("images_")})
        raise ValueError(f"Stream '{stream}' not in {clip}. Available: {avail}")

    n_frames = npz[f"images_{stream}"].shape[0]
    sampled = list(range(0, n_frames, frame_stride))
    windows = [sampled[i : i + window_size] for i in range(0, len(sampled), window_size)]
    windows = [w for w in windows if len(w) >= 2]
    if not windows:
        raise ValueError(f"{clip}/{stream} has no window of >=2 frames.")
    if window_index >= len(windows):
        raise ValueError(
            f"window-index {window_index} out of range; {clip}/{stream} has {len(windows)} windows."
        )
    frame_idxs = windows[window_index]

    images, masks = [], []
    for f in frame_idxs:
        img = torch.tensor(npz[f"images_{stream}"][f], dtype=torch.float32).permute(2, 0, 1) / 255.0
        right = torch.tensor(npz[f"masks_{stream}_hand_RIGHT"][f] > 0)
        left = torch.tensor(npz[f"masks_{stream}_hand_LEFT"][f] > 0)
        obj = torch.tensor(npz[f"masks_{stream}_object"][f] > 0)
        bg = ~(right | left | obj)
        masks.append(torch.stack([right, left, obj, bg], dim=0).float())
        images.append(img)

    return torch.stack(images), torch.stack(masks), frame_idxs


# ===================================================================== #
#                          MODEL LOADING                                #
# ===================================================================== #


def _is_wrapped_training_ckpt(path: str) -> bool:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return isinstance(raw, dict) and "model_state_dict" in raw


def load_reconstructor(path: str, device: str):
    """Load either the original reconstructor (auto-detected flat state_dict) or a
    fine-tuned wrapped checkpoint (built as the extended 16-channel WorldMirror).

    Same two-branch logic as benchmark.load_reconstructor, minus the hand-head —
    the RGB re-render doesn't use it.
    """
    print(f"  loading {path}", flush=True)
    if _is_wrapped_training_ckpt(path):
        reconstructor = WorldMirror(enable_norm=False)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        missing, unexpected = reconstructor.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"    (fine-tuned) {len(sd)} tensors, {len(missing)} missing, "
                  f"{len(unexpected)} unexpected", flush=True)
        reconstructor = reconstructor.to(device=device, dtype=torch.bfloat16)
    else:
        mm = ModelManager(device=device, torch_dtype=torch.bfloat16)
        mm.load_model(path, device=device, torch_dtype=torch.bfloat16)
        reconstructor = mm.fetch_model("reconstructor")
        if reconstructor is None:
            raise RuntimeError(f"ModelManager could not load a reconstructor from {path}.")
    return reconstructor.to(device).eval()


# ===================================================================== #
#                         INFERENCE / METRICS                           #
# ===================================================================== #


@torch.no_grad()
def render_window(reconstructor, images_seq, device, res_w, res_h):
    """Run the reconstructor on [1,S,3,H,W] and re-render RGB at the input cameras.

    Returns pred_rgb [S,3,H,W] in [0,1] on CPU.
    """
    B, S = images_seq.shape[:2]
    views = {
        "img": images_seq,
        "is_target": torch.zeros((B, S), dtype=torch.bool, device=device),
        "is_static": torch.zeros((B, S), dtype=torch.bool, device=device),
        "timestamp": torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0).expand(B, -1),
    }

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = reconstructor(views, is_inference=True, use_motion=False)

    gaussians = predictions["splats"]
    input_w2c = homo_matrix_inverse(predictions["rendered_extrinsics"][0])

    # rasterizer.forward returns colors first; later code added depth/alpha/mask
    # channels, so just take element 0 regardless of arity.
    out = reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[input_w2c],
        render_Ks=[predictions["rendered_intrinsics"][0]],
        render_timestamps=[predictions["rendered_timestamps"][0]],
        sh_degree=0,
        width=res_w,
        height=res_h,
        render_classes=[0, 1, 2, 3],
    )
    target_rgb = out[0] if isinstance(out, (tuple, list)) else out
    # target_rgb: [1, S, H, W, 3] in [0,1]
    return target_rgb[0].permute(0, 3, 1, 2).clamp(0, 1).float().cpu()


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    return 99.0 if mse == 0 else -10.0 * math.log10(mse)


def _ssim_fn(device):
    """torchmetrics SSIM if available, else None (metric is skipped)."""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception:
        return None


# ===================================================================== #
#                          VISUALIZATION                                #
# ===================================================================== #


def _to_img(t: torch.Tensor) -> np.ndarray:
    return t.permute(1, 2, 0).numpy()


def build_comparison_figure(gt_rgb, model_renders, frame_idxs, title, out_path):
    """gt_rgb [S,3,H,W]; model_renders: dict name -> {"rgb":[S,3,H,W], "psnr":[S], "ssim":[S]|None}.

    Rows = frames, columns = ground truth + one per model. Each model cell is
    annotated with its per-frame PSNR (and SSIM if computed).
    """
    names = list(model_renders.keys())
    S = gt_rgb.shape[0]
    n_cols = 1 + len(names)
    H, W = gt_rgb.shape[2], gt_rgb.shape[3]
    cell = 2.6
    fig, axes = plt.subplots(
        S, n_cols, figsize=(cell * n_cols, cell * S * (H / W) + 0.6), squeeze=False
    )

    col_titles = ["Ground truth", *names]
    for r in range(S):
        axes[r][0].imshow(_to_img(gt_rgb[r]))
        axes[r][0].set_ylabel(f"frame {frame_idxs[r]}", fontsize=9)
        for c, name in enumerate(names, start=1):
            mr = model_renders[name]
            axes[r][c].imshow(_to_img(mr["rgb"][r]))
            label = f"PSNR {mr['psnr'][r]:.2f}"
            if mr["ssim"] is not None:
                label += f"  SSIM {mr['ssim'][r]:.3f}"
            axes[r][c].set_xlabel(label, fontsize=8)
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=11)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===================================================================== #
#                                 CLI                                   #
# ===================================================================== #


def parse_args():
    p = argparse.ArgumentParser(description="Side-by-side reconstruction comparison demo.")
    p.add_argument("--data-root", type=str,
                   default="/work/courses/3dv/team32/training_data_modal",
                   help="Same data_root as training/benchmark so the test split matches.")
    p.add_argument("--val-fraction", type=float, default=0.1)

    p.add_argument("--clip", type=str, default=None,
                   help="Clip stem, e.g. clip-0042. Default: first test-split clip.")
    p.add_argument("--stream", type=str, default=None,
                   help=f"Stream name. Default: first of {STREAMS}.")
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--window-size", type=int, default=6)
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--max-display-frames", type=int, default=6,
                   help="Cap rows in the figure (subsamples the window evenly).")
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

    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="outputs/recon_demo")
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
          f"window={args.window_index}", flush=True)

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
    images, gt_masks, frame_idxs = load_window(
        args.data_root, clip, stream, args.window_size, args.frame_stride, args.window_index
    )
    images_seq = images.unsqueeze(0).to(args.device)  # [1,S,3,H,W]
    H, W = args.img_shape

    # subsample frames for display
    S = images.shape[0]
    if S > args.max_display_frames:
        sel = np.linspace(0, S - 1, args.max_display_frames).round().astype(int).tolist()
    else:
        sel = list(range(S))

    ssim_metric = _ssim_fn(args.device)

    # --- run each model ---
    model_renders: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for name, path in runs:
        print(f"[{name}]", flush=True)
        reconstructor = load_reconstructor(path, args.device)
        pred_rgb = render_window(reconstructor, images_seq, args.device, W, H)

        gt_rgb = images
        if pred_rgb.shape[-2:] != gt_rgb.shape[-2:]:
            gt_cmp = torch.nn.functional.interpolate(
                gt_rgb, size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            gt_cmp = gt_rgb

        per_psnr, per_ssim = [], ([] if ssim_metric is not None else None)
        for i in range(S):
            per_psnr.append(psnr(pred_rgb[i], gt_cmp[i]))
            if ssim_metric is not None:
                ssim_metric.reset()
                ssim_metric.update(pred_rgb[i:i+1].to(args.device), gt_cmp[i:i+1].to(args.device))
                per_ssim.append(ssim_metric.compute().item())

        model_renders[name] = {
            "rgb": pred_rgb[sel],
            "psnr": [per_psnr[i] for i in sel],
            "ssim": [per_ssim[i] for i in sel] if per_ssim is not None else None,
        }
        summary[name] = {
            "mean_psnr": float(np.mean(per_psnr)),
            "mean_ssim": float(np.mean(per_ssim)) if per_ssim is not None else None,
            "per_frame_psnr": per_psnr,
        }
        del reconstructor
        torch.cuda.empty_cache()

    # --- figure ---
    out_dir = Path(args.output_dir) / f"{clip}_{stream}_w{args.window_index}"
    fig_path = out_dir / "comparison.png"
    build_comparison_figure(
        images[sel], model_renders, [frame_idxs[i] for i in sel],
        title=f"{clip} / {stream} / window {args.window_index}",
        out_path=fig_path,
    )

    # --- summary ---
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"clip": clip, "stream": stream, "window_index": args.window_index,
                   "frame_idxs": frame_idxs, "models": summary}, f, indent=2)

    print("\n=== Mean reconstruction quality (higher PSNR/SSIM = better) ===")
    for name, s in summary.items():
        ssim_str = f"  SSIM {s['mean_ssim']:.4f}" if s["mean_ssim"] is not None else ""
        print(f"  {name:>14}:  PSNR {s['mean_psnr']:.3f}{ssim_str}")
    if len(summary) > 1:
        best = max(summary, key=lambda n: summary[n]["mean_psnr"])
        print(f"  -> best by PSNR: {best}")
    print(f"\nFigure: {fig_path}\nSummary: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
