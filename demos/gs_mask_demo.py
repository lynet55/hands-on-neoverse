"""
Compare the two segmentation approaches on a val-set NPZ clip:

  * "Naïve Gaussian" = reconstructor + DPT head. The 2D segmentation head
    (hand_pred_head) predicts ``predictions["seg_labels"]`` directly from image
    tokens. Run in its canonical regime (whole clip, is_inference=True), the same
    way ``eval_segmentation.py`` scores it.

  * "Gaussian Params" = per-Gaussian mask logits. Each Gaussian carries 4 mask
    channels that are rasterized (alpha-composited) into a 2D mask. Run in its
    training regime (one frame per forward, is_static=True, timestamp=0,
    is_inference=False) — the regime ``training_gs_mask.GsMaskReconstructor``
    optimized and reported val mIoU in. Feeding a multi-frame dynamic sequence
    instead routes through the 4DGS motion path the mask channels never saw.

Both are overlaid against ground truth. The output video has four columns:
  input | Naïve Gaussian | Gaussian Params | Ground Truth
with per-frame mIoU printed on each prediction column, and per-class / mean IoU
for both variants printed to stdout.

Loading:
  * DPT head: a hand_pred_head state dict, loaded onto the reconstructor
    (matches eval_segmentation.py).
  * gs_mask head: a nested ``{"gs_head", "gs_head_dynamic"}`` under
    ``model_state_dict`` (training_gs_mask.save_checkpoint), overlaid onto the
    reconstructor's already-16-channel gs heads.

Usage:
    python eval_segmentation_gs_mask.py \
        --npz /work/courses/3dv/team32/training_data_modal/clip-000157.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from diffsynth.models import ModelManager
from diffsynth import save_video
from diffsynth.utils.auxiliary import homo_matrix_inverse


CLASS_NAMES = ["right_hand", "left_hand", "object", "background"]
CLASS_COLORS = [
    (255,  60,  60, 160),   # right_hand  — red
    ( 60, 120, 255, 160),   # left_hand   — blue
    ( 60, 220,  60, 160),   # object      — green
    (  0,   0,   0,   0),   # background  — transparent
]


# ---------- checkpoint loading ----------

def load_hand_head(reconstructor, path: str, device: str):
    """Load the DPT/hand segmentation head (predictions['seg_labels'])."""
    print(f"Loading DPT (hand) head from {path} ...")
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    if not any(k.startswith("hand_pred_head.") for k in sd.keys()):
        sd = {f"hand_pred_head.{k}": v for k, v in sd.items()}
    else:
        sd = {k: v for k, v in sd.items() if k.startswith("hand_pred_head.")}
    missing, _ = reconstructor.load_state_dict(sd, strict=False)
    if any("hand_pred_head" in k for k in missing):
        print("    WARNING: some hand_pred_head keys missing from checkpoint.")
    reconstructor.hand_pred_head.float()
    reconstructor.to(device).eval()
    return reconstructor


def load_gs_mask_head(reconstructor, path: str, device: str):
    """Overlay the trained gs_head / gs_head_dynamic from a gs_mask checkpoint.

    Nested ``model_state_dict = {"gs_head", "gs_head_dynamic"}`` from
    training_gs_mask.save_checkpoint. The base reconstructor already has
    16-channel gs heads (4 mask channels baked in), so these load with strict=True.
    """
    print(f"Loading gs_mask head from {path} ...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    params = ckpt.get("model_state_dict", ckpt)
    if "gs_head" not in params:
        raise RuntimeError(
            f"{path} has no 'gs_head' in its model_state_dict (keys: {list(params)[:8]}). "
            "Expected a gs_mask checkpoint from training_gs_mask.py."
        )
    gsr = reconstructor.gs_renderer
    gsr.gs_head.load_state_dict(params["gs_head"], strict=True)
    print("    gs_head loaded (strict).")
    if "gs_head_dynamic" in params and hasattr(gsr, "gs_head_dynamic"):
        gsr.gs_head_dynamic.load_state_dict(params["gs_head_dynamic"], strict=True)
        print("    gs_head_dynamic loaded (strict).")
    gsr.gs_head.float()
    if hasattr(gsr, "gs_head_dynamic"):
        gsr.gs_head_dynamic.float()
    reconstructor.to(device).eval()
    return reconstructor


# ---------- inference ----------

@torch.no_grad()
def predict_naive_labels(reconstructor, imgs_tensor, device):
    """Naïve Gaussian = DPT head. Whole clip, is_inference=True (eval_segmentation regime).

    Returns pred labels [S, H, W] from predictions['seg_labels'].
    """
    S = imgs_tensor.shape[0]
    views = {
        "img": imgs_tensor.unsqueeze(0).to(device, non_blocking=True),  # [1, S, 3, H, W]
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, S), dtype=torch.bool, device=device),
        "timestamp": torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0),
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = reconstructor(views, is_inference=True, use_motion=False)
    if "seg_labels" not in predictions:
        raise RuntimeError(
            "predictions has no 'seg_labels' — the DPT head did not run. Check that the "
            "reconstructor has enable_hand_pred and --hand_head_path is a hand_pred_head ckpt."
        )
    seg = predictions["seg_labels"][0]                 # [S, H, W, C]
    return seg.float().argmax(dim=-1).cpu().numpy()


@torch.no_grad()
def predict_gs_params_labels(reconstructor, imgs_tensor, device, batch_size=4,
                             bg_alpha_thresh=None, bg_bias=0.0, bg_class=3):
    """Gaussian Params = rasterized per-Gaussian mask logits.

    Mirrors GsMaskReconstructor.forward: each frame is its own 1-frame STATIC
    sequence (is_static=True, timestamp=0, is_inference=False), then the mask
    channels are rasterized at that frame's own camera/resolution.
    Returns pred labels [S, H, W].

    Background handling (experiment #1) — the mask channels are alpha-composited
    onto a zero background, so any pixel with low coverage has all-zero class
    logits and argmax ties to channel 0 (right_hand), spawning spurious hands in
    empty regions. Two optional, off-by-default corrections:
      * bg_bias: add a constant to the background channel logit before argmax,
        so all-zero / uncertain pixels resolve to background instead of class 0.
      * bg_alpha_thresh: hard-assign pixels whose composited alpha (coverage) is
        below this threshold to bg_class.
    """
    S, _, H, W = imgs_tensor.shape
    labels = []
    for start in range(0, S, batch_size):
        chunk = imgs_tensor[start:start + batch_size]            # [b, 3, H, W]
        b = chunk.shape[0]
        imgs = chunk.unsqueeze(1).to(device, non_blocking=True)  # [b, 1, 3, H, W]
        views = {
            "img": imgs,
            "is_target": torch.zeros((b, 1), dtype=torch.bool, device=device),
            "is_static": torch.ones((b, 1), dtype=torch.bool, device=device),
            "timestamp": torch.zeros((b, 1), dtype=torch.int64, device=device),
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = reconstructor(views, is_inference=False, use_motion=False)

        gaussians = predictions["splats"]
        c2w = predictions["rendered_extrinsics"]    # [b, 1, 4, 4]
        Ks = predictions["rendered_intrinsics"]     # [b, 1, 3, 3]
        ts = predictions["rendered_timestamps"]     # [b, 1]

        for j in range(b):
            w2c = homo_matrix_inverse(c2w[j])       # [1, 4, 4]
            _, _, alpha, masks = reconstructor.gs_renderer.rasterizer.forward(
                render_splats=[gaussians[j]],
                render_viewmats=[w2c],
                render_Ks=[Ks[j]],
                render_timestamps=[ts[j]],
                sh_degree=0,
                width=W,
                height=H,
            )
            if masks is None:
                raise RuntimeError(
                    "Rasterizer returned no mask logits — the loaded gs_head has no trained "
                    "mask channels. Check --gs_mask_head_path points to a gs_mask checkpoint."
                )
            # masks: [1, 1, H, W, C]
            labels.append(_label_from_rendered(masks, alpha, bg_alpha_thresh, bg_bias, bg_class))
    return np.stack(labels)


def _label_from_rendered(masks, alpha, bg_alpha_thresh, bg_bias, bg_class):
    """Shared argmax + background correction for a single rasterized frame.

    masks/alpha: rasterizer outputs [1, 1, H, W, C] / [1, 1, H, W, 1]. Returns [H, W] np.uint."""
    logits = masks[0, 0].float()                    # [H, W, C]
    if bg_bias:
        logits[..., bg_class] = logits[..., bg_class] + bg_bias
    label = logits.argmax(dim=-1)
    if bg_alpha_thresh is not None:
        cov = alpha[0, 0].float().squeeze(-1)       # [H, W] composited coverage
        label = torch.where(cov < bg_alpha_thresh,
                            torch.full_like(label, bg_class), label)
    return label.cpu().numpy()


@torch.no_grad()
def predict_gs_params_labels_multiframe(reconstructor, imgs_tensor, device, seq_len=8,
                                        bg_alpha_thresh=None, bg_bias=0.0, bg_class=3):
    """Experiment #2: give the reconstructor multi-frame context instead of 1 frame.

    Each chunk of ``seq_len`` consecutive frames is run as ONE sequence, so
    prepare_splats voxel-merges Gaussians across all frames into a single field;
    every frame's mask is then rasterized from that merged (multi-view) field at
    the frame's own predicted camera. Falls back to the same background handling
    as the per-frame path. Returns pred labels [S, H, W].
    """
    S, _, H, W = imgs_tensor.shape
    labels = [None] * S
    for start in range(0, S, seq_len):
        idx = list(range(start, min(start + seq_len, S)))
        n = len(idx)
        imgs = imgs_tensor[idx].unsqueeze(0).to(device, non_blocking=True)  # [1, n, 3, H, W]
        views = {
            "img": imgs,
            "is_target": torch.zeros((1, n), dtype=torch.bool, device=device),
            "is_static": torch.ones((1, n), dtype=torch.bool, device=device),
            "timestamp": torch.arange(n, dtype=torch.int64, device=device).unsqueeze(0),
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = reconstructor(views, is_inference=False, use_motion=False)

        merged = predictions["splats"][0]            # one merged field for the whole chunk
        c2w = predictions["rendered_extrinsics"][0]  # [n, 4, 4]
        Ks = predictions["rendered_intrinsics"][0]   # [n, 3, 3]
        ts = predictions["rendered_timestamps"][0]   # [n]

        for k, s in enumerate(idx):
            w2c = homo_matrix_inverse(c2w[k:k + 1])  # [1, 4, 4]
            _, _, alpha, masks = reconstructor.gs_renderer.rasterizer.forward(
                render_splats=[merged],
                render_viewmats=[w2c],
                render_Ks=[Ks[k:k + 1]],
                render_timestamps=[ts[k:k + 1]],
                sh_degree=0,
                width=W,
                height=H,
            )
            if masks is None:
                raise RuntimeError("Rasterizer returned no mask logits in multiframe path.")
            labels[s] = _label_from_rendered(masks, alpha, bg_alpha_thresh, bg_bias, bg_class)
    return np.stack(labels)


# ---------- visualization / metrics ----------

def build_gt_label(npz, stream: str, frame_idx: int) -> np.ndarray:
    h, w = npz[f"images_{stream}"].shape[1:3]
    label = np.full((h, w), 3, dtype=np.uint8)       # background
    for cls_id, key_suffix in [(2, "object"), (1, "hand_LEFT"), (0, "hand_RIGHT")]:
        key = f"masks_{stream}_{key_suffix}"
        if key in npz:
            label[npz[key][frame_idx] > 0] = cls_id
    return label


def overlay_label(pil_img: Image.Image, label_hw: np.ndarray) -> Image.Image:
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for cls_id, color in enumerate(CLASS_COLORS):
        if color[3] == 0:
            continue
        alpha_arr = ((label_hw == cls_id).astype(np.uint8) * color[3])
        colored = Image.new("RGBA", base.size, color[:3] + (0,))
        colored.putalpha(Image.fromarray(alpha_arr, mode="L"))
        overlay = Image.alpha_composite(overlay, colored)
    return Image.alpha_composite(base, overlay).convert("RGB")


def add_caption(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width, 18], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return out


def add_legend(img: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(img)
    x, y = 4, 22
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        if color[3] == 0:
            continue
        draw.rectangle([x, y, x + 12, y + 12], fill=color[:3])
        draw.text((x + 16, y), name, fill=(255, 255, 255))
        y += 16
    return img


def compute_iou(pred: np.ndarray, gt: np.ndarray, n_classes: int = 4):
    ious = []
    for c in range(n_classes):
        tp = np.logical_and(pred == c, gt == c).sum()
        fp = np.logical_and(pred == c, gt != c).sum()
        fn = np.logical_and(pred != c, gt == c).sum()
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    return ious


def print_iou(label: str, mean_ious):
    print(f"  [{label}] Per-class IoU:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name:12s}: {mean_ious[c]:.4f}")
    print(f"    {'mIoU':12s}: {np.nanmean(mean_ious):.4f}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", nargs="+",
                        default=["/work/courses/3dv/team32/training_data_modal/clip-000157.npz"])
    parser.add_argument("--reconstructor_path",
                        default="models/NeoVerse/reconstructor.ckpt",
                        help="Base reconstructor shared by both segmentation variants.")
    parser.add_argument("--hand_head_path",
                        default="models/NeoVerse/hand_seg_model_opt_best.ckpt",
                        help="Naïve Gaussian: DPT/hand_pred_head weights (predictions['seg_labels']).")
    parser.add_argument("--gs_mask_head_path",
                        default="models/NeoVerse/gs_mask_model_run20260513-154438_epoch009.ckpt",
                        help="Gaussian Params: nested {gs_head, gs_head_dynamic} state dict.")
    parser.add_argument("--output", default="outputs/eval_seg_gs_mask.mp4")
    parser.add_argument("--stream", default=None,
                        help="Which stream to use, e.g. 'stream1201-1'. Defaults to first found.")
    parser.add_argument("--stride", type=int, default=3,
                        help="Frame stride (matches training default of 3)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Frames per forward pass for the Gaussian Params variant.")
    parser.add_argument("--bg_alpha_thresh", type=float, default=None,
                        help="Exp#1: assign pixels with composited alpha below this to background.")
    parser.add_argument("--bg_bias", type=float, default=0.0,
                        help="Exp#1: additive bias on the background mask-logit channel before argmax.")
    parser.add_argument("--gsparam_seq_len", type=int, default=1,
                        help="Exp#2: >1 runs the Gaussian Params variant with multi-frame context "
                             "(chunks of this many frames merged into one Gaussian field).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("gs_mask eval needs a CUDA GPU — the rasterizer is CUDA-only.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---- load base reconstructor + both heads ----
    print("Loading reconstructor ...")
    model_manager = ModelManager()
    model_manager.load_model(args.reconstructor_path, device=device,
                             torch_dtype=torch.bfloat16)
    reconstructor = model_manager.fetch_model("reconstructor")
    if reconstructor is None:
        raise RuntimeError(
            f"ModelManager could not load a reconstructor from {args.reconstructor_path}."
        )
    load_hand_head(reconstructor, args.hand_head_path, device)
    load_gs_mask_head(reconstructor, args.gs_mask_head_path, device)

    naive_clip_ious, gsparam_clip_ious = [], []

    for npz_path in args.npz:
        clip_name = Path(npz_path).stem
        out_path = Path(args.output).parent / f"eval_seg_gs_mask_{clip_name}.mp4"
        print(f"\n{'='*50}")
        print(f"Clip: {clip_name}")

        npz = np.load(npz_path)
        stream_keys = [k for k in npz.keys() if k.startswith("images_")]
        stream = args.stream if args.stream else stream_keys[0].replace("images_", "")

        images_np = npz[f"images_{stream}"]        # (T, H, W, 3) uint8
        T = images_np.shape[0]
        frame_indices = list(range(0, T, args.stride))
        print(f"  stream={stream}  frames={T}  evaluating={len(frame_indices)}")

        imgs_tensor = torch.stack([
            F.to_tensor(Image.fromarray(images_np[i]))
            for i in frame_indices
        ], dim=0)                                  # (S, 3, H, W)
        S = imgs_tensor.shape[0]

        # ---- predictions ----
        naive_labels = predict_naive_labels(reconstructor, imgs_tensor, device)
        if args.gsparam_seq_len > 1:
            gsparam_labels = predict_gs_params_labels_multiframe(
                reconstructor, imgs_tensor, device, seq_len=args.gsparam_seq_len,
                bg_alpha_thresh=args.bg_alpha_thresh, bg_bias=args.bg_bias,
            )
        else:
            gsparam_labels = predict_gs_params_labels(
                reconstructor, imgs_tensor, device, args.batch_size,
                bg_alpha_thresh=args.bg_alpha_thresh, bg_bias=args.bg_bias,
            )

        # ---- GT labels ----
        gt_labels = np.stack([build_gt_label(npz, stream, i) for i in frame_indices])

        # ---- IoU (per frame, nanmean over frames) ----
        naive_mean = np.nanmean([compute_iou(naive_labels[s], gt_labels[s]) for s in range(S)], axis=0)
        gsparam_mean = np.nanmean([compute_iou(gsparam_labels[s], gt_labels[s]) for s in range(S)], axis=0)
        naive_clip_ious.append(naive_mean)
        gsparam_clip_ious.append(gsparam_mean)

        print_iou("Naïve Gaussian (DPT head)", naive_mean)
        print_iou("Gaussian Params (gs_mask)", gsparam_mean)

        # ---- render video: input | Naïve | Gaussian Params | GT ----
        H, W = images_np.shape[1], images_np.shape[2]
        out_frames = []
        for s, fi in enumerate(frame_indices):
            pil = Image.fromarray(images_np[fi])
            naive_overlay   = overlay_label(pil, naive_labels[s])
            gsparam_overlay = overlay_label(pil, gsparam_labels[s])
            gt_overlay      = overlay_label(pil, gt_labels[s])

            naive_miou   = np.nanmean(compute_iou(naive_labels[s], gt_labels[s]))
            gsparam_miou = np.nanmean(compute_iou(gsparam_labels[s], gt_labels[s]))

            col1 = add_legend(add_caption(pil,            f"{clip_name}  frame {fi}"))
            col2 = add_caption(naive_overlay,   f"Naïve Gaussian  mIoU={naive_miou:.2f}")
            col3 = add_caption(gsparam_overlay, f"Gaussian Params  mIoU={gsparam_miou:.2f}")
            col4 = add_caption(gt_overlay,      "Ground Truth")

            combined = Image.new("RGB", (W * 4, H))
            combined.paste(col1, (0,     0))
            combined.paste(col2, (W,     0))
            combined.paste(col3, (W * 2, 0))
            combined.paste(col4, (W * 3, 0))
            out_frames.append(combined)

        save_video(out_frames, str(out_path), fps=10)
        print(f"  Saved {out_path}")

        torch.cuda.empty_cache()

    # ---- overall summary ----
    print(f"\n{'='*50}")
    print("OVERALL (mean across all clips):")
    print_iou("Naïve Gaussian (DPT head)", np.nanmean(naive_clip_ious, axis=0))
    print_iou("Gaussian Params (gs_mask)", np.nanmean(gsparam_clip_ious, axis=0))


if __name__ == "__main__":
    main()