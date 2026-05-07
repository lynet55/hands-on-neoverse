"""
Evaluate segmentation on a val-set NPZ clip.

Outputs:
  - Per-class IoU and mIoU printed to stdout
  - Side-by-side video: input | prediction | ground truth

Usage:
    python eval_segmentation.py \
        --npz diffsynth/data/training_data/clip-001053.npz \
        --hand_head_path models/NeoVerse/hand_seg_model_opt_run20260426-130617_epoch004.ckpt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from diffsynth.models import ModelManager
from diffsynth import save_video
from diffsynth.data.training.training_gs_mask import GsMaskReconstructor, TrainConfig

CLASS_NAMES = ["right_hand", "left_hand", "object", "background"]
# RGBA overlay colors
CLASS_COLORS = [
    (255,  60,  60, 160),   # right_hand  — red
    ( 60, 120, 255, 160),   # left_hand   — blue
    ( 60, 220,  60, 160),   # object      — green
    (  0,   0,   0,   0),   # background  — transparent
]


# ---------- helpers ----------

def build_gt_label(npz, stream: str, frame_idx: int) -> np.ndarray:
    """Combine per-class binary masks into a single HxW label array."""
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
    for cls_id, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
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


def load_hand_head_model(args, device):
    print(f"Loading reconstructor ...")
    model_manager = ModelManager()
    model_manager.load_model(args.reconstructor_path, device="cpu",
                             torch_dtype=torch.bfloat16)
    reconstructor = model_manager.fetch_model("reconstructor")

    print(f"Loading hand head from {args.hand_head_path} ...")
    ckpt = torch.load(args.hand_head_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    if not any(k.startswith("hand_pred_head.") for k in sd.keys()):
        sd = {f"hand_pred_head.{k}": v for k, v in sd.items()}
    else:
        sd = {k: v for k, v in sd.items() if k.startswith("hand_pred_head.")}
    reconstructor.load_state_dict(sd, strict=False)
    reconstructor.hand_pred_head.float()
    reconstructor.to(device).eval()
    return reconstructor


def load_gs_mask_model(args, device):
    print(f"Loading GS mask model from {args.gs_mask_path} ...")
    cfg = TrainConfig(
        device=device,
        reconstruction_model_path=args.reconstructor_path,
        resume_from=None,
    )
    model = GsMaskReconstructor(cfg)
    ckpt = torch.load(args.gs_mask_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    model.reconstructor.gs_renderer.gs_head.load_state_dict(sd["gs_head"], strict=True)
    if "gs_head_dynamic" in sd and hasattr(model.reconstructor.gs_renderer, "gs_head_dynamic"):
        model.reconstructor.gs_renderer.gs_head_dynamic.load_state_dict(sd["gs_head_dynamic"], strict=True)
    model.set_eval_mode()
    return model


@torch.no_grad()
def predict_hand_head(reconstructor, imgs_tensor, device):
    S = imgs_tensor.shape[0]
    views = {
        "img":       imgs_tensor.unsqueeze(0).to(device),
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, S), dtype=torch.bool, device=device),
        "timestamp": torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0),
    }

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        predictions = reconstructor(views, is_inference=True, use_motion=False)
    seg_logits = predictions["seg_labels"][0]          # (S, H, W, 4)
    return seg_logits.float().argmax(dim=-1).cpu().numpy()


@torch.no_grad()
def predict_gs_mask(model, imgs_tensor, device, batch_size):
    pred_chunks = []
    for start in range(0, imgs_tensor.shape[0], batch_size):
        chunk = imgs_tensor[start:start + batch_size]
        logits = model.forward(chunk)                  # (B, 4, H, W)
        pred_chunks.append(logits.float().argmax(dim=1).cpu().numpy())
    return np.concatenate(pred_chunks, axis=0)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", nargs="+",
                        default=["diffsynth/data/training_data/clip-001053.npz"])
    parser.add_argument("--reconstructor_path",
                        default="models/NeoVerse/reconstructor.ckpt")
    parser.add_argument("--hand_head_path",
                        default="models/NeoVerse/hand_seg_model_opt_run20260426-130617_epoch004.ckpt")
    parser.add_argument("--gs_mask_path", default=None,
                        help="Evaluate a rendered-GS-mask checkpoint instead of hand_pred_head.")
    parser.add_argument("--output", default="outputs/eval_seg.mp4")
    parser.add_argument("--stream", default=None,
                        help="Which stream to use, e.g. 'stream1201-1'. Defaults to first found.")
    parser.add_argument("--stride", type=int, default=3,
                        help="Frame stride (matches training default of 3)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for GS-mask evaluation.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---- load model ----
    if args.gs_mask_path:
        model = load_gs_mask_model(args, device)
        eval_mode = "gs_mask"
    else:
        model = load_hand_head_model(args, device)
        eval_mode = "hand_head"

    all_clip_ious = []

    for npz_path in args.npz:
        clip_name = Path(npz_path).stem
        output_arg = Path(args.output)
        if len(args.npz) == 1:
            out_path = output_arg
        else:
            out_path = output_arg.parent / f"{output_arg.stem}_{clip_name}{output_arg.suffix}"
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

        if eval_mode == "gs_mask":
            pred_labels = predict_gs_mask(model, imgs_tensor, device, args.batch_size)
        else:
            pred_labels = predict_hand_head(model, imgs_tensor, device)
        S = pred_labels.shape[0]

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
        H, W = images_np.shape[1], images_np.shape[2]
        out_frames = []
        for s, fi in enumerate(frame_indices):
            pil = Image.fromarray(images_np[fi])
            pred_overlay = overlay_label(pil, pred_labels[s])
            gt_overlay   = overlay_label(pil, gt_labels[s])

            frame_miou = np.nanmean(compute_iou(pred_labels[s], gt_labels[s]))
            col1 = add_legend(add_caption(pil,          f"{clip_name}  frame {fi}"))
            col2 = add_caption(pred_overlay, f"Prediction  mIoU={frame_miou:.2f}")
            col3 = add_caption(gt_overlay,   "Ground Truth")

            combined = Image.new("RGB", (W * 3, H))
            combined.paste(col1, (0,   0))
            combined.paste(col2, (W,   0))
            combined.paste(col3, (W*2, 0))
            out_frames.append(combined)

        save_video(out_frames, str(out_path), fps=10)
        print(f"  Saved {out_path}")

        del imgs_tensor
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
