"""Interactive multi-model viewer for the three NeoVerse heads we trained.

One Gradio app that can load **all three kinds** of our checkpoints at once and
render them side-by-side for the same clip, with per-class render toggles
(right hand / left hand / object / background).

The three kinds load differently (see ``evals/benchmark_rec.py`` for the source
of truth this mirrors):

  - ``reconstructor`` — the base ``reconstructor.ckpt``. Auto-detected and built
    by ``ModelManager`` (flat, registered state dict, ``enable_norm=False``).
  - ``hand_seg`` — base reconstructor + the 2D ``hand_pred_head`` overlaid from a
    fine-tuned checkpoint. Segmentation lives in ``predictions["seg_labels"]``.
  - ``gs_mask`` — base reconstructor + the per-Gaussian ``gs_head`` /
    ``gs_head_dynamic`` overlaid (nested ``{"gs_head", "gs_head_dynamic"}`` state
    dict). Segmentation lives in the *rasterized per-Gaussian mask channels*.
  - ``vel_reg`` — a full fine-tuned ``WorldMirror`` loaded from ``model_state_dict``
    with ``strict=False``. Shares the wrapped-training-ckpt structure with gs_mask
    (hence loaded via the same ``_is_wrapped`` path).

Rendering toggles map onto the rasterizer's ``render_classes`` argument, which
keeps only Gaussians whose ``seg_label`` is in the selected set
(0=right_hand, 1=left_hand, 2=object, 3=background). For ``gs_mask`` models we
override each Gaussian's ``seg_label`` from its ``mask_logits`` argmax so the
toggles reflect *that* model's per-Gaussian segmentation rather than the
(untrained) 2D head.

Caching is two-level so the app stays interactive:
  - The forward pass (building the Gaussian splats) is the slow part — it is
    cached per (model, input clip). Loaded models are cached too.
  - Re-rasterizing with a different class subset / render mode is comparatively
    cheap and is what a toggle change triggers; results are also memoized so
    flipping a checkbox back is instant.

Launch (public share link by default, loads models/ checkpoints by default):
    python -m demos.multi_model_demo
    python -m demos.multi_model_demo --data_root /work/courses/3dv/team32/training_data_modal
    python -m demos.multi_model_demo --low_vram --no-share
"""

import argparse
import gc
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from PIL import Image
from torchvision.transforms import functional as F

from NeoVerse.diffsynth.models import ModelManager
from NeoVerse.diffsynth import save_video
from NeoVerse.diffsynth.utils.auxiliary import homo_matrix_inverse
from NeoVerse.diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror

# Reuse the benchmark's gs_mask loader and the eval visualisation helpers so this
# demo stays in lockstep with how the models are scored.
from evals.benchmark_rec import load_gs_mask_head
from demos.eval_segmentation import (
    CLASS_NAMES,        # ["right_hand", "left_hand", "object", "background"]
    overlay_label,
    add_caption,
    add_legend,
)
from demos.eval_segmentation_gs_mask import render_mask_labels


# ===================================================================== #
#                                CONFIG                                 #
# ===================================================================== #

DTYPE = torch.bfloat16
OUTPUT_DIR = "outputs/multi_model_demo"

parser = argparse.ArgumentParser()
parser.add_argument("--models_dir", default="models/NeoVerse",
                    help="Directory scanned for *.ckpt checkpoints (default: models/NeoVerse).")
parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt",
                    help="Base reconstructor that head-only models (hand_seg, gs_mask) overlay onto.")
parser.add_argument("--data_root", default="/work/courses/3dv/team32/training_data_modal",
                    help="Directory of clip-*.npz files offered in the clip dropdown. "
                         "Falls back gracefully if empty — you can still upload a video.")
parser.add_argument("--default_clip", default="clip-001053",
                    help="Clip selected on startup (like the evals' default). Falls back to the "
                         "first clip found under --data_root if this one is absent.")
parser.add_argument("--stride", type=int, default=3, help="Frame stride (training default is 3).")
parser.add_argument("--max_frames", type=int, default=8,
                    help="Cap on stride-sampled frames per render (keeps the app snappy).")
parser.add_argument("--cache_size", type=int, default=8,
                    help="Max number of cached forward passes (GPU memory bound).")
parser.add_argument("--low_vram", action="store_true",
                    help="Keep models on CPU; move to GPU only during their forward pass.")
parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=True,
                    help="Create a public share link (default: on).")
parser.add_argument("--server_port", type=int, default=7860)
args, _ = parser.parse_known_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================== #
#                          CHECKPOINT DISCOVERY                         #
# ===================================================================== #

def detect_kind(path: str) -> str:
    """Classify a checkpoint into one of the four loading paths.

    Filename hints come first (robust for our shipped checkpoints), then a
    structural fallback that mirrors the role-based loaders in benchmark_rec.py:
      - a nested ``gs_head`` in the (possibly wrapped) state dict   → gs_mask
      - a wrapped ``model_state_dict`` (full fine-tuned WorldMirror) → vel_reg
      - otherwise                                                    → reconstructor
    ``hand_seg`` is only reliably distinguishable from ``vel_reg`` by name, since
    both ship as full wrapped checkpoints — so it is matched by filename.
    """
    name = os.path.basename(path).lower()
    if "gs_mask" in name:
        return "gs_mask"
    if "velocity" in name or "vel_reg" in name:
        return "vel_reg"
    if "hand_seg" in name or "hand_pred" in name or "classification" in name:
        return "hand_seg"
    if "reconstructor" in name:
        return "reconstructor"

    # Structural fallback for checkpoints we don't recognise by name.
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return "reconstructor"
    wrapped = isinstance(raw, dict) and "model_state_dict" in raw
    sd = raw["model_state_dict"] if wrapped else raw
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    if "gs_head" in keys:
        return "gs_mask"
    if wrapped:
        # Head-only (hand) checkpoint vs full model: heads carry only hand_pred_head.*
        non_head = [k for k in keys if not k.startswith("hand_pred_head.")]
        return "hand_seg" if not non_head else "vel_reg"
    return "reconstructor"


def discover_checkpoints() -> "OrderedDict[str, dict]":
    """Map a friendly label → {path, kind} for every checkpoint under models_dir."""
    out: "OrderedDict[str, dict]" = OrderedDict()
    ckpts = sorted(Path(args.models_dir).glob("*.ckpt"))
    for p in ckpts:
        kind = detect_kind(str(p))
        label = f"{p.stem}  [{kind}]"
        out[label] = {"path": str(p), "kind": kind}
    return out


CHECKPOINTS = discover_checkpoints()


# ===================================================================== #
#                            MODEL LOADING                              #
# ===================================================================== #

# label -> loaded reconstructor (cached for the lifetime of the process).
_MODELS: "dict[str, torch.nn.Module]" = {}


def _load_base_reconstructor() -> WorldMirror:
    mm = ModelManager()
    mm.load_model(args.reconstructor_path,
                  device=("cpu" if args.low_vram else DEVICE), torch_dtype=DTYPE)
    rec = mm.fetch_model("reconstructor")
    if rec is None:
        raise gr.Error(f"ModelManager could not load a reconstructor from {args.reconstructor_path}.")
    return rec


def _overlay_hand_head(rec: WorldMirror, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    if not any(k.startswith("hand_pred_head.") for k in sd.keys()):
        sd = {f"hand_pred_head.{k}": v for k, v in sd.items()}
    else:
        sd = {k: v for k, v in sd.items() if k.startswith("hand_pred_head.")}
    rec.load_state_dict(sd, strict=False)
    rec.hand_pred_head.float()   # head stays fp32, matching training/eval


def load_model(label: str) -> torch.nn.Module:
    """Load (and cache) the reconstructor for a checkpoint label."""
    if label in _MODELS:
        return _MODELS[label]

    info = CHECKPOINTS[label]
    path, kind = info["path"], info["kind"]
    target = "cpu" if args.low_vram else DEVICE
    print(f"Loading model '{label}' (kind={kind}) from {path} ...", flush=True)

    if kind == "vel_reg":
        # Full fine-tuned WorldMirror — build the extended 16-channel architecture
        # directly and load the wrapped weights non-strictly (see benchmark_rec).
        rec = WorldMirror(enable_norm=False)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        rec.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        rec = _load_base_reconstructor()
        if kind == "hand_seg":
            _overlay_hand_head(rec, path)
        elif kind == "gs_mask":
            load_gs_mask_head(rec, path, target)
        # kind == "reconstructor": base only, nothing to overlay.

    rec = rec.to(device=target, dtype=DTYPE).eval()
    _MODELS[label] = rec
    print(f"  '{label}' ready on {target}.", flush=True)
    return rec


# ===================================================================== #
#                          INPUT → VIEWS                                #
# ===================================================================== #

def list_clips() -> list[str]:
    root = Path(args.data_root)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("clip-*.npz"))


def list_streams(clip: str | None) -> list[str]:
    if not clip:
        return []
    npz_path = Path(args.data_root) / f"{clip}.npz"
    if not npz_path.is_file():
        return []
    npz = np.load(str(npz_path), mmap_mode="r")
    return sorted(k.replace("images_", "") for k in npz.files if k.startswith("images_"))


def build_views_from_npz(clip: str, stream: str):
    """Return (views, raw_pil_frames, W, H) for a clip/stream."""
    npz_path = Path(args.data_root) / f"{clip}.npz"
    npz = np.load(str(npz_path))
    images_np = npz[f"images_{stream}"]            # (T, H, W, 3) uint8
    T = images_np.shape[0]
    frame_indices = list(range(0, T, args.stride))[: args.max_frames]
    pil = [Image.fromarray(images_np[i]) for i in frame_indices]
    imgs = torch.stack([F.to_tensor(p) for p in pil], dim=0)   # (S, 3, H, W)
    S = imgs.shape[0]
    views = {
        "img": imgs.unsqueeze(0).to(DEVICE),
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=DEVICE),
        "is_static": torch.zeros((1, S), dtype=torch.bool, device=DEVICE),
        "timestamp": torch.arange(S, dtype=torch.int64, device=DEVICE).unsqueeze(0),
    }
    H, W = images_np.shape[1], images_np.shape[2]
    return views, pil, W, H


def default_clip(clips: list[str]) -> str | None:
    """Pick the startup clip: the configured default if present, else the first."""
    if not clips:
        return None
    return args.default_clip if args.default_clip in clips else clips[0]


# ===================================================================== #
#                       FORWARD + RENDER (CACHED)                       #
# ===================================================================== #

# (model_label, input_signature) -> predictions (kept on GPU; LRU-evicted).
_PRED_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
# (model_label, input_signature, classes, mode) -> rendered PIL frames.
# This is what makes toggling instant: a class/mode combo seen before is never
# re-rasterized. Bounded to a few times the prediction cache.
_RENDER_CACHE: "OrderedDict[tuple, list]" = OrderedDict()


def _evict_predictions():
    while len(_PRED_CACHE) > args.cache_size:
        _, pred = _PRED_CACHE.popitem(last=False)
        del pred
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def get_predictions(label: str, views: dict, input_sig: tuple) -> dict:
    """Forward pass for one model on one input, memoised on (model, input)."""
    key = (label, input_sig)
    if key in _PRED_CACHE:
        _PRED_CACHE.move_to_end(key)
        return _PRED_CACHE[key]

    rec = load_model(label)
    kind = CHECKPOINTS[label]["kind"]
    if args.low_vram:
        rec.to(DEVICE)
    try:
        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")):
            predictions = rec(views, is_inference=True, use_motion=False)
    finally:
        if args.low_vram:
            rec.to("cpu")
            torch.cuda.empty_cache()

    # For gs_mask, make the per-class toggle reflect this model's *per-Gaussian*
    # segmentation: overwrite each Gaussian's seg_label with its mask_logits argmax.
    # (Otherwise render_classes filters on the untrained 2D hand head.)
    if kind == "gs_mask":
        for gs in predictions["splats"][0]:
            if getattr(gs, "mask_logits", None) is not None:
                gs.seg_label = gs.mask_logits.float().argmax(dim=-1).to(torch.int64)

    _PRED_CACHE[key] = predictions
    _evict_predictions()
    return predictions


@torch.no_grad()
def render_rgb_frames(rec, predictions, classes: tuple[int, ...], W: int, H: int) -> list[Image.Image]:
    """Rasterize RGB at the input cameras, keeping only the selected classes."""
    gaussians = predictions["splats"]
    c2w = predictions["rendered_extrinsics"][0]
    K = predictions["rendered_intrinsics"][0]
    ts = predictions["rendered_timestamps"][0]
    w2c = homo_matrix_inverse(c2w)

    # All four selected → pass None (no filtering = full scene).
    render_classes = None if len(classes) == 4 else list(classes)
    rgb, _, _, _ = rec.gs_renderer.rasterizer.forward(
        gaussians, render_viewmats=[w2c], render_Ks=[K], render_timestamps=[ts],
        sh_degree=0, width=W, height=H, render_classes=render_classes,
    )
    return [
        Image.fromarray((rgb[0, i].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
        for i in range(rgb.shape[1])
    ]


@torch.no_grad()
def render_seg_frames(rec, predictions, kind: str, raw_pil, W: int, H: int) -> list[Image.Image]:
    """Segmentation overlay on the input frames (model's own segmentation source)."""
    if kind == "gs_mask":
        labels = render_mask_labels(rec, predictions, width=W, height=H)   # [S,H,W]
    else:
        # hand_seg / reconstructor / vel_reg: the 2D hand_pred_head output.
        labels = predictions["seg_labels"][0].float().argmax(dim=-1).cpu().numpy()
    return [overlay_label(raw_pil[i], labels[i]) for i in range(len(raw_pil))]


def render_model_frames(label, views, raw_pil, input_sig, classes, mode, W, H) -> list[Image.Image]:
    """Per-model frames for the combined strip, memoised on the full render signature.

    A cache hit (same model, input, classes and mode) returns instantly without
    re-rasterizing; a miss still reuses the cached forward pass via get_predictions.
    """
    key = (label, input_sig, classes, mode)
    if key in _RENDER_CACHE:
        _RENDER_CACHE.move_to_end(key)
        return _RENDER_CACHE[key]

    rec = load_model(label)
    kind = CHECKPOINTS[label]["kind"]
    predictions = get_predictions(label, views, input_sig)
    if mode == "Segmentation overlay":
        frames = render_seg_frames(rec, predictions, kind, raw_pil, W, H)
    else:
        frames = render_rgb_frames(rec, predictions, classes, W, H)

    _RENDER_CACHE[key] = frames
    while len(_RENDER_CACHE) > 4 * args.cache_size:
        _RENDER_CACHE.popitem(last=False)
    return frames


# ===================================================================== #
#                            MAIN CALLBACK                              #
# ===================================================================== #

def run(clip, stream, model_labels, class_choices, mode):
    if not model_labels:
        raise gr.Error("Select at least one model to render.")

    # ---- build the shared input from the selected clip/stream ----
    if not clip:
        raise gr.Error("No clip available — check --data_root contains clip-*.npz files.")
    if not stream:
        streams = list_streams(clip)
        if not streams:
            raise gr.Error(f"No image streams found in {clip}.")
        stream = streams[0]
    views, raw_pil, W, H = build_views_from_npz(clip, stream)
    input_sig = ("npz", clip, stream, args.stride, args.max_frames)
    src_label = f"{clip}  {stream}"

    classes = tuple(sorted(CLASS_NAMES.index(c) for c in class_choices)) if class_choices else ()
    if mode == "RGB render" and not classes:
        raise gr.Error("Select at least one class to render (or switch to Segmentation overlay).")

    # ---- per-model render (each memoised) + combined strip ----
    columns = [[add_caption(p, f"input  ·  {src_label}") for p in raw_pil]]
    info_lines = [f"Input: {src_label}  |  frames: {len(raw_pil)}  |  {W}x{H}"]

    for label in model_labels:
        frames = render_model_frames(label, views, raw_pil, input_sig, classes, mode, W, H)
        tag = mode if mode == "Segmentation overlay" else "+".join(CLASS_NAMES[i] for i in classes)
        captioned = [add_caption(f, f"{label.split('  [')[0]}  ·  {tag}") for f in frames]
        columns.append(captioned)
        n_gauss = sum(gs.means.shape[0]
                      for gs in get_predictions(label, views, input_sig)["splats"][0])
        info_lines.append(f"{label}: {n_gauss:,} gaussians")

    # ---- stitch all columns into one side-by-side video ----
    n = min(len(c) for c in columns)
    combined = []
    for i in range(n):
        row = [c[i] for c in columns]
        rw = sum(im.width for im in row)
        rh = max(im.height for im in row)
        canvas = Image.new("RGB", (rw, rh))
        x = 0
        for im in row:
            canvas.paste(im, (x, 0))
            x += im.width
        combined.append(add_legend(canvas) if mode == "Segmentation overlay" else canvas)

    combined_path = os.path.join(OUTPUT_DIR, "combined.mp4")
    save_video(combined, combined_path, fps=8)
    return combined_path, "\n".join(info_lines)


# ===================================================================== #
#                                  UI                                   #
# ===================================================================== #

def build_ui():
    clips = list_clips()
    start_clip = default_clip(clips)
    start_streams = list_streams(start_clip)
    default_models = list(CHECKPOINTS.keys())[: min(2, len(CHECKPOINTS))]

    with gr.Blocks(title="NeoVerse — Multi-Model Viewer") as demo:
        gr.Markdown(
            "# NeoVerse — Multi-Model Viewer\n"
            "Load any of the three trained heads (**hand_seg**, **gs_mask**, "
            "**vel_reg**) plus the **base reconstructor** and render them "
            "side-by-side on the same clip. Toggle which segmentation classes are "
            "rendered, or switch to a segmentation overlay. Forward passes are "
            "cached, so flipping toggles re-renders quickly."
        )

        with gr.Row():
            with gr.Column(scale=1):
                clip_dd = gr.Dropdown(clips, value=start_clip, label="Clip (clip-*.npz)")
                stream_dd = gr.Dropdown(start_streams,
                                        value=(start_streams[0] if start_streams else None),
                                        label="Stream")

                model_sel = gr.CheckboxGroup(
                    list(CHECKPOINTS.keys()), value=default_models,
                    label="Models to render (load several at once)",
                )
                class_sel = gr.CheckboxGroup(
                    CLASS_NAMES, value=list(CLASS_NAMES),
                    label="Render classes (RGB mode)",
                )
                mode_sel = gr.Radio(
                    ["RGB render", "Segmentation overlay"],
                    value="RGB render", label="Render mode",
                )
                run_btn = gr.Button("Render", variant="primary")
                info_box = gr.Textbox(label="Info", interactive=False, lines=6)

            with gr.Column(scale=2):
                out_video = gr.Video(label="Input | model renders (side-by-side)", height=460)

        # Repopulate streams when the clip changes.
        def _on_clip(c):
            streams = list_streams(c)
            return gr.update(choices=streams, value=(streams[0] if streams else None))
        clip_dd.change(_on_clip, clip_dd, stream_dd)

        run_btn.click(
            run,
            inputs=[clip_dd, stream_dd, model_sel, class_sel, mode_sel],
            outputs=[out_video, info_box],
        )

    return demo


if __name__ == "__main__":
    if DEVICE != "cuda":
        print("WARNING: no CUDA device found — the Gaussian-splat rasterizer is CUDA-only "
              "and rendering will fail. Run this on a GPU node.", flush=True)
    print(f"Discovered {len(CHECKPOINTS)} checkpoint(s) in {args.models_dir}:", flush=True)
    for label, info in CHECKPOINTS.items():
        print(f"  - {label}  ({info['path']})", flush=True)

    build_ui().queue(max_size=8).launch(
        share=args.share, show_error=True,
        server_name="0.0.0.0", server_port=args.server_port,
    )
