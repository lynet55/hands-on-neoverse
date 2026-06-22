"""Velocity regularization for background Gaussians in WorldMirror.

Goal
----
We want background Gaussians (those whose predicted segmentation class is 3)
to have near-zero velocity, while leaving hand/object Gaussians free to move.
The hand-segmentation head already knows where background is; this training
loop uses that signal to regularise the 4DGS velocity field.

Since we only have raw input video (no GT depth, cameras, or optical flow),
we use the frozen pretrained model as a teacher for everything except the RGB
reconstruction loss.  The teacher provides stable pseudo-GT that prevents
catastrophic forgetting while the backbone is partially unfrozen.

Loss summary
------------
  Lrgb          -- L2 + LPIPS vs. input frames  (true GT)
  Lregular      -- alpha coverage penalty        (self-supervised)
  Lbg_vel       -- L2 norm of background Gaussian velocities  (core goal)
  Lpreserve_fg  -- keep foreground pixel-velocity close to teacher
  Lcamera       -- student camera params ≈ teacher  (distillation)
  Ldepth        -- student depth ≈ teacher           (distillation)
  Lseg          -- student seg logits ≈ teacher      (distillation)

Lmotion (NeoVerse paper) is intentionally omitted: it requires GT velocity
fields from dynamic-scene datasets that are not available here, and applying
it naively from teacher predictions would contradict the background-velocity
goal.
"""

import copy
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from NeoVerse.diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from NeoVerse.diffsynth.models.model_manager import ModelManager
from NeoVerse.diffsynth.utils.auxiliary import homo_matrix_inverse
from training.SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceHandObjectDataset(HandObjectSegmentationDataset):
    """Returns fixed-length sequences of consecutive frames.

    Each sample is a tuple (images, masks) where
      images : Float[T, 3, H, W]  in [0, 1]
      masks  : Float[T, 4, H, W]  one-hot channels: right, left, obj, background
    """

    def __init__(
        self,
        data_root: str = "/work/courses/3dv/team32/training_data_modal",
        num_key_frames: int = 4,
        frame_stride: int = 1,
        random_reverse: bool = True,
        streams=None,
        clip_names=None,
    ):
        self.num_key_frames = num_key_frames
        self.num_frames = 2 * num_key_frames - 1
        self.frame_stride = frame_stride
        self.random_reverse = random_reverse
        self._clip_names_filter = set(clip_names) if clip_names is not None else None
        super().__init__(data_root=data_root, streams=streams)

    def _build_index(self) -> None:
        self.samples = []
        data_root_path = Path(self.data_root)
        if not data_root_path.exists():
            raise FileNotFoundError(
                f"data_root does not exist: {data_root_path}. "
                f"Expected clip-*.npz files under this directory."
            )
        for npz_path in sorted(data_root_path.glob("clip-*.npz")):
            clip_name = npz_path.stem
            if self._clip_names_filter is not None and clip_name not in self._clip_names_filter:
                continue
            npz = np.load(str(npz_path), mmap_mode="r")
            n_frames = next(npz[k].shape[0] for k in npz.files if k.startswith("images_"))
            for stream in self.streams:
                if f"images_{stream}" not in npz.files:
                    continue
                max_start = n_frames - (self.num_frames - 1) * self.frame_stride
                if max_start <= 0:
                    continue
                for frame_idx in range(max_start):
                    self.samples.append(
                        {"clip_name": clip_name, "stream": stream, "frame_idx": frame_idx}
                    )
        if not self.samples:
            raise RuntimeError(
                f"No sequence samples found in data_root: {self.data_root}. "
                f"Check that clips are long enough for num_key_frames={self.num_key_frames} "
                f"and frame_stride={self.frame_stride}, and that clip_names/streams match."
            )

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        clip_name, stream, frame_idx = s["clip_name"], s["stream"], s["frame_idx"]
        npz = np.load(str(self.data_root / f"{clip_name}.npz"), mmap_mode="r")

        indices = [frame_idx + i * self.frame_stride for i in range(self.num_frames)]
        images, masks = [], []
        for i in indices:
            image = (
                torch.tensor(npz[f"images_{stream}"][i], dtype=torch.float32)
                .permute(2, 0, 1) / 255.0
            )
            right = torch.tensor(npz[f"masks_{stream}_hand_RIGHT"][i] > 0)
            left  = torch.tensor(npz[f"masks_{stream}_hand_LEFT"][i]  > 0)
            obj   = torch.tensor(npz[f"masks_{stream}_object"][i]     > 0)
            background = ~(right | left | obj)
            masks.append(torch.stack([right, left, obj, background], dim=0).float())
            images.append(image)

        images = torch.stack(images, dim=0)  # (T, 3, H, W)
        masks  = torch.stack(masks,  dim=0)  # (T, 4, H, W)
        if self.random_reverse and torch.rand(1).item() > 0.5:
            images = torch.flip(images, dims=[0])
            masks  = torch.flip(masks,  dims=[0])
        return images, masks


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_root: str = "/work/courses/3dv/team32/training_data_modal"
    streams: list[str] = None
    clip_names: list[str] = None
    num_key_frames: int = 3
    # Keyframe gap for training. After _build_views reorders to every-other keyframes,
    # a target sits `frame_stride` original frames from its keyframes. Set to 3 to match
    # the benchmark/demo regime (keyframes 6 apart) so the velocity heads are trained at
    # the SAME motion scale they are evaluated at — a mismatch re-creates the train/eval
    # gap that caused earlier collapse. Revert to 1 for the 2x-framerate (small-motion)
    # regime, but then also evaluate there.
    frame_stride: int = 3
    random_reverse: bool = True

    # Which backbone submodules to unfreeze alongside the velocity heads.
    # Empty by default: an interpolation eval showed that leaving the motion
    # backbone trainable let the *backward* velocity field collapse to ~0
    # everywhere (foreground included) while forward stayed alive, producing
    # "two hands" ghosting. Training only the velocity heads removes that
    # degree of freedom. Re-add substrings here only with a real reconstruction
    # signal on velocity (fix #4) in place.
    # Unfreeze the motion-specific blocks ONLY (they feed the velocity / dynamic-GS heads).
    # NOT frame_blocks (the shared trunk feeding camera/depth/gs/seg) — keeping those frozen
    # isolates the unfreeze to velocity, so the static reconstruction can't be damaged.
    # Rationale vs the 24h failure: that run unfroze frame_blocks too, had no interp signal,
    # and used one high LR; here interp+preserve_fg constrain velocity, only motion blocks
    # move, and they get a separate low LR (backbone_learning_rate).
    backbone_unfreeze_substrings: list[str] = field(
        default_factory=lambda: [
            "visual_geometry_transformer.motion_fwd_blocks",
            "visual_geometry_transformer.motion_bwd_blocks",
        ]
    )

    # Model paths
    reconstruction_model_path: str = "models/NeoVerse/reconstructor.ckpt"
    hand_seg_model_path: str = "models/NeoVerse/hand_seg_model_opt_best.ckpt"
    save_model_path_prefix: str = "models/NeoVerse/velocity_regularization"

    # Resume STUDENT weights + optimizer from a prior velocity_regularization checkpoint.
    # The teacher stays the original reconstructor (anti-forgetting anchor) regardless.
    # When set, the run_id (hence log_dir + checkpoint namespace) is inherited from the
    # filename so TensorBoard logging continues on the same run from the same global step.
    # Start from the PURE reconstructor (= the teacher), not a prior velreg checkpoint:
    # student = teacher at step 0, so the velocity heads begin in the original's best
    # fg/bg-separated basin (4-6x) with preserve_fg=0. The old _batch20400 ckpt was the
    # parity run that never moved background velocity, so it offered no advantage.
    resume_checkpoint: str = None

    # When True, freeze everything except velocity heads + backbone_unfreeze_substrings.
    # When False, all parameters are trainable (use with a low learning rate).
    freeze_except_velocity: bool = True

    # Training hyperparameters
    batch_size: int = 1
    epochs: int = 10
    # Lowered 10x (was 1e-4): every collapse happened within the ~500-step warmup window,
    # i.e. the model was perturbed out of the healthy "fg/bg separated" basin as LR ramped.
    # A smaller step lets bg_vel nudge background down without leaving that basin. If it
    # still collapses at 1e-5, LR is not the lever (capacity is) — see backbone unfreeze.
    learning_rate: float = 1e-5
    # Separate, much lower LR for any unfrozen backbone (motion) params. Pretrained
    # weights need gentle updates; the 24h collapse used a single 1e-4 for everything.
    backbone_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    # Loss weights
    # -- true-GT losses (input video is the only GT we have) --
    rgb_loss_weight: float = 1.0       # Lrgb: keyframe reconstruction (monitor only)
    lpips_loss_weight: float = 0.1     # weight of LPIPS inside the interpolation loss
    regularization_weight: float = 0.1 # Lregular: alpha coverage

    # -- THE velocity learning signal --
    # Render the held-out in-between frames at fractional timestamps (velocity
    # transitions the keyframe Gaussians to reach them) and supervise against the
    # real frames. This is the ONLY term that backprops real data into the velocity
    # heads — keyframe RGB is rendered at integer timestamps where velocity is the
    # identity, so it cannot.
    interp_loss_weight: float = 1.0

    # -- core goal: push background Gaussian velocities to zero --
    # History: weight 1.0 did nothing (bg stayed ~0.007); weight 20.0 collapsed the
    # ENTIRE field to ~0 (fg included) in ~2k steps, because interp's velocity gradient
    # is too weak (at large gaps static ≈ correct) to counter it. Kept gentle and paired
    # with a foreground FLOOR (preserve_fg) that holds fg up while this zeros bg on the
    # background mask only. Tune live via vel_bg_*_mag (should fall) and vel_fg_*_mag
    # (should hold).
    bg_gaussian_vel_weight: float = 5.0
    # "l1" penalises ‖v‖ (constant gradient -> actually drives bg velocity to 0);
    # "l2" penalises ‖v‖² (gradient vanishes near 0, so it stalls at the teacher's
    # ~0.005 unless given a huge weight that collapses the shared head). L1 is much
    # stronger per unit weight, so if foreground velocity starts dropping, lower
    # bg_gaussian_vel_weight rather than raising it.
    bg_vel_loss_norm: str = "l1"  # "l1" | "l2"

    # -- distillation losses: prevent catastrophic forgetting --
    # These use the frozen teacher as pseudo-GT; they carry no new information
    # but stabilise all predictions that we are not deliberately changing.
    camera_distill_weight: float = 5.0
    depth_distill_weight: float = 1.0
    # The FOREGROUND FLOOR. interp's velocity gradient turned out too weak to keep the
    # foreground alive against bg_vel (the field collapsed with this at 0). This anchors
    # fg velocity to the teacher's (now stride-3, reasonable-scale) value so it cannot
    # collapse, while bg_vel zeros only the background. Different masks -> they coexist.
    # preserve_fg is now L1 (constant gradient, matched to bg_vel's L1) — this is the
    # foreground anchor that holds it up while bg_vel pushes background down. Keep it
    # >= bg_gaussian_vel_weight so the foreground is held at least as hard as bg is pushed.
    # If foreground still falls, raise it; if bg won't drop, lower bg_gaussian_vel_weight.
    preserve_foreground_vel_weight: float = 10.0  # pixel-vel head, fg pixels only
    # Light insurance against fwd/bwd asymmetry.
    velocity_balance_weight: float = 1.0
    preserve_segmentation_weight: float = 1.0

    # LR schedule
    warmup_steps: int = 500
    lr_min_factor: float = 0.1

    # Checkpointing
    checkpoint_interval_batches: int = 100

    # In-training interpolation eval (logged to TB + printed) so you can watch the
    # held-out interpolation quality and fg/bg velocity without stopping to run the
    # demo. Uses the same window each time for comparability. 0 disables.
    eval_interval_batches: int = 500
    eval_clip: str = "clip-001160"  # same window you've been eyeballing in the demo
    eval_stream: str = None         # default: first stream
    eval_window_index: int = 0
    eval_num_keyframes: int = 3
    # None -> tied to frame_stride (in __post_init__) so the eval is always at the SAME
    # motion scale the model trains on. Override only if you deliberately want a
    # different (out-of-distribution) eval regime.
    eval_frame_stride: int = None
    eval_img_shape: tuple = (280, 280)
    val_fraction: float = 0.1      # for deriving the test split the eval window comes from

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # When resuming: by default this run gets its OWN fresh run_id -> its own
    # TensorBoard log dir + checkpoint namespace (a separate TB run), rather than
    # appending into the resumed run's existing log. Set True to continue the SAME
    # TB run/namespace (the old append-in-place behaviour). Either way the global
    # step continues from where the checkpoint left off.
    inherit_run_id_on_resume: bool = False

    # Unique id for this run, used to namespace checkpoints and logs so that
    # different runs don't overwrite each other.
    run_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # Logging. Left as None so __post_init__ can derive it from run_id, keeping
    # the checkpoint timestamp and the TensorBoard run in sync.
    log_dir: str = None
    num_workers: int = 2
    pin_memory: bool = True

    def __post_init__(self):
        # Tie the eval motion scale to training unless explicitly overridden.
        if self.eval_frame_stride is None:
            self.eval_frame_stride = self.frame_stride
        # When resuming, only inherit the prior run's id (continuing the same TB run +
        # checkpoint namespace) if explicitly asked. Otherwise keep the fresh run_id so
        # this run logs to its own TensorBoard directory.
        if self.resume_checkpoint is not None and self.inherit_run_id_on_resume:
            m = re.search(r"velocity_regularization_(\d{8}-\d{6})", self.resume_checkpoint)
            if m:
                self.run_id = m.group(1)
        if self.log_dir is None:
            self.log_dir = f"runs/velocity_regularization_{self.run_id}"

    @property
    def num_frames(self) -> int:
        return 2 * self.num_key_frames - 1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dbg(msg: str):
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detach_predictions(predictions: dict) -> dict:
    """Return a shallow copy with every tensor detached.

    Used to guard teacher outputs before passing them into the student renderer
    so that no gradient can flow back through the teacher path.
    """
    return {
        k: v.detach() if isinstance(v, torch.Tensor) else v
        for k, v in predictions.items()
    }


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class VelocityRegularizationModel:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        dbg(f"Loading reconstructor from {cfg.reconstruction_model_path} ...")
        model_manager = ModelManager()
        model_manager.load_model(
            cfg.reconstruction_model_path,
            device=cfg.device,
            torch_dtype=torch.bfloat16,
        )
        self.reconstructor: WorldMirror = model_manager.fetch_model("reconstructor")
        self.reconstructor.gs_renderer.training = False

        if cfg.hand_seg_model_path is not None:
            dbg(f"Loading hand_pred_head from {cfg.hand_seg_model_path} ...")
            hand_ckpt = torch.load(cfg.hand_seg_model_path, map_location="cpu")
            hand_sd = hand_ckpt.get("model_state_dict", hand_ckpt)
            if not any(k.startswith("hand_pred_head.") for k in hand_sd):
                hand_sd = {f"hand_pred_head.{k}": v for k, v in hand_sd.items()}
            else:
                hand_sd = {k: v for k, v in hand_sd.items() if k.startswith("hand_pred_head.")}
            device = next(self.reconstructor.parameters()).device
            hand_sd = {k: v.to(device) for k, v in hand_sd.items()}
            self.reconstructor.load_state_dict(hand_sd, strict=False)
            self.reconstructor.hand_pred_head.float()
            dbg("hand_pred_head weights loaded.")

        dbg("Reconstructor loaded.")

        # Frozen copy used as teacher / anti-forgetting anchor.
        self.teacher: WorldMirror = copy.deepcopy(self.reconstructor).eval()
        self.teacher.requires_grad_(False)

        # --- Parameter freezing ---
        if cfg.freeze_except_velocity:
            self.reconstructor.requires_grad_(False)
            n_train, n_total, n_backbone = 0, 0, 0
            for name, param in self.reconstructor.named_parameters():
                n_total += 1
                is_vel = "velocity_fwd_head" in name or "velocity_bwd_head" in name
                is_backbone = any(sub in name for sub in cfg.backbone_unfreeze_substrings)
                if is_vel or is_backbone:
                    param.requires_grad = True
                    n_train += 1
                if is_backbone:
                    n_backbone += 1
            dbg(f"Trainable parameters: {n_train}/{n_total} ({n_backbone} backbone)")
            if n_train == 0:
                raise RuntimeError(
                    "No parameters matched velocity heads or backbone_unfreeze_substrings. "
                    "Check TrainConfig.backbone_unfreeze_substrings."
                )
            # Fail loudly rather than silently train heads-only if the names are wrong.
            if cfg.backbone_unfreeze_substrings and n_backbone == 0:
                raise RuntimeError(
                    f"backbone_unfreeze_substrings={cfg.backbone_unfreeze_substrings} matched "
                    "NO parameters — check the module names (otherwise it's a silent no-op)."
                )
        else:
            self.reconstructor.requires_grad_(True)

        # Velocity heads must always be float32 so loss gradients are stable.
        self.reconstructor.velocity_fwd_head = self.reconstructor.velocity_fwd_head.float()
        self.reconstructor.velocity_bwd_head = self.reconstructor.velocity_bwd_head.float()
        for param in self.reconstructor.velocity_fwd_head.parameters():
            param.requires_grad = True
        for param in self.reconstructor.velocity_bwd_head.parameters():
            param.requires_grad = True

        # Any UNFROZEN backbone params -> float32 master weights too, so the low-LR (1e-6)
        # updates aren't lost to bf16 precision. (Forward still autocasts to bf16; only the
        # accumulating master weight is float32, standard mixed-precision.)
        if cfg.freeze_except_velocity and cfg.backbone_unfreeze_substrings:
            n_fp32 = 0
            for name, param in self.reconstructor.named_parameters():
                is_vel = "velocity_fwd_head" in name or "velocity_bwd_head" in name
                if param.requires_grad and not is_vel and param.dtype != torch.float32:
                    param.data = param.data.float()
                    n_fp32 += 1
            dbg(f"Converted {n_fp32} unfrozen backbone params to float32.")

        # Safety net: confirm ONLY the velocity heads and the explicitly-listed backbone
        # substrings are trainable, so the depth / camera / seg / gs heads can never be
        # touched by a typo in backbone_unfreeze_substrings (e.g. a substring that also
        # matched a static head).
        if cfg.freeze_except_velocity:
            allowed = ("velocity_fwd_head", "velocity_bwd_head", *cfg.backbone_unfreeze_substrings)
            leaked = sorted({
                n for n, p in self.reconstructor.named_parameters()
                if p.requires_grad and not any(a in n for a in allowed)
            })
            if leaked:
                raise RuntimeError(
                    "Unexpected trainable params would change non-velocity heads: "
                    f"{leaked[:8]}{' ...' if len(leaked) > 8 else ''}"
                )

        self.device = cfg.device

        # --- Resume STUDENT weights from a prior velocity_regularization checkpoint ---
        # Done AFTER the teacher deepcopy (teacher must stay the original) and AFTER the
        # velocity heads are cast to float32 (so dtypes match the saved heads). The raw
        # checkpoint dict is stashed for train() to restore the optimizer + global step.
        self.resume_ckpt = None
        if cfg.resume_checkpoint is not None:
            if not os.path.exists(cfg.resume_checkpoint):
                raise FileNotFoundError(
                    f"resume_checkpoint not found: {cfg.resume_checkpoint}"
                )
            dbg(f"Resuming student weights from {cfg.resume_checkpoint} ...")
            ckpt = torch.load(cfg.resume_checkpoint, map_location=cfg.device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = self.reconstructor.load_state_dict(sd, strict=False)
            dbg(f"Resumed weights: {len(sd)} tensors, "
                f"{len(missing)} missing, {len(unexpected)} unexpected.")
            if unexpected:
                dbg(f"  e.g. unexpected: {unexpected[:5]}")
            # Loading may have changed head dtypes back to the checkpoint's; re-assert float32.
            self.reconstructor.velocity_fwd_head.float()
            self.reconstructor.velocity_bwd_head.float()
            self.resume_ckpt = ckpt

        self.lpips_fn = None
        if lpips_lib is not None:
            try:
                self.lpips_fn = lpips_lib.LPIPS(net="vgg").to(self.device)
            except Exception:
                self.lpips_fn = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_views(self, images: torch.Tensor):
        """Reorder a window of consecutive frames into the contiguous
        ``[keyframes..., targets...]`` layout the model expects, and give the
        held-out (target) frames *fractional* timestamps.

        Even original positions -> keyframes (the context the model sees), placed
        first. Odd original positions -> interpolation targets, placed after, at
        half-integer timestamps (0.5, 1.5, ...). Two reasons this matters:
          * ``prepare_contexts`` slices the first ``(is_target == False)`` frames as
            context, so the ``[keyframes..., targets...]`` ordering is mandatory
            (the previous interleaved ``is_target`` silently mis-assigned context).
          * the renderer only applies velocity when the render timestamp falls
            *between* keyframe timestamps; integer timestamps render the Gaussians
            unmoved. Half-integer target timestamps are what make velocity matter.

        Returns ``(views, num_keyframes)``. ``views["img"]`` is the *reordered*
        tensor — use it (not the original) as the GT so renders stay aligned.
        """
        B, T, C, H, W = images.shape
        device = images.device

        even = list(range(0, T, 2))      # keyframes (context)
        odd = list(range(1, T, 2))       # interpolation targets
        order = even + odd
        num_keyframes = len(even)

        images = images[:, torch.tensor(order, device=device)]   # [keyframes..., targets...]

        is_target = torch.tensor(
            [False] * len(even) + [True] * len(odd), device=device
        ).unsqueeze(0).expand(B, T)
        is_static = torch.zeros((B, T), dtype=torch.bool, device=device)
        # timestamp = original_position / 2  ->  keyframes 0,1,2,...  targets 0.5,1.5,...
        timestamps = torch.tensor(
            [p / 2.0 for p in order], dtype=torch.float32, device=device
        ).unsqueeze(0).expand(B, T)
        valid_mask = torch.ones((B, T, H, W), dtype=torch.bool, device=device)

        views = {
            "img": images,
            "is_target": is_target,
            "is_static": is_static,
            "timestamp": timestamps,
            "valid_mask": valid_mask,
        }
        return views, num_keyframes

    def forward(self, images: torch.Tensor):
        images = images.to(self.device, non_blocking=True)
        B, T, C, H, W = images.shape
        assert T == self.cfg.num_frames, (
            f"Expected {self.cfg.num_frames} frames, got {T}."
        )
        views, num_keyframes = self._build_views(images)
        with torch.amp.autocast(self.device, dtype=torch.bfloat16):
            predictions = self.reconstructor(views, is_inference=False, use_motion=True)
        return predictions, views, num_keyframes

    def forward_with_teacher(self, images: torch.Tensor):
        predictions, views, num_keyframes = self.forward(images)
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                teacher_predictions = self.teacher(views, is_inference=False, use_motion=True)
        return predictions, teacher_predictions, views, num_keyframes

    def render_predictions(self, predictions: dict, height: int, width: int):
        gaussians         = predictions["splats"]
        # Clone these so the renderer's in-place ops (pred_all_extrinsic[...,:3,3] *= scale)
        # don't corrupt the autograd graph built during the student forward pass.
        render_c2w        = predictions["rendered_extrinsics"].clone()
        render_intrs      = predictions["rendered_intrinsics"].clone()
        render_timestamps = predictions["rendered_timestamps"].clone()
        render_w2c        = homo_matrix_inverse(render_c2w)

        rendered_colors, rendered_depths, rendered_alphas, _ = (
            self.reconstructor.gs_renderer.rasterizer.forward(
                render_splats=gaussians,
                render_viewmats=[render_w2c[b] for b in range(render_w2c.shape[0])],
                render_Ks=[render_intrs[b] for b in range(render_intrs.shape[0])],
                render_timestamps=[render_timestamps[b] for b in range(render_timestamps.shape[0])],
                sh_degree=self.reconstructor.gs_renderer.sh_degree,
                width=width,
                height=height,
            )
        )
        return rendered_colors, rendered_depths, rendered_alphas

    def lpips_loss(self, rendered_colors: torch.Tensor, gt_images: torch.Tensor) -> torch.Tensor:
        if self.lpips_fn is None:
            return rendered_colors.new_zeros(())
        B, T = rendered_colors.shape[:2]
        flat_pred = rendered_colors.permute(0, 1, 4, 2, 3).reshape(-1, 3, rendered_colors.shape[2], rendered_colors.shape[3])
        flat_gt   = gt_images.permute(0, 1, 4, 2, 3).reshape(-1, 3, gt_images.shape[2], gt_images.shape[3])
        return self.lpips_fn(flat_pred * 2.0 - 1.0, flat_gt * 2.0 - 1.0).mean()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gaussian_background_velocity_loss(
    predictions: dict, teacher_predictions: dict, norm: str = "l1"
) -> torch.Tensor:
    """Penalty on velocities of background pixels (class 3), masked by the frozen seg head.

    norm="l1" penalises the velocity magnitude ‖v‖ — its gradient is ~constant in
    magnitude (the unit vector v/‖v‖), so it drives background velocity all the way to 0.
    norm="l2" penalises ‖v‖², whose gradient ∝ v vanishes near 0 and therefore stalls.

    Reads velocity and seg directly from the top-level predictions dict.
    """
    vel_fwd = predictions["velocity_fwd"]  # (B, T-1, H, W, D)
    vel_bwd = predictions["velocity_bwd"]  # (B, T-1, H, W, D)

    seg = teacher_predictions["seg_labels"].detach().clone()        # (B, T, H, W, 4)

    B, T, H, W, classes = seg.shape
    class_preds = seg.argmax(dim=-1)       # (B, T, H, W)
    bg_fwd = (class_preds[:, :-1] == classes - 1).unsqueeze(-1).float()  # (B, T-1, H, W, 1)
    bg_bwd = (class_preds[:,  1:] == classes - 1).unsqueeze(-1).float()

    if bg_fwd.sum() == 0:
        return vel_fwd.new_zeros(())

    if norm == "l1":
        # clamp_min before sqrt so the gradient is well-defined as v -> 0 (a plain
        # .norm() gives a NaN gradient at exactly zero, which is where we are pushing).
        mag_fwd = vel_fwd.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        mag_bwd = vel_bwd.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        loss_fwd = (mag_fwd * bg_fwd).sum() / bg_fwd.sum()
        loss_bwd = (mag_bwd * bg_bwd).sum() / bg_bwd.sum()
    else:  # "l2"
        loss_fwd = (vel_fwd.pow(2) * bg_fwd).sum() / bg_fwd.sum()
        loss_bwd = (vel_bwd.pow(2) * bg_bwd).sum() / bg_bwd.sum()
    return 0.5 * (loss_fwd + loss_bwd)


def preserve_foreground_velocity_loss(
    predictions: dict,
    teacher_predictions: dict,
) -> torch.Tensor:
    """Keep the pixel-space velocity head output close to the teacher on foreground pixels.

    L1 (‖v - tv‖₁), NOT L2: its gradient is constant in magnitude (sign(v - tv)) so it
    matches bg_vel's L1 push and does not vanish at the tiny (~0.006) velocities here.
    The L2 version had a gradient ∝ (v - tv) that faded toward zero and got steamrolled
    by bg_vel, collapsing the foreground. L1 anchors both magnitude AND direction.

    Background pixels (class 3) are excluded — those are handled by
    gaussian_background_velocity_loss above.

    Returns (combined, loss_fwd, loss_bwd) so the two directions can be logged
    separately — the backward term dying silently is exactly the failure we hit.
    """
    vel_fwd  = predictions["velocity_fwd"]        # (B, T-1, H, W, D)
    vel_bwd  = predictions["velocity_bwd"]        # (B, T-1, H, W, D)
    tvel_fwd = teacher_predictions["velocity_fwd"].detach().clone()
    tvel_bwd = teacher_predictions["velocity_bwd"].detach().clone()

    # seg_labels shape: (B, T, H, W, 4) — use all but the last / first frame
    # to align with fwd/bwd velocity time indices.
    seg = teacher_predictions["seg_labels"].detach().clone()
    class_preds = seg.argmax(dim=-1)              # (B, T, H, W)
    fg_fwd = (class_preds[:, :-1] != 3).unsqueeze(-1).float()  # (B, T-1, H, W, 1)
    fg_bwd = (class_preds[:,  1:] != 3).unsqueeze(-1).float()

    zero = vel_fwd.new_zeros(())
    # L1 (.abs()) -> constant gradient, matched to bg_vel; abs is safe at 0 (subgradient 0).
    # Guard each direction independently: if the bwd foreground mask is empty the
    # bwd head must NOT be left unconstrained (that is how it collapsed before).
    loss_fwd = ((vel_fwd - tvel_fwd).abs() * fg_fwd).sum() / fg_fwd.sum() if fg_fwd.sum() > 0 else zero
    loss_bwd = ((vel_bwd - tvel_bwd).abs() * fg_bwd).sum() / fg_bwd.sum() if fg_bwd.sum() > 0 else zero
    return 0.5 * (loss_fwd + loss_bwd), loss_fwd, loss_bwd


def velocity_balance_loss(predictions: dict, teacher_predictions: dict) -> torch.Tensor:
    """Keep forward/backward velocity *magnitudes* matched on foreground pixels.

    A symmetric motion pair (t->t+1 forward, t+1->t backward) should have similar
    magnitude; in the failed run bwd collapsed to ~0 while fwd stayed ~0.04. This
    penalises the gap between their mean foreground magnitudes. It is grid-aligned
    only in aggregate (mean over fg), so it avoids the pixel-shift issue of a
    per-pixel fwd+bwd cancellation term.
    """
    seg = teacher_predictions["seg_labels"].detach().argmax(dim=-1)   # (B, S, H, W)
    fg_f = (seg[:, :-1] != 3).float()
    fg_b = (seg[:,  1:] != 3).float()
    # Stable magnitude (clamp before sqrt): a plain .norm() has a NaN gradient at
    # exactly zero velocity, which the L1 bg_vel penalty drives the background toward.
    vf = predictions["velocity_fwd"].pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()  # (B, S-1, H, W)
    vb = predictions["velocity_bwd"].pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
    mf = (vf * fg_f).sum() / fg_f.sum().clamp(min=1.0)
    mb = (vb * fg_b).sum() / fg_b.sum().clamp(min=1.0)
    return (mf - mb).pow(2)


def camera_distillation_loss(predictions: dict, teacher_predictions: dict) -> torch.Tensor:
    """Distill camera parameters from the frozen teacher (anti-forgetting).

    Tries a priority list of camera-related keys and uses the first pair that
    both student and teacher expose.  All teacher tensors are detached.
    """
    cam_key_pairs = [
        ("rendered_extrinsics", "rendered_extrinsics"),
        ("rendered_intrinsics", "rendered_intrinsics"),
        ("camera_params",       "camera_params"),
        ("camera_poses",        "camera_poses"),
        ("camera_intrs",        "camera_intrs"),
    ]
    for sk, tk in cam_key_pairs:
        if sk in predictions and tk in teacher_predictions:
            return F.mse_loss(predictions[sk], teacher_predictions[tk].detach().clone())
    return predictions[next(iter(predictions))].new_zeros(())


def depth_distillation_loss(
    predictions: dict,
    rendered_depths: torch.Tensor,
    teacher_predictions: dict,
    teacher_rendered_depths: torch.Tensor,
) -> torch.Tensor:
    """Distill depth from the frozen teacher (anti-forgetting).

    Two complementary signals:
      1. Rendered depth (student Gaussians) vs. teacher rendered depth —
         pixel-aligned under the same camera trajectory.
      2. Any explicit depth-head output present in both dicts.

    All teacher tensors are detached.
    """
    device = rendered_depths.device if rendered_depths is not None else next(iter(predictions.values())).device
    loss = torch.tensor(0.0, device=device)

    if rendered_depths is not None and teacher_rendered_depths is not None:
        loss = loss + F.l1_loss(rendered_depths, teacher_rendered_depths.detach())

    for key in ("depth", "depthmap", "predicted_depth"):
        if key in predictions and key in teacher_predictions:
            loss = loss + F.l1_loss(predictions[key], teacher_predictions[key].detach().clone())
            break

    return loss


def segmentation_distillation_loss(
    predictions: dict,
    teacher_predictions: dict,
) -> torch.Tensor:
    """Keep segmentation logits close to the teacher (anti-forgetting)."""
    return F.mse_loss(
        predictions["seg_labels"],
        teacher_predictions["seg_labels"].detach().clone(),
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Saves run-namespaced, timestamped checkpoints.

    Each run gets a unique ``run_id`` (see TrainConfig) so different runs never
    overwrite each other. Because each checkpoint is very large, only the most
    recent rolling checkpoint *from this run* is kept on disk: when a newer one
    is written the previous one is deleted. The ``best`` checkpoint is tracked
    separately and is never pruned by the rolling logic.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._last_rolling_path: str | None = None

    def _save(self, model, optimizer, epoch, loss, path, global_step):
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "run_id": self.cfg.run_id,
                "model_state_dict": model.reconstructor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
        dbg(f"Saved checkpoint: {path}")

    def save_rolling(
        self,
        model: VelocityRegularizationModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        global_step: int,
    ):
        """Save the latest checkpoint and delete the previous one from this run.

        Named by global step so the filename is monotonic across resumes (and so a
        resumed run never collides with or deletes the checkpoint it resumed from).
        """
        path = (
            f"{self.cfg.save_model_path_prefix}_{self.cfg.run_id}"
            f"_epoch{epoch + 1}_step{global_step}.ckpt"
        )
        self._save(model, optimizer, epoch, loss, path, global_step)

        prev = self._last_rolling_path
        if prev is not None and prev != path and os.path.exists(prev):
            try:
                os.remove(prev)
                dbg(f"Removed old checkpoint: {prev}")
            except OSError as e:
                dbg(f"Could not remove old checkpoint {prev}: {e}")
        self._last_rolling_path = path

    def save_best(
        self,
        model: VelocityRegularizationModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        global_step: int,
    ):
        """Save the best checkpoint for this run (kept across the whole run)."""
        path = f"{self.cfg.save_model_path_prefix}_{self.cfg.run_id}_best.ckpt"
        self._save(model, optimizer, epoch, loss, path, global_step)


def resume_global_step(ckpt: dict, ckpt_path: str, steps_per_epoch: int) -> int:
    """Recover the global step to continue logging from.

    Prefers the ``global_step`` saved in the checkpoint; otherwise parses it from the
    filename (``_step{N}`` for new ckpts, or ``_epoch{E}_batch{N}`` for legacy ones).
    """
    if ckpt.get("global_step") is not None:
        return int(ckpt["global_step"])
    m_step = re.search(r"_step(\d+)", ckpt_path)
    if m_step:
        return int(m_step.group(1))
    m_e = re.search(r"_epoch(\d+)", ckpt_path)
    m_b = re.search(r"_batch(\d+)", ckpt_path)
    epoch = int(m_e.group(1)) if m_e else 1          # filenames store epoch+1
    batch = int(m_b.group(1)) if m_b else 0          # filenames store batch_idx+1
    return (epoch - 1) * steps_per_epoch + max(batch - 1, 0)


# ---------------------------------------------------------------------------
# In-training interpolation evaluation (mirrors interpolation_compare_demo.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_interpolation(model: "VelocityRegularizationModel", cfg: TrainConfig,
                           writer: SummaryWriter, step: int, _cache: dict = {}):
    """Run the held-out interpolation eval on the current student (and the frozen
    teacher = "original") on one fixed window, log scalars + the comparison figure to
    TensorBoard, and print the same block interpolation_compare_demo.py prints.

    The window and the teacher's metrics are cached (teacher never changes), so each
    call only re-runs the student. Eval forces gs_renderer.training=False (predicted
    cameras) and restores train mode afterwards.
    """
    if _cache.get("disabled"):
        return

    # Lazy imports so importing this module never pulls in matplotlib/the demo.
    # interpolation_compare_demo / reconstruction_compare_demo live at the repo root;
    # ensure it's importable regardless of how training was launched, and disable the
    # eval gracefully (rather than crashing training) if anything is missing.
    import sys
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import demos.interpolation_compare_demo as demo
        from demos.reconstruction_compare_demo import _ssim_fn, get_test_clips
        from demos.reconstruction_compare_demo import psnr as _psnr
        from evals.bechmark_interpolation import render_interpolated_targets
        from training.SimpleHandObjectSegmentationDataset import STREAMS
    except Exception as e:
        dbg(f"[eval] disabled — could not import demo helpers: {e}")
        _cache["disabled"] = True
        return

    H, W = cfg.eval_img_shape

    # ---- one-time setup: load the window, ssim metric ----
    if "kf" not in _cache:
        clips = get_test_clips(cfg.data_root, cfg.val_fraction)
        clip = cfg.eval_clip or clips[0]
        stream = cfg.eval_stream or STREAMS[0]
        kf, tgt, tgt_masks, kf_idx, tgt_idx = demo.load_interp_window(
            cfg.data_root, clip, stream,
            cfg.eval_num_keyframes, cfg.eval_frame_stride, cfg.eval_window_index,
        )
        _cache.update(clip=clip, stream=stream, kf=kf, tgt=tgt, masks=tgt_masks,
                      kf_idx=kf_idx, tgt_idx=tgt_idx, ssim=_ssim_fn(cfg.device))
        dbg(f"[eval] window: clip={clip} stream={stream} "
            f"keyframes={kf_idx} targets(hidden)={tgt_idx}")
    c = _cache

    def _metrics_for(net):
        pred, has_vel, vfwd, vbwd = render_interpolated_targets(
            net, c["kf"].unsqueeze(0), cfg.device, W, H, return_velocity=True
        )
        pred = pred.cpu()
        gt, masks = c["tgt"], c["masks"]
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            masks = F.interpolate(masks, size=pred.shape[-2:], mode="nearest")

        def _to_render(v):
            if v is None:
                return None
            return F.interpolate(v.unsqueeze(1), size=pred.shape[-2:],
                                 mode="bilinear", align_corners=False)[:, 0].cpu()
        vfwd_r, vbwd_r = _to_render(vfwd), _to_render(vbwd)

        V = pred.shape[0]
        per_psnr, per_bg, per_ssim = [], [], []
        vstat = {"fwd": ([], []), "bwd": ([], [])}
        for i in range(V):
            per_psnr.append(_psnr(pred[i], gt[i]))
            bg_pix = masks[i, 3] > 0.5
            per_bg.append(demo.masked_psnr(pred[i], gt[i], bg_pix[None].float()))
            if c["ssim"] is not None:
                c["ssim"].reset()
                c["ssim"].update(pred[i:i + 1].to(cfg.device), gt[i:i + 1].to(cfg.device))
                per_ssim.append(c["ssim"].compute().item())
            for dn, vr in (("fwd", vfwd_r), ("bwd", vbwd_r)):
                if vr is not None:
                    vstat[dn][0].append(demo.mean_in_region(vr[i], bg_pix))
                    vstat[dn][1].append(demo.mean_in_region(vr[i], ~bg_pix))
        res = {
            "psnr": float(np.mean(per_psnr)),
            "bg_psnr": float(np.nanmean(per_bg)),
            "ssim": float(np.mean(per_ssim)) if per_ssim else float("nan"),
        }
        for dn in ("fwd", "bwd"):
            res[f"vel_{dn}_bg"] = float(np.nanmean(vstat[dn][0])) if vstat[dn][0] else float("nan")
            res[f"vel_{dn}_fg"] = float(np.nanmean(vstat[dn][1])) if vstat[dn][1] else float("nan")
        render = {"rgb": pred, "psnr": per_psnr, "psnr_bg": per_bg,
                  "ssim": per_ssim if per_ssim else None}
        return res, render

    was_training = model.reconstructor.training
    model.reconstructor.eval()
    try:
        # Teacher ("original") is frozen — compute once and cache.
        if "teacher_res" not in c:
            c["teacher_res"], c["teacher_render"] = _metrics_for(model.teacher)
        student_res, student_render = _metrics_for(model.reconstructor)
    finally:
        if was_training:
            model.reconstructor.train()
            model.reconstructor.gs_renderer.training = False

    results = {"original": c["teacher_res"], "velocity_reg": student_res}
    renders = {"original": c["teacher_render"], "velocity_reg": student_render}

    # ---- log scalars to TB ----
    for name, res in results.items():
        for k, v in res.items():
            writer.add_scalar(f"eval/{name}/{k}", v, step)
    # Convenience deltas (student - original) on the headline metrics.
    for k in ("psnr", "bg_psnr", "ssim"):
        writer.add_scalar(f"eval/delta/{k}", student_res[k] - c["teacher_res"][k], step)

    # ---- log the comparison figure to TB ----
    try:
        fig = demo.build_interp_figure(
            c["tgt"], renders, c["tgt_idx"],
            title=f"interp eval @ step {step} — {c['clip']}/{c['stream']}",
            out_path=None, return_fig=True,
        )
        writer.add_figure("eval/interpolation_comparison", fig, step)
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        dbg(f"[eval] could not log figure: {e}")

    # ---- print the demo-style block ----
    print(f"\n=== [in-training eval @ step {step}] interpolation quality (higher=better) ===", flush=True)
    for name, s in results.items():
        print(f"  {name:>14}:  PSNR {s['psnr']:.3f}   bg-PSNR {s['bg_psnr']:.3f}   SSIM {s['ssim']:.4f}", flush=True)
    print("  --- predicted velocity magnitude (background should be ~0) ---", flush=True)
    for name, s in results.items():
        for d in ("fwd", "bwd"):
            vb, vf = s[f"vel_{d}_bg"], s[f"vel_{d}_fg"]
            ratio = f"  fg/bg {vf / vb:.1f}x" if vb and vb > 0 else ""
            print(f"      {name:>12} {d}:  bg {vb:.4f}   fg {vf:.4f}{ratio}", flush=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    dataset = SequenceHandObjectDataset(
        data_root=cfg.data_root,
        num_key_frames=cfg.num_key_frames,
        frame_stride=cfg.frame_stride,
        random_reverse=cfg.random_reverse,
        streams=cfg.streams,
        clip_names=cfg.clip_names,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    model = VelocityRegularizationModel(cfg)

    # Two param groups: velocity heads at learning_rate, any unfrozen backbone at the
    # (lower) backbone_learning_rate. AdamW keeps per-group LRs; the scheduler scales
    # each group from its own base_lr.
    head_params, backbone_params = [], []
    for name, p in model.reconstructor.named_parameters():
        if not p.requires_grad:
            continue
        if "velocity_fwd_head" in name or "velocity_bwd_head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = [{"params": head_params, "lr": cfg.learning_rate}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_learning_rate})
    dbg(f"Optimizer: {len(head_params)} head params @ {cfg.learning_rate}"
        + (f", {len(backbone_params)} backbone params @ {cfg.backbone_learning_rate}"
           if backbone_params else " (backbone frozen)"))
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    total_steps = len(loader) * cfg.epochs
    if cfg.warmup_steps > 0 and cfg.warmup_steps < total_steps:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=cfg.warmup_steps
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - cfg.warmup_steps, 1),
            eta_min=cfg.learning_rate * cfg.lr_min_factor,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[cfg.warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=cfg.learning_rate * cfg.lr_min_factor,
        )

    # ---- resume optimizer state + global step (model weights already loaded) ----
    start_step = 0
    if model.resume_ckpt is not None:
        try:
            optimizer.load_state_dict(model.resume_ckpt["optimizer_state_dict"])
            dbg("Resumed optimizer state.")
        except Exception as e:
            dbg(f"Could not resume optimizer state ({e}); continuing with a fresh optimizer.")
        start_step = resume_global_step(model.resume_ckpt, cfg.resume_checkpoint, len(loader))
        dbg(f"Resuming logging from global step {start_step} (LR schedule restarts fresh).")

    writer = SummaryWriter(cfg.log_dir)
    ckpt_mgr = CheckpointManager(cfg)
    dbg(f"Training on {len(dataset)} sequences, {len(loader)} batches/epoch.")
    dbg(f"Run id: {cfg.run_id}")
    best_loss = float("inf")
    # Anomaly detection makes any NaN in backward fatal and ~2x slower; the
    # non-finite-grad guard below handles NaNs gracefully instead. Re-enable for debugging.
    # torch.autograd.set_detect_anomaly(True)

    # Baseline eval at the resume point so the TB curves start where you left off.
    if cfg.eval_interval_batches > 0:
        try:
            evaluate_interpolation(model, cfg, writer, start_step)
        except Exception as e:
            dbg(f"[eval] baseline eval failed ({e}); disabling in-training eval.")
            cfg.eval_interval_batches = 0
    for epoch in range(cfg.epochs):
        model.reconstructor.train()
        # nn.Module.train() flips gs_renderer.training back to True, which routes
        # render() into the GT-camera branch (prepare_cameras reads views["camera_poses"],
        # which we never provide -> KeyError). We have no GT cameras, so force the
        # predicted-camera branch. This must run AFTER .train() every epoch.
        model.reconstructor.gs_renderer.training = False
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, _gt_masks) in enumerate(loader):
            # _gt_masks are available for inspection / future use but are not
            # used as training signal here — we rely on the model's own
            # segmentation predictions so the loss stays self-consistent.
            optimizer.zero_grad()

            # ---- forward passes ----------------------------------------
            predictions, teacher_predictions, views, num_keyframes = (
                model.forward_with_teacher(images)
            )

            # Student render. Renders all S+V frames; the target frames sit at
            # fractional timestamps, so they are produced by velocity-transitioning
            # the keyframe Gaussians — that is where the velocity gradient comes from.
            H, W = images.shape[-2], images.shape[-1]
            rendered_colors, rendered_depths, rendered_alphas = model.render_predictions(
                predictions, height=H, width=W
            )

            # Teacher render — completely outside the autograd graph.
            # detach_predictions() ensures no tensor aliasing leaks gradients.
            with torch.no_grad():
                # Deep-clone all tensors to break any storage sharing with the student graph
                safe_teacher = {
                    k: v.clone() if isinstance(v, torch.Tensor) else
                    {sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                        for sk, sv in v.items()} if isinstance(v, dict) else v
                    for k, v in teacher_predictions.items()
                }
                _, teacher_rendered_depths, _ = model.render_predictions(
                    safe_teacher, height=H, width=W
                )
            # ---- true-GT losses (input video) --------------------------
            

            # GT must be the REORDERED frames (views["img"]) so they line up with
            # rendered_colors (which follows the [keyframes..., targets...] order).
            gt_images = views["img"].permute(0, 1, 3, 4, 2)             # (B,T,H,W,3)
            S = num_keyframes
            kf_pred,  kf_gt  = rendered_colors[:, :S], gt_images[:, :S]  # keyframes (integer ts)
            tgt_pred, tgt_gt = rendered_colors[:, S:], gt_images[:, S:]  # held-out in-between

            # Lrgb (keyframe reconstruction): rendered at integer timestamps where
            # velocity is the identity, so it carries NO velocity gradient (and trains
            # nothing while the backbone is frozen). Kept purely for monitoring.
            rgb_l2_loss = F.mse_loss(kf_pred, kf_gt)
            rgb_loss = rgb_l2_loss

            # Linterp: the held-out frames are rendered at half-integer timestamps, so
            # the renderer transitions keyframe Gaussians along velocity to reach them.
            # This is the only term that backprops real data into the velocity heads.
            interp_l2_loss = F.mse_loss(tgt_pred, tgt_gt)
            interp_lpips = (
                model.lpips_loss(tgt_pred, tgt_gt)
                if cfg.lpips_loss_weight > 0
                else tgt_pred.new_zeros(())
            )
            interp_loss = interp_l2_loss + cfg.lpips_loss_weight * interp_lpips
            lpips_loss = interp_lpips  # logged below

            # Lregular: prevent Gaussians from going transparent
            reg_loss = F.l1_loss(rendered_alphas, torch.ones_like(rendered_alphas))

            # ---- core goal: background Gaussian velocity → 0 -----------
            # This operates on per-Gaussian attributes inside predictions["splats"].
            # If your WorldMirror version does not expose per-Gaussian seg_labels /
            # velocity_fwd / velocity_bwd inside the splats dict, replace this call
            # with the appropriate access pattern.
            bg_vel_loss = gaussian_background_velocity_loss(
                predictions, teacher_predictions, norm=cfg.bg_vel_loss_norm
            )

            # ---- distillation losses (anti-forgetting) -----------------

            # Keep camera predictions stable
            camera_loss = camera_distillation_loss(predictions, teacher_predictions)

            # Keep depth predictions stable (rendered + head)
            depth_loss = depth_distillation_loss(
                predictions, rendered_depths, teacher_predictions, teacher_rendered_depths
            )

            # Keep foreground pixel-velocity head stable (both directions)
            preserve_fg_loss, preserve_fg_fwd, preserve_fg_bwd = (
                preserve_foreground_velocity_loss(predictions, teacher_predictions)
            )

            # Keep forward/backward velocity magnitudes balanced (anti-collapse)
            balance_loss = velocity_balance_loss(predictions, teacher_predictions)

            # Keep segmentation head stable
            seg_loss = segmentation_distillation_loss(predictions, teacher_predictions)

            # ---- total loss --------------------------------------------
            loss = (
                cfg.rgb_loss_weight            * rgb_loss
                + cfg.interp_loss_weight       * interp_loss
                + cfg.regularization_weight    * reg_loss
                + cfg.bg_gaussian_vel_weight   * bg_vel_loss
                + cfg.camera_distill_weight    * camera_loss
                + cfg.depth_distill_weight     * depth_loss
                + cfg.preserve_foreground_vel_weight * preserve_fg_loss
                + cfg.velocity_balance_weight        * balance_loss
                + cfg.preserve_segmentation_weight   * seg_loss
            )

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.reconstructor.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )
            # Guard: a non-finite gradient (e.g. the magnitude of a velocity driven to
            # exactly zero) would poison every weight via optimizer.step(). Skip the
            # update for that batch instead of crashing / corrupting the model.
            if not torch.isfinite(total_norm):
                dbg(f"Non-finite grad norm at epoch {epoch+1} batch {batch_idx}; skipping update.")
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            # Offset by start_step so a resumed run continues the same TB step axis.
            step = start_step + epoch * len(loader) + batch_idx

            # ---- logging -----------------------------------------------
            writer.add_scalar("train/loss_step",          loss_val,                step)
            writer.add_scalar("train/lr",                 scheduler.get_last_lr()[0], step)
            writer.add_scalar("train/rgb_l2",             rgb_l2_loss.item(),      step)
            writer.add_scalar("train/lpips",              lpips_loss.item(),        step)
            writer.add_scalar("train/rgb_loss",           rgb_loss.item(),          step)
            writer.add_scalar("train/interp_l2",          interp_l2_loss.item(),    step)
            writer.add_scalar("train/interp_loss",        interp_loss.item(),       step)
            writer.add_scalar("train/alpha_reg",          reg_loss.item(),          step)
            writer.add_scalar("train/bg_gaussian_vel",    bg_vel_loss.item(),       step)
            writer.add_scalar("train/camera_distill",     camera_loss.item(),       step)
            writer.add_scalar("train/depth_distill",      depth_loss.item(),        step)
            writer.add_scalar("train/preserve_fg_vel",    preserve_fg_loss.item(),  step)
            writer.add_scalar("train/preserve_fg_fwd",    preserve_fg_fwd.item(),   step)
            writer.add_scalar("train/preserve_fg_bwd",    preserve_fg_bwd.item(),   step)
            writer.add_scalar("train/velocity_balance",   balance_loss.item(),      step)
            writer.add_scalar("train/seg_distill",        seg_loss.item(),          step)

            # Raw foreground velocity magnitude per direction — the direct collapse
            # signal. If vel_fg_bwd_mag trends to ~0 while vel_fg_fwd_mag stays up,
            # the backward head is dying again; stop and revisit.
            with torch.no_grad():
                seg_dbg = teacher_predictions["seg_labels"].argmax(dim=-1)
                fgf = (seg_dbg[:, :-1] != 3).float()
                fgb = (seg_dbg[:,  1:] != 3).float()
                bgf, bgb = 1.0 - fgf, 1.0 - fgb
                vmag_f = predictions["velocity_fwd"].norm(dim=-1)
                vmag_b = predictions["velocity_bwd"].norm(dim=-1)
                vfm = (vmag_f * fgf).sum() / fgf.sum().clamp(min=1.0)
                vbm = (vmag_b * fgb).sum() / fgb.sum().clamp(min=1.0)
                # Background magnitudes — the objective. Watch these trend toward 0.
                vfm_bg = (vmag_f * bgf).sum() / bgf.sum().clamp(min=1.0)
                vbm_bg = (vmag_b * bgb).sum() / bgb.sum().clamp(min=1.0)
                # The preserve_fg TARGET: the teacher's foreground velocity on the
                # is_inference=False (training) path. If this is ~0 (≪ the 0.028 the
                # teacher shows on the eval path), preserve_fg has no real floor and
                # cannot stop the foreground from collapsing — that would explain why
                # both train and eval fg collapsed despite weight 10.
                tvmag_f = teacher_predictions["velocity_fwd"].norm(dim=-1)
                tvmag_b = teacher_predictions["velocity_bwd"].norm(dim=-1)
                tvfm = (tvmag_f * fgf).sum() / fgf.sum().clamp(min=1.0)
                tvbm = (tvmag_b * fgb).sum() / fgb.sum().clamp(min=1.0)
            writer.add_scalar("train/vel_fg_fwd_mag",     vfm.item(),               step)
            writer.add_scalar("train/vel_fg_bwd_mag",     vbm.item(),               step)
            writer.add_scalar("train/vel_bg_fwd_mag",     vfm_bg.item(),            step)
            writer.add_scalar("train/vel_bg_bwd_mag",     vbm_bg.item(),            step)
            writer.add_scalar("train/teacher_vel_fg_fwd_mag", tvfm.item(),          step)
            writer.add_scalar("train/teacher_vel_fg_bwd_mag", tvbm.item(),          step)

            if batch_idx % 10 == 0:
                dbg(
                    f"Epoch {epoch+1}/{cfg.epochs}  batch {batch_idx}/{len(loader)}  "
                    f"loss={loss_val:.5f}  interp={interp_loss.item():.5f}  "
                    f"rgb={rgb_loss.item():.5f}  bg_vel={bg_vel_loss.item():.5f}  "
                    f"cam={camera_loss.item():.5f}  depth={depth_loss.item():.5f}"
                )

            # In-training interpolation eval -> TB scalars + figure + printout.
            if cfg.eval_interval_batches > 0 and (batch_idx + 1) % cfg.eval_interval_batches == 0:
                try:
                    evaluate_interpolation(model, cfg, writer, step)
                except Exception as e:
                    dbg(f"[eval] failed ({e}); disabling in-training eval.")
                    cfg.eval_interval_batches = 0

            if cfg.checkpoint_interval_batches > 0 and (batch_idx + 1) % cfg.checkpoint_interval_batches == 0:
                ckpt_mgr.save_rolling(model, optimizer, epoch, loss_val, global_step=step)

        avg_loss = epoch_loss / max(len(loader), 1)
        elapsed  = time.time() - t0
        dbg(f"Epoch {epoch+1}/{cfg.epochs}  avg_loss={avg_loss:.5f}  ({elapsed:.0f}s)")
        writer.add_scalar("train/loss_epoch", avg_loss, start_step + (epoch + 1) * len(loader))

        epoch_end_step = start_step + (epoch + 1) * len(loader) - 1
        ckpt_mgr.save_rolling(model, optimizer, epoch, avg_loss, global_step=epoch_end_step)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_mgr.save_best(model, optimizer, epoch, avg_loss, global_step=epoch_end_step)
            dbg(f"New best loss: {best_loss:.5f}")

    writer.close()
    dbg("Training finished.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = TrainConfig()
    os.makedirs(cfg.save_model_path_prefix.rsplit("/", 1)[0], exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    train(cfg)



if __name__ == "__main__":
    main()