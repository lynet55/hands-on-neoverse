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
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager
from diffsynth.utils.auxiliary import homo_matrix_inverse
from diffsynth.data.SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset

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
        data_root: str = "diffsynth/data/training_data_modal",
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
    frame_stride: int = 1
    random_reverse: bool = True

    # Which backbone submodules to unfreeze alongside the velocity heads.
    # Keeping this list narrow limits the risk of catastrophic forgetting.
    backbone_unfreeze_substrings: list[str] = field(
        default_factory=lambda: [
            "visual_geometry_transformer.motion_fwd_blocks",
            "visual_geometry_transformer.motion_bwd_blocks",
            "visual_geometry_transformer.frame_blocks",
        ]
    )

    # Model paths
    reconstruction_model_path: str = "models/NeoVerse/reconstructor.ckpt"
    hand_seg_model_path: str = "models/NeoVerse/hand_seg_model_opt_best.ckpt"
    save_model_path_prefix: str = "models/NeoVerse/velocity_regularization"

    # When True, freeze everything except velocity heads + backbone_unfreeze_substrings.
    # When False, all parameters are trainable (use with a low learning rate).
    freeze_except_velocity: bool = True

    # Training hyperparameters
    batch_size: int = 1
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    # Loss weights
    # -- true-GT losses (input video is the only GT we have) --
    rgb_loss_weight: float = 1.0       # Lrgb: L2 + lpips_loss_weight * LPIPS
    lpips_loss_weight: float = 0.1     # weight of LPIPS inside Lrgb
    regularization_weight: float = 0.1 # Lregular: alpha coverage

    # -- core goal: push background Gaussian velocities to zero --
    bg_gaussian_vel_weight: float = 1.0

    # -- distillation losses: prevent catastrophic forgetting --
    # These use the frozen teacher as pseudo-GT; they carry no new information
    # but stabilise all predictions that we are not deliberately changing.
    camera_distill_weight: float = 5.0
    depth_distill_weight: float = 1.0
    preserve_foreground_vel_weight: float = 1.0  # pixel-vel head, fg pixels only
    preserve_segmentation_weight: float = 1.0

    # LR schedule
    warmup_steps: int = 500
    lr_min_factor: float = 0.1

    # Checkpointing
    checkpoint_interval_batches: int = 100

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_dir: str = field(
        default_factory=lambda: (
            "runs/velocity_regularization_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    num_workers: int = 2
    pin_memory: bool = True

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
            n_train, n_total = 0, 0
            for name, param in self.reconstructor.named_parameters():
                n_total += 1
                is_vel = "velocity_fwd_head" in name or "velocity_bwd_head" in name
                is_backbone = any(sub in name for sub in cfg.backbone_unfreeze_substrings)
                if is_vel or is_backbone:
                    param.requires_grad = True
                    n_train += 1
            dbg(f"Trainable parameters: {n_train}/{n_total}")
            if n_train == 0:
                raise RuntimeError(
                    "No parameters matched velocity heads or backbone_unfreeze_substrings. "
                    "Check TrainConfig.backbone_unfreeze_substrings."
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

        self.device = cfg.device

        self.lpips_fn = None
        if lpips_lib is not None:
            try:
                self.lpips_fn = lpips_lib.LPIPS(net="vgg").to(self.device)
            except Exception:
                self.lpips_fn = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_views(self, images: torch.Tensor) -> dict:
        B, T, C, H, W = images.shape
        device = images.device

        is_target = (
            (torch.arange(T, device=device) % 2 == 1)
            .unsqueeze(0)
            .expand(B, T)
        )

        is_static = torch.zeros((B, T), dtype=torch.bool, device=device)

        timestamps = (
            torch.arange(T, device=device)
            .unsqueeze(0)
            .expand(B, T)
        )

        # NEW
        valid_mask = torch.ones(
            (B, T, H, W),
            dtype=torch.bool,
            device=device
        )

        return {
            "img": images,
            "is_target": is_target,
            "is_static": is_static,
            "timestamp": timestamps,
            "valid_mask": valid_mask,
        }

    def forward(self, images: torch.Tensor):
        images = images.to(self.device, non_blocking=True)
        B, T, C, H, W = images.shape
        assert T == self.cfg.num_frames, (
            f"Expected {self.cfg.num_frames} frames, got {T}."
        )
        views = self._build_views(images)
        with torch.amp.autocast(self.device, dtype=torch.bfloat16):
            predictions = self.reconstructor(views, is_inference=False, use_motion=True)
        return predictions, views

    def forward_with_teacher(self, images: torch.Tensor):
        predictions, views = self.forward(images)
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                teacher_predictions = self.teacher(views, is_inference=False, use_motion=True)
        return predictions, teacher_predictions

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

def gaussian_background_velocity_loss(predictions: dict, teacher_predictions: dict) -> torch.Tensor:
    """L2 norm penalty on velocities of background pixels (class 3).

    Mirrors preserve_foreground_velocity_loss: reads velocity and seg directly
    from the top-level predictions dict instead of per-Gaussian splat attributes.
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

    loss_fwd = (vel_fwd.pow(2) * bg_fwd).sum() / bg_fwd.sum()
    loss_bwd = (vel_bwd.pow(2) * bg_bwd).sum() / bg_bwd.sum()
    return 0.5 * (loss_fwd + loss_bwd)


def preserve_foreground_velocity_loss(
    predictions: dict,
    teacher_predictions: dict,
) -> torch.Tensor:
    """Keep the pixel-space velocity head output close to the teacher on foreground pixels.

    Uses the student's own segmentation prediction as the foreground mask so
    that the mask is consistent with what the student currently believes.
    Background pixels (class 3) are excluded — those are handled by
    gaussian_background_velocity_loss above.
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

    if fg_fwd.sum() == 0:
        return vel_fwd.new_zeros(())

    loss_fwd = ((vel_fwd - tvel_fwd).pow(2) * fg_fwd).sum() / fg_fwd.sum()
    loss_bwd = ((vel_bwd - tvel_bwd).pow(2) * fg_bwd).sum() / fg_bwd.sum()
    return 0.5 * (loss_fwd + loss_bwd)


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

def save_checkpoint(
    model: VelocityRegularizationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    cfg: TrainConfig,
    suffix: str = "",
):
    path = f"{cfg.save_model_path_prefix}_epoch{epoch + 1}{suffix}.ckpt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.reconstructor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    dbg(f"Saved checkpoint: {path}")


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
    optimizer = torch.optim.AdamW(
        [p for p in model.reconstructor.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

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

    writer = SummaryWriter(cfg.log_dir)
    dbg(f"Training on {len(dataset)} sequences, {len(loader)} batches/epoch.")
    best_loss = float("inf")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(cfg.epochs):
        model.reconstructor.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, _gt_masks) in enumerate(loader):
            # _gt_masks are available for inspection / future use but are not
            # used as training signal here — we rely on the model's own
            # segmentation predictions so the loss stays self-consistent.
            optimizer.zero_grad()

            # ---- forward passes ----------------------------------------
            predictions, teacher_predictions = model.forward_with_teacher(images)

            # Student render
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
            

            # Lrgb: photometric reconstruction
            gt_images   = images.to(model.device).permute(0, 1, 3, 4, 2)  # (B,T,H,W,3)
            rgb_l2_loss = F.mse_loss(rendered_colors, gt_images)
            lpips_loss  = (
                model.lpips_loss(rendered_colors, gt_images)
                if cfg.lpips_loss_weight > 0
                else rendered_colors.new_zeros(())
            )
            rgb_loss = rgb_l2_loss + cfg.lpips_loss_weight * lpips_loss

            # Lregular: prevent Gaussians from going transparent
            reg_loss = F.l1_loss(rendered_alphas, torch.ones_like(rendered_alphas))

            # ---- core goal: background Gaussian velocity → 0 -----------
            # This operates on per-Gaussian attributes inside predictions["splats"].
            # If your WorldMirror version does not expose per-Gaussian seg_labels /
            # velocity_fwd / velocity_bwd inside the splats dict, replace this call
            # with the appropriate access pattern.
            bg_vel_loss = gaussian_background_velocity_loss(predictions, teacher_predictions)

            # ---- distillation losses (anti-forgetting) -----------------

            # Keep camera predictions stable
            camera_loss = camera_distillation_loss(predictions, teacher_predictions)

            # Keep depth predictions stable (rendered + head)
            depth_loss = depth_distillation_loss(
                predictions, rendered_depths, teacher_predictions, teacher_rendered_depths
            )

            # Keep foreground pixel-velocity head stable
            preserve_fg_loss = preserve_foreground_velocity_loss(predictions, teacher_predictions)

            # Keep segmentation head stable
            seg_loss = segmentation_distillation_loss(predictions, teacher_predictions)

            # ---- total loss --------------------------------------------
            loss = (
                cfg.rgb_loss_weight            * rgb_loss
                + cfg.regularization_weight    * reg_loss
                + cfg.bg_gaussian_vel_weight   * bg_vel_loss
                + cfg.camera_distill_weight    * camera_loss
                + cfg.depth_distill_weight     * depth_loss
                + cfg.preserve_foreground_vel_weight * preserve_fg_loss
                + cfg.preserve_segmentation_weight   * seg_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.reconstructor.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            step = epoch * len(loader) + batch_idx

            # ---- logging -----------------------------------------------
            writer.add_scalar("train/loss_step",          loss_val,                step)
            writer.add_scalar("train/lr",                 scheduler.get_last_lr()[0], step)
            writer.add_scalar("train/rgb_l2",             rgb_l2_loss.item(),      step)
            writer.add_scalar("train/lpips",              lpips_loss.item(),        step)
            writer.add_scalar("train/rgb_loss",           rgb_loss.item(),          step)
            writer.add_scalar("train/alpha_reg",          reg_loss.item(),          step)
            writer.add_scalar("train/bg_gaussian_vel",    bg_vel_loss.item(),       step)
            writer.add_scalar("train/camera_distill",     camera_loss.item(),       step)
            writer.add_scalar("train/depth_distill",      depth_loss.item(),        step)
            writer.add_scalar("train/preserve_fg_vel",    preserve_fg_loss.item(),  step)
            writer.add_scalar("train/seg_distill",        seg_loss.item(),          step)

            if batch_idx % 10 == 0:
                dbg(
                    f"Epoch {epoch+1}/{cfg.epochs}  batch {batch_idx}/{len(loader)}  "
                    f"loss={loss_val:.5f}  rgb={rgb_loss.item():.5f}  "
                    f"bg_vel={bg_vel_loss.item():.5f}  "
                    f"cam={camera_loss.item():.5f}  depth={depth_loss.item():.5f}"
                )

            if cfg.checkpoint_interval_batches > 0 and (batch_idx + 1) % cfg.checkpoint_interval_batches == 0:
                print("inside checkpoint saving")
                save_checkpoint(model, optimizer, epoch, loss_val, cfg)

        avg_loss = epoch_loss / max(len(loader), 1)
        elapsed  = time.time() - t0
        dbg(f"Epoch {epoch+1}/{cfg.epochs}  avg_loss={avg_loss:.5f}  ({elapsed:.0f}s)")
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        save_checkpoint(model, optimizer, epoch, avg_loss, cfg)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.reconstructor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"{cfg.save_model_path_prefix}_best.ckpt",
            )
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