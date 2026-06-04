"""Regularize velocity predictions for background pixels.

This script trains only the forward/backward velocity heads in the WorldMirror
reconstructor. It uses the model's own segmentation prediction to decide where
background is predicted (class 3) and applies an L2 penalty to the predicted
forward/backward velocities at those pixels. Classes 0-2 do not contribute to
this regularization loss.

The training loop is intentionally lightweight: the backbone and classification
head are frozen, and only velocity head parameters are updated.
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
from diffsynth.data.SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset


class SequenceHandObjectDataset(HandObjectSegmentationDataset):
    """Dataset that returns fixed-length sequences of consecutive frames."""

    def __init__(
        self,
        data_root: str = "diffsynth/data/training_data",
        num_frames: int = 2,
        frame_stride: int = 1,
        streams=None,
        clip_names=None,
    ):
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self._clip_names_filter = set(clip_names) if clip_names is not None else None
        super().__init__(data_root=data_root, streams=streams)

    def _build_index(self) -> None:
        self.samples = []
        for npz_path in sorted(Path(self.data_root).glob("clip-*.npz")):
            clip_name = npz_path.stem
            if self._clip_names_filter is not None and clip_name not in self._clip_names_filter:
                continue
            npz = np.load(str(npz_path), mmap_mode="r")
            n_frames = next(
                npz[k].shape[0] for k in npz.files if k.startswith("images_")
            )
            for stream in self.streams:
                if f"images_{stream}" not in npz.files:
                    continue
                max_start = n_frames - (self.num_frames - 1) * self.frame_stride
                if max_start <= 0:
                    continue
                for frame_idx in range(0, max_start):
                    self.samples.append(
                        {
                            "clip_name": clip_name,
                            "stream": stream,
                            "frame_idx": frame_idx,
                        }
                    )

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        clip_name, stream, frame_idx = s["clip_name"], s["stream"], s["frame_idx"]
        npz = np.load(str(self.data_root / f"{clip_name}.npz"), mmap_mode="r")

        indices = [frame_idx + i * self.frame_stride for i in range(self.num_frames)]
        images = []
        for idx_frame in indices:
            image = torch.tensor(
                npz[f"images_{stream}"][idx_frame], dtype=torch.float32
            ).permute(2, 0, 1) / 255.0
            images.append(image)

        return torch.stack(images, dim=0)


@dataclass
class TrainConfig:
    # Data
    data_root: str = "diffsynth/data/training_data"
    streams: list[str] = None
    clip_names: list[str] = None
    num_frames: int = 2
    frame_stride: int = 1
    backbone_unfreeze_substrings: list[str] = field(
        default_factory=lambda: [
            "visual_geometry_transformer.motion_fwd_blocks",
            "visual_geometry_transformer.motion_bwd_blocks",
            "visual_geometry_transformer.frame_blocks",
        ]
    )

    # Model
    reconstruction_model_path: str = "models/NeoVerse/reconstructor.ckpt"
    # The segmentation (hand_pred_head) weights are NOT stored in reconstructor.ckpt;
    # that head is freshly initialized when the model is built. It is trained separately
    # (training_25.py / training_optimized.py) and must be loaded on top of the reconstructor,
    # exactly like benchmark.py and eval_segmentation.py do. Without it, the background mask
    # used by the velocity regularization comes from a random head.
    hand_head_path: str = "models/NeoVerse/hand_seg_model_opt_run20260507-224244_epoch005.ckpt"
    save_model_path_prefix: str = "models/NeoVerse/velocity_regularization"
    low_vram: bool = False

    # Training
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    preserve_foreground_weight: float = 1.0
    preserve_segmentation_weight: float = 1.0
    checkpoint_interval_batches: int = 100

    # Environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_dir: str = field(
        default_factory=lambda: "runs/velocity_regularization_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    num_workers: int = 2
    pin_memory: bool = True


def dbg(msg: str):
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        dbg("Reconstructor loaded.")

        # Load the separately-trained segmentation head. The background mask that drives
        # the velocity regularization (class_preds == 3) is only meaningful with these weights.
        self._load_hand_head(cfg.hand_head_path)

        # Teacher is deep-copied AFTER the head is loaded so its reference predictions
        # (seg_logits_ref, vel_*_ref) come from the trained head, not the random init.
        self.teacher: WorldMirror = copy.deepcopy(self.reconstructor).eval()
        self.teacher.requires_grad_(False)

        # Freeze current model parameters, then unfreeze velocity heads and selected backbone blocks.
        self.reconstructor.requires_grad_(False)
        n_train, n_total = 0, 0
        for name, param in self.reconstructor.named_parameters():
            n_total += 1
            if "velocity_fwd_head" in name or "velocity_bwd_head" in name:
                param.requires_grad = True
                n_train += 1
            elif any(sub in name for sub in cfg.backbone_unfreeze_substrings):
                param.requires_grad = True
                n_train += 1
        dbg(f"Trainable parameters: {n_train}/{n_total}")

        if n_train == 0:
            raise RuntimeError("No parameters matched velocity heads or backbone unfreeze substrings.")

        self.reconstructor.eval()
        self.device = cfg.device

    def _load_hand_head(self, hand_head_path: str):
        """Load the separately-trained hand_pred_head weights onto the reconstructor.

        Mirrors benchmark.py / eval_segmentation.py: the checkpoint may be a raw head
        state_dict or wrapped under "model_state_dict", and its keys may or may not be
        prefixed with "hand_pred_head.".
        """
        dbg(f"Loading hand head from {hand_head_path} ...")
        ckpt = torch.load(hand_head_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not any(k.startswith("hand_pred_head.") for k in sd.keys()):
            sd = {f"hand_pred_head.{k}": v for k, v in sd.items()}
        else:
            sd = {k: v for k, v in sd.items() if k.startswith("hand_pred_head.")}
        missing, unexpected = self.reconstructor.load_state_dict(sd, strict=False)
        head_missing = [k for k in missing if "hand_pred_head" in k]
        if head_missing:
            raise RuntimeError(
                f"{len(head_missing)} hand_pred_head keys missing from {hand_head_path}, "
                f"e.g. {head_missing[:5]}. The background mask would come from a random head."
            )
        # The head was trained in fp32, but load_state_dict casts the fp32 weights into this
        # module's bf16 params, leaving the head in bf16 — which matches the bf16 backbone tokens
        # in forward(). benchmark.py can keep the head in fp32 because it wraps the whole forward
        # in bf16 autocast; this script's manual forward has no autocast, so a fp32 head would
        # raise a dtype mismatch. The head only feeds an argmax background mask, so bf16 is fine.
        dbg(f"Hand head loaded ({len(sd)} tensors).")

    def forward(self, images: torch.Tensor):
        """Return predicted segmentation logits + velocity predictions."""
        images = images.to(self.device, non_blocking=True)
        B, S, C, H, W = images.shape
        assert S >= 2, "Velocity regularization requires at least 2 frames."

        # Current model forward pass. The trainable backbone blocks here will receive gradients
        # from the velocity and segmentation consistency losses.
        token_list, patch_start_idx, fwd_token_list, bwd_token_list = self.reconstructor.visual_geometry_transformer(
            images, use_motion=True
        )
        seg_logits, _ = self.reconstructor.hand_pred_head(
            token_list, images=images, patch_start_idx=patch_start_idx
        )
        vel_fwd, _ = self.reconstructor.velocity_fwd_head(
            fwd_token_list,
            images=images[:, :-1],
            patch_start_idx=patch_start_idx,
        )
        vel_bwd, _ = self.reconstructor.velocity_bwd_head(
            bwd_token_list,
            images=images[:, 1:],
            patch_start_idx=patch_start_idx,
        )

        with torch.no_grad():
            teacher_token_list, teacher_patch_start_idx, teacher_fwd_token_list, teacher_bwd_token_list = self.teacher.visual_geometry_transformer(
                images, use_motion=True
            )
            seg_logits_ref, _ = self.teacher.hand_pred_head(
                teacher_token_list, images=images, patch_start_idx=teacher_patch_start_idx
            )
            vel_fwd_ref, _ = self.teacher.velocity_fwd_head(
                teacher_fwd_token_list,
                images=images[:, :-1],
                patch_start_idx=teacher_patch_start_idx,
            )
            vel_bwd_ref, _ = self.teacher.velocity_bwd_head(
                teacher_bwd_token_list,
                images=images[:, 1:],
                patch_start_idx=teacher_patch_start_idx,
            )

        return seg_logits, vel_fwd, vel_bwd, seg_logits_ref, vel_fwd_ref, vel_bwd_ref


def masked_background_velocity_loss(velocity: torch.Tensor, class_labels: torch.Tensor):
    """Compute L2 loss only for pixels predicted as background (class 3)."""
    mask = (class_labels == 3).unsqueeze(-1).float()
    if mask.sum() == 0:
        return velocity.new_zeros(())
    return (velocity.pow(2) * mask).sum() / mask.sum()


def preserve_foreground_velocity_loss(
    velocity: torch.Tensor,
    reference: torch.Tensor,
    class_labels: torch.Tensor,
):
    """Keep velocity for classes 0-2 close to the pretrained reference."""
    mask = (class_labels != 3).unsqueeze(-1).float()
    if mask.sum() == 0:
        return velocity.new_zeros(())
    return ((velocity - reference).pow(2) * mask).sum() / mask.sum()


def segmentation_consistency_loss(pred_logits: torch.Tensor, ref_logits: torch.Tensor):
    """Preserve segmentation predictions when partially unfreezing the backbone."""
    return F.mse_loss(pred_logits, ref_logits)


def save_checkpoint(model: VelocityRegularizationModel, optimizer, epoch: int, loss: float, cfg: TrainConfig, tag: str):
    # `tag` makes every checkpoint uniquely named (e.g. "epoch3_batch200", "epoch3", "best")
    # so periodic intra-epoch saves and end-of-epoch saves no longer overwrite each other.
    path = f"{cfg.save_model_path_prefix}_{tag}.ckpt"
    # Save the model weights as a FLAT state_dict so ModelManager can detect the
    # WorldMirror reconstructor by its key hash. Wrapping the weights under
    # "model_state_dict" hides them one level down and breaks model-type detection.
    torch.save(model.reconstructor.state_dict(), path)
    # Training metadata (for resuming) goes into a sidecar file.
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{path}.trainstate",
    )
    dbg(f"Saved checkpoint: {path}")


def train(cfg: TrainConfig):
    dataset = SequenceHandObjectDataset(
        data_root=cfg.data_root,
        num_frames=cfg.num_frames,
        frame_stride=cfg.frame_stride,
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
    optimizer = cfg.optimizer(
        [p for p in model.reconstructor.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    ) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * cfg.epochs, eta_min=cfg.learning_rate * 0.1
    )
    writer = SummaryWriter(cfg.log_dir)

    dbg(f"Training on {len(dataset)} sequences, {len(loader)} batches per epoch.")
    best_loss = float("inf")

    for epoch in range(cfg.epochs):
        model.reconstructor.train()
        epoch_loss = 0.0
        step = 0
        start_time = time.time()

        for batch_idx, images in enumerate(loader):
            optimizer.zero_grad()
            seg_logits, vel_fwd, vel_bwd, seg_logits_ref, vel_fwd_ref, vel_bwd_ref = model.forward(images)

            class_preds = seg_logits.argmax(dim=-1)
            loss_fwd = masked_background_velocity_loss(vel_fwd, class_preds[:, :-1])
            loss_bwd = masked_background_velocity_loss(vel_bwd, class_preds[:, 1:])
            preserve_fwd = preserve_foreground_velocity_loss(
                vel_fwd, vel_fwd_ref, class_preds[:, :-1]
            )
            preserve_bwd = preserve_foreground_velocity_loss(
                vel_bwd, vel_bwd_ref, class_preds[:, 1:]
            )
            loss_seg = segmentation_consistency_loss(seg_logits, seg_logits_ref)

            loss = 0.5 * (loss_fwd + loss_bwd)
            loss = loss + cfg.preserve_foreground_weight * 0.5 * (preserve_fwd + preserve_bwd)
            loss = loss + cfg.preserve_segmentation_weight * loss_seg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.reconstructor.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            step += 1

            writer.add_scalar("train/loss_step", loss_value, epoch * len(loader) + batch_idx)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch * len(loader) + batch_idx)
            writer.add_scalar("train/loss_fwd", loss_fwd.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar("train/loss_bwd", loss_bwd.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar("train/loss_preserve_fwd", preserve_fwd.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar("train/loss_preserve_bwd", preserve_bwd.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar("train/loss_seg_consistency", loss_seg.item(), epoch * len(loader) + batch_idx)

            if batch_idx % 10 == 0:
                dbg(
                    f"Epoch {epoch+1}/{cfg.epochs} batch {batch_idx}/{len(loader)} "
                    f"loss={loss_value:.6f} loss_fwd={loss_fwd.item():.6f} "
                    f"loss_bwd={loss_bwd.item():.6f}"
                )

            if cfg.checkpoint_interval_batches > 0 and (batch_idx + 1) % cfg.checkpoint_interval_batches == 0:
                save_checkpoint(model, optimizer, epoch, loss_value, cfg, tag=f"epoch{epoch+1}_batch{batch_idx+1}")

        avg_loss = epoch_loss / max(step, 1)
        elapsed = time.time() - start_time
        dbg(f"Epoch {epoch+1}/{cfg.epochs} avg_loss={avg_loss:.6f} ({elapsed:.0f}s)")

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        save_checkpoint(model, optimizer, epoch, avg_loss, cfg, tag=f"epoch{epoch+1}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f"{cfg.save_model_path_prefix}_best.ckpt"
            torch.save(model.reconstructor.state_dict(), best_path)
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"{best_path}.trainstate",
            )
            dbg(f"New best loss: {best_loss:.6f}")

    dbg("Training finished.")


def main():
    cfg = TrainConfig()
    os.makedirs(cfg.save_model_path_prefix.rsplit(".", 1)[0], exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
