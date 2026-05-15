"""Self-supervised velocity regularization training.

Only velocity_fwd_head and velocity_bwd_head are trainable. All other model
weights are frozen. A frozen hand_pred_head classifies pixels into the four
mask classes; background pixels (class 3) are regularized to zero velocity.
"""

import os
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.models.model_manager import ModelManager
from diffsynth.data.SimpleHandObjectSegmentationDataset import HandObjectSegmentationDataset, ClipStreamSampler


def dbg(msg: str):
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class TrainConfig:
    img_shape: tuple = (280, 280)
    patch_size: int = 14
    embed_dim: int = 1024
    num_classes: int = 4

    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0
    vel_loss_weight: float = 1.0
    frame_stride: int = 2
    val_fraction: float = 0.1
    seq_len: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    reconstruction_model_path: str = "models/NeoVerse/reconstructor.ckpt"
    save_model_path_prefix: str = "models/NeoVerse/vel_reg_model"
    data_root: str = "diffsynth/data/training_data_modal"
    resume_from: str = "latest"

    run_id: str = ""
    log_dir: str = ""
    num_workers: int = 2
    pin_memory: bool = True

    def __post_init__(self):
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not self.log_dir:
            self.log_dir = f"runs/neoverse_vel_reg_{self.run_id}"


class SequenceHandObjectDataset(HandObjectSegmentationDataset):
    """Dataset that returns short image sequences for motion training."""

    def __init__(self, data_root: str, frame_stride: int = 1, seq_len: int = 2, streams=None, clip_names=None):
        self.data_root = Path(data_root)
        self.streams = streams or self.streams if hasattr(self, 'streams') else None
        self._frame_stride = frame_stride
        self._seq_len = seq_len
        self._clip_names_filter = clip_names
        self.samples = []
        self._build_index()

    def _build_index(self) -> None:
        for npz_path in sorted(self.data_root.glob("clip-*.npz")):
            clip_name = npz_path.stem
            if self._clip_names_filter is not None and clip_name not in self._clip_names_filter:
                continue
            npz = np.load(str(npz_path), mmap_mode="r")
            n_frames = next(npz[k].shape[0] for k in npz.files if k.startswith("images_"))
            for stream in (self.streams or [k.split("images_")[-1] for k in npz.files if k.startswith("images_")]):
                if f"images_{stream}" not in npz.files:
                    continue
                for frame_idx in range(0, n_frames - self._seq_len + 1, self._frame_stride):
                    self.samples.append(
                        {"clip_name": clip_name, "stream": stream, "frame_idx": frame_idx}
                    )

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        clip_name, stream, frame_idx = s["clip_name"], s["stream"], s["frame_idx"]
        npz = np.load(str(self.data_root / f"{clip_name}.npz"), mmap_mode="r")

        frames = []
        for offset in range(self._seq_len):
            image = torch.tensor(
                npz[f"images_{stream}"][frame_idx + offset], dtype=torch.float32
            ).permute(2, 0, 1) / 255.0
            frames.append(image)
        images = torch.stack(frames, dim=0)

        return images, clip_name, stream


class VelRegularizer:
    """Wrapper that freezes everything except the motion heads."""

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

        n_train = 0
        n_total = 0
        for name, param in self.reconstructor.named_parameters():
            n_total += 1
            if "velocity_fwd_head" in name or "velocity_bwd_head" in name:
                param.requires_grad = True
                n_train += 1
            else:
                param.requires_grad = False
        dbg(f"Trainable: {n_train}/{n_total} params (velocity heads only).")
        if n_train == 0:
            raise RuntimeError("No parameters matched 'velocity_fwd_head' or 'velocity_bwd_head'.")

        self.backbone = self.reconstructor.visual_geometry_transformer
        self.vel_fwd_head = self.reconstructor.velocity_fwd_head
        self.vel_bwd_head = self.reconstructor.velocity_bwd_head
        self.class_head = self.reconstructor.hand_pred_head

        self.vel_fwd_head = self.vel_fwd_head.float()
        self.vel_bwd_head = self.vel_bwd_head.float()
        for p in self.vel_fwd_head.parameters():
            p.requires_grad = True
        for p in self.vel_bwd_head.parameters():
            p.requires_grad = True

        self.set_train_mode()
        dbg("Motion heads cast to float32.")

    def set_train_mode(self):
        self.reconstructor.eval()
        self.vel_fwd_head.train()
        self.vel_bwd_head.train()

    def set_eval_mode(self):
        self.reconstructor.eval()
        self.vel_fwd_head.eval()
        self.vel_bwd_head.eval()

    def trainable_parameters(self):
        params = list(self.vel_fwd_head.parameters()) + list(self.vel_bwd_head.parameters())
        return params

    @torch.no_grad()
    def _extract_motion_features(self, imgs: torch.Tensor):
        """Run frozen backbone once and return motion token lists."""
        token_list, patch_start_idx, fwd_token_list, bwd_token_list = self.backbone(
            imgs, use_motion=True
        )
        return token_list, patch_start_idx, fwd_token_list, bwd_token_list

    def forward(self, images: torch.Tensor):
        """Predict motion and segmentation for a short image sequence."""
        cfg = self.cfg
        imgs = images.to(cfg.device, non_blocking=True)

        token_list, patch_start_idx, fwd_token_list, bwd_token_list = self._extract_motion_features(imgs)

        token_list_fp32 = [t.float() for t in token_list]
        fwd_token_list_fp32 = [t.float() for t in fwd_token_list]
        bwd_token_list_fp32 = [t.float() for t in bwd_token_list]

        with torch.no_grad():
            classifications, _ = self.class_head(
                token_list_fp32,
                images=imgs.float(),
                patch_start_idx=patch_start_idx,
            )

        vel_fwd, vel_fwd_conf = self.vel_fwd_head(
            fwd_token_list_fp32,
            images=imgs[:, :-1].float(),
            patch_start_idx=patch_start_idx,
        )
        vel_bwd, vel_bwd_conf = self.vel_bwd_head(
            bwd_token_list_fp32,
            images=imgs[:, 1:].float(),
            patch_start_idx=patch_start_idx,
        )

        return {
            "velocity_fwd": vel_fwd,
            "velocity_bwd": vel_bwd,
            "classifications": classifications,
        }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def masked_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def velocity_regularization_loss(outputs, cfg: TrainConfig):
    """L2 regularize background pixels to zero motion."""
    classifications = outputs["classifications"]
    vel_fwd = outputs["velocity_fwd"]
    vel_bwd = outputs["velocity_bwd"]

    if classifications.ndim != vel_fwd.ndim + 1:
        raise RuntimeError("Unexpected classification/velocity shape mismatch.")

    class_ids = classifications.argmax(dim=-1)
    bg_fwd = (class_ids[:, :-1] == 3).unsqueeze(-1).float()
    bg_bwd = (class_ids[:, 1:] == 3).unsqueeze(-1).float()

    vel_fwd_vec = vel_fwd[..., :3]
    vel_bwd_vec = vel_bwd[..., :3]

    fwd_loss = masked_mean((vel_fwd_vec.square().sum(dim=-1, keepdim=True)), bg_fwd)
    bwd_loss = masked_mean((vel_bwd_vec.square().sum(dim=-1, keepdim=True)), bg_bwd)
    loss = cfg.vel_loss_weight * 0.5 * (fwd_loss + bwd_loss)

    return loss, fwd_loss, bwd_loss, bg_fwd.mean().item(), bg_bwd.mean().item()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, cfg, best_val_loss, global_step=None):
    params = {
        "velocity_fwd_head": model.vel_fwd_head.state_dict(),
        "velocity_bwd_head": model.vel_bwd_head.state_dict(),
    }
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": params,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "run_id": cfg.run_id,
    }
    torch.save(ckpt, f"{cfg.save_model_path_prefix}_latest.ckpt")
    epoch_path = f"{cfg.save_model_path_prefix}_run{cfg.run_id}_epoch{epoch+1:03d}.ckpt"
    torch.save(ckpt, epoch_path)
    dbg(f"  -> saved {os.path.basename(epoch_path)}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(ckpt, f"{cfg.save_model_path_prefix}_best.ckpt")
        dbg(f"  -> new best val (val_loss={best_val_loss:.4f})")
    return best_val_loss


def load_checkpoint(model, optimizer, cfg, train_loader_len: int):
    path_map = {"latest": f"{cfg.save_model_path_prefix}_latest.ckpt",
                "best":   f"{cfg.save_model_path_prefix}_best.ckpt"}
    path = path_map.get(cfg.resume_from, cfg.resume_from)
    if path is None or not os.path.exists(path):
        dbg(f"No checkpoint at {path!r} — starting fresh.")
        return 0, 0, float("inf")

    dbg(f"Resuming from {path} ...")
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    params = ckpt["model_state_dict"]
    model.vel_fwd_head.load_state_dict(params["velocity_fwd_head"], strict=True)
    model.vel_bwd_head.load_state_dict(params["velocity_bwd_head"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(cfg.device)

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", start_epoch * max(1, train_loader_len)))
    best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
    dbg(f"Resumed: start_epoch={start_epoch}  global_step={global_step}  best_val_loss={best_val_loss:.4f}")
    return start_epoch, global_step, best_val_loss


@torch.no_grad()
def evaluate(model, val_loader, cfg):
    model.set_eval_mode()
    total_loss = 0.0
    total_fwd = 0.0
    total_bwd = 0.0
    total_bg_fwd = 0.0
    total_bg_bwd = 0.0
    n_steps = 0

    for batch in val_loader:
        images, _, _ = batch
        outputs = model.forward(images)
        loss, fwd_loss, bwd_loss, bg_fwd, bg_bwd = velocity_regularization_loss(outputs, cfg)
        total_loss += loss.item()
        total_fwd += fwd_loss.item()
        total_bwd += bwd_loss.item()
        total_bg_fwd += bg_fwd
        total_bg_bwd += bg_bwd
        n_steps += 1

    model.set_train_mode()
    if n_steps == 0:
        return None, None, None, None, None
    return (
        total_loss / n_steps,
        total_fwd / n_steps,
        total_bwd / n_steps,
        total_bg_fwd / n_steps,
        total_bg_bwd / n_steps,
    )


def train():
    dbg("=== train() [vel_reg] ===")
    cfg = TrainConfig()
    dbg(f"device={cfg.device}, batch={cfg.batch_size}, epochs={cfg.epochs}, lr={cfg.learning_rate}")

    model = VelRegularizer(cfg)
    model.set_train_mode()

    all_clips = sorted(p.stem for p in Path(cfg.data_root).glob("clip-*.npz"))
    n_val = max(1, int(len(all_clips) * cfg.val_fraction))
    val_clips = set(all_clips[-n_val:])
    train_clips = set(all_clips[:-n_val])
    dbg(f"Clips: {len(all_clips)} total → {len(train_clips)} train / {len(val_clips)} val")

    train_ds = SequenceHandObjectDataset(cfg.data_root, cfg.frame_stride, cfg.seq_len, clip_names=train_clips)
    val_ds = SequenceHandObjectDataset(cfg.data_root, cfg.frame_stride, cfg.seq_len, clip_names=val_clips)
    dbg(f"Samples: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds,
        sampler=ClipStreamSampler(train_ds, shuffle_clips=True),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        sampler=ClipStreamSampler(val_ds, shuffle_clips=False),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01)

    train_loader_len = len(train_loader)
    start_epoch, global_step, best_val_loss = load_checkpoint(model, optimizer, cfg, train_loader_len)

    class_names = ["right_hand", "left_hand", "object", "background"]
    writer = SummaryWriter(log_dir=cfg.log_dir, flush_secs=10)
    os.makedirs(os.path.dirname(cfg.save_model_path_prefix), exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        dbg(f"=== Epoch {epoch+1}/{cfg.epochs} ===")
        epoch_loss = 0.0
        epoch_fwd = 0.0
        epoch_bwd = 0.0
        epoch_bg_fwd = 0.0
        epoch_bg_bwd = 0.0
        step = -1
        t_epoch = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            step += 1
            images, _, _ = batch
            outputs = model.forward(images)
            loss, fwd_loss, bwd_loss, bg_fwd, bg_bwd = velocity_regularization_loss(outputs, cfg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_fwd += fwd_loss.item()
            epoch_bwd += bwd_loss.item()
            epoch_bg_fwd += bg_fwd
            epoch_bg_bwd += bg_bwd
            global_step += 1

            writer.add_scalar("train/loss_step", loss_val, global_step)
            writer.add_scalar("train/fwd_step", fwd_loss.item(), global_step)
            writer.add_scalar("train/bwd_step", bwd_loss.item(), global_step)
            writer.add_scalar("train/bg_fwd_ratio", bg_fwd, global_step)
            writer.add_scalar("train/bg_bwd_ratio", bg_bwd, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if step < 5 or step % 20 == 0:
                dbg(
                    f"  step {step}: loss={loss_val:.4f} "
                    f"(fwd={fwd_loss.item():.4f} bwd={bwd_loss.item():.4f}) "
                    f"bg_fwd={bg_fwd:.3f} bg_bwd={bg_bwd:.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        if step < 0:
            dbg("WARNING: epoch produced 0 steps.")
            continue

        scheduler.step()
        n_steps = step + 1
        avg_loss = epoch_loss / n_steps
        avg_fwd = epoch_fwd / n_steps
        avg_bwd = epoch_bwd / n_steps
        avg_bg_fwd = epoch_bg_fwd / n_steps
        avg_bg_bwd = epoch_bg_bwd / n_steps
        elapsed = time.time() - t_epoch

        val_loss, val_fwd, val_bwd, val_bg_fwd, val_bg_bwd = evaluate(model, val_loader, cfg)

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/fwd_epoch", avg_fwd, epoch)
        writer.add_scalar("train/bwd_epoch", avg_bwd, epoch)
        writer.add_scalar("train/bg_fwd_ratio_epoch", avg_bg_fwd, epoch)
        writer.add_scalar("train/bg_bwd_ratio_epoch", avg_bg_bwd, epoch)
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)
        if val_loss is not None:
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("val/fwd_epoch", val_fwd, epoch)
            writer.add_scalar("val/bwd_epoch", val_bwd, epoch)
            writer.add_scalar("val/bg_fwd_ratio_epoch", val_bg_fwd, epoch)
            writer.add_scalar("val/bg_bwd_ratio_epoch", val_bg_bwd, epoch)

        dbg(
            f"Epoch {epoch+1}/{cfg.epochs} loss={avg_loss:.4f} "
            f"fwd={avg_fwd:.4f} bwd={avg_bwd:.4f} "
            f"bg_fwd={avg_bg_fwd:.3f} bg_bwd={avg_bg_bwd:.3f} "
            f"val_loss={val_loss:.4f} ({elapsed:.0f}s)"
        )

        best_val_loss = save_checkpoint(
            model, optimizer, epoch, avg_loss, val_loss if val_loss is not None else avg_loss,
            cfg, best_val_loss, global_step
        )

    writer.close()
    dbg("=== Training complete ===")


if __name__ == "__main__":
    train()
