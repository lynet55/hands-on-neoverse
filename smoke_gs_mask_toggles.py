"""Smoke-test the Gaussian-Params training experiments (#3 Lovász, #4 unfreeze,
#5 higher-res, #6 geometry-detach) for a handful of steps, BEFORE committing to a
multi-day training run. Validates the toggles don't crash and loss is finite.

Run inside the neoverse venv on a 5060ti GPU node.
"""
import torch
from torch.utils.data import DataLoader

from diffsynth.data.training import training_gs_mask as T

REAL_DATA = "/work/courses/3dv/team32/training_data_modal"
SMOKE_CLIPS = {"clip-001100", "clip-001270"}


def run(tag, **overrides):
    print(f"\n{'='*60}\nSMOKE: {tag}  overrides={overrides}\n{'='*60}", flush=True)
    cfg = T.TrainConfig(data_root=REAL_DATA, frame_stride=10, **overrides)
    model = T.GsMaskReconstructor(cfg)
    model.set_train_mode()

    ds = T.StridedHandObjectDataset(cfg.data_root, cfg.frame_stride, clip_names=SMOKE_CLIPS)
    loader = DataLoader(ds, sampler=T.ClipStreamSampler(ds, shuffle_clips=False),
                        batch_size=2, num_workers=0)

    criterion = torch.nn.CrossEntropyLoss(weight=cfg.class_weights.to(cfg.device))
    dice = T.DiceLoss()
    lovasz = T.LovaszSoftmax()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.learning_rate)
    print(f"  trainable tensors: {len(model.trainable_parameters())}", flush=True)

    it = iter(loader)
    for step in range(3):
        images, gt_mask, _, _ = next(it)
        gt_mask = gt_mask.to(cfg.device)
        opt.zero_grad()
        out = model.forward(images)
        loss, ce, dl, rgb, depth, rendered, gt_mask = T.compute_losses(
            out, gt_mask, criterion, dice, cfg, lovasz)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip_norm)
        opt.step()
        miou, _ = T.compute_miou(rendered.detach(), gt_mask, cfg.num_classes)
        print(f"  step {step}: loss={loss.item():.4f} ce={ce.item():.4f} dice={dl.item():.4f} "
              f"mIoU={miou:.4f} gnorm={float(gnorm):.3f} rendered={tuple(rendered.shape)}", flush=True)
        assert torch.isfinite(loss), "non-finite loss!"
    print(f"  OK: {tag}", flush=True)


if __name__ == "__main__":
    # Re-validate #4 unfreeze after the in-place render fix, plus the full target combo.
    run("unfreeze2",          unfreeze_last_n_blocks=2)
    run("lovasz_detach_unf2", lovasz_loss_weight=0.1, detach_geometry_for_mask=True,
        unfreeze_last_n_blocks=2)
    print("\nALL SMOKE TESTS PASSED.", flush=True)
