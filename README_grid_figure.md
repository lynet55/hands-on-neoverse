# Keyframe comparison grid (segmentation over time)

Builds a single figure where **rows = 4 consecutive keyframes** (time, top→bottom)
and **columns = Ground Truth | Naïve Gaussian | Gaussian Params | HaMeR skeleton**.
Lets you compare both how well each mask aligns with GT and how the predictions
react over time (the velocity-head-grid / NeoVerse-deformation-panel layout).

Default target: `clip-001100`, stream `stream1201-1`, frames `39 42 45 48`
(the highest hand-motion window in that clip).

## Pipeline (3 stages, 2 GPU jobs + 1 CPU step)

Two separate venvs are involved, so it is split into stages that share a single
output dir via `metadata.json`:

1. **Stage 1 — NeoVerse** (`neoverse` venv, GPU)
   `make_grid_neoverse.py` runs both segmentation variants and dumps
   `raw_*/naive_*/gsparam_*/gt_*.png` + `metadata.json`.
2. **Stage 2 — HaMeR** (`.hamer` venv in `../hamer`, GPU)
   `render_skeleton_frames.py` reads `metadata.json`, runs HaMeR, projects the
   21 MANO joints to the full frame and draws `skeleton_*.png`.
3. **Stage 3 — compose** (no GPU)
   `compose_grid.py` lays everything out into `grid_<clip>.png`.

## Run it

```bash
# stage 1 + stage 2 (stage 2 waits for stage 1 via slurm dependency)
J1=$(sbatch --parsable job_grid_stage1.sh)
sbatch --dependency=afterok:$J1 ../hamer/job_grid_stage2.sh

# stage 3, after both finish (login node is fine, no GPU)
source neoverse/bin/activate
python compose_grid.py --grid_dir outputs/grid_clip-001100
```

Output: `outputs/grid_clip-001100/grid_clip-001100.png`.

## Re-targeting to another clip / frames

Edit the `--npz`, `--stream`, `--frames`, `--out_dir` in `job_grid_stage1.sh`
(and the matching `--grid_dir` in `../hamer/job_grid_stage2.sh`). The two
`*_dir` paths must point at the same directory. Frames are absolute indices into
the raw clip (not stride-sampled).

## Checkpoints used (override with flags in `make_grid_neoverse.py`)

- reconstructor: `models/NeoVerse/reconstructor.ckpt`
- Naïve Gaussian (DPT/hand_pred_head): `models/NeoVerse/hand_seg_model_opt_best.ckpt`
- Gaussian Params (gs_mask): `models/NeoVerse/gs_mask_model_best.ckpt`
- HaMeR: `_DATA/hamer_ckpts/checkpoints/hamer.ckpt` (repo default)
