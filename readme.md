
# Hands on Neoverse

This repository contains a modified fork of [NeoVerse](https://github.com/IamCreateAI/NeoVerse).

Forked from upstream commit
[`7595f14f`](https://github.com/IamCreateAI/NeoVerse/commit/7595f14f022599c3d2bec54c470c5eab80fe9c6a)
(`Add --low_vram mode to reduce peak GPU memory usage`, 2026-02-26), the last
upstream commit before our changes. The unmodified upstream reconstructor lives
under [`NeoVerse/`](NeoVerse/); our additions (HOT3D data pipeline, training,
benchmarks, demos) sit in the top-level folders described below.

## Repository layout

- **`demos/`** — visual, qualitative tools that render a model's output for a single clip so you can *see* what it does (segmentation overlays, side-by-side reconstructions, velocity-interpolation comparisons).
- **`evals/`** — quantitative benchmarks that score a model over a validation split (PSNR/SSIM/LPIPS for rendering, mIoU/IoU/accuracy/boundary-F1 for segmentation).
- **`hot3d/`** — the HOT3D data pipeline: download clips, extract images, render MANO hand/object masks, and pack everything into the per-clip NPZ training files.
- **`training/`** — training entry points for the heads we added on top of the frozen reconstructor (hand segmentation, per-Gaussian mask, velocity regularization).

### `demos/`

| Script | What it does | Run |
| --- | --- | --- |
| `multi_model_demo.py` | **Interactive multi-model viewer** — loads all three trained heads (`hand_seg`, `gs_mask`, `vel_reg`) plus the base reconstructor, renders any subset side-by-side on the same clip, with checkboxes to toggle which classes (left/right hand, object, background) are rendered and a segmentation-overlay mode. Caches forward passes and renders so toggling stays snappy; shares a public link by default. | `bash demos/run_multi_model_demo.sh` |
| `eval_segmentation.py` | Scores the 2D `hand_pred_head` on one NPZ clip and writes an `input \| prediction \| ground-truth` video. | `sbatch demos/eval_seg_job.sh` |
| `eval_segmentation_gs_mask.py` | Same evaluation but for the per-Gaussian `gs_mask` head — scores the rasterized mask channels instead of the 2D head. | `python -m demos.eval_segmentation_gs_mask --npz diffsynth/data/training_data/clip-001053.npz --gs_mask_head_path models/NeoVerse/gs_mask_model_run20260510-175056_epoch006.ckpt` |
| `reconstruction_demo.py` | Reconstruction-only Gradio app (~1–2 GB VRAM, no diffusion model). | `python -m demos.reconstruction_demo --low_vram` |
| `reconstruction_compare_demo.py` | Renders one window with one or two reconstructors next to the ground truth, annotated with per-frame PSNR/SSIM. | `python -m demos.reconstruction_compare_demo` |
| `interpolation_compare_demo.py` | Hides the odd frames of a window and compares how each model reconstructs them via the velocity field. | `python -m demos.interpolation_compare_demo` |
| `interpolation_velocity_regions_demo.py` | Variant that splits the predicted velocity field into the four segmentation regions and characterises each one separately. | `python -m demos.interpolation_velocity_regions_demo` |

Example outputs below (reduced from [`outputs/`](outputs/); full-res artifacts live there).
Each is captioned with the script that produced it and the file it writes.

**`eval_segmentation_gs_mask.py`** → `outputs/eval_seg_gs_mask_<clip>.mp4` (`input | prediction | ground truth`)

```bash
python -m demos.eval_segmentation_gs_mask --npz diffsynth/data/training_data/clip-001053.npz \
    --gs_mask_head_path models/NeoVerse/gs_mask_model_run20260510-175056_epoch006.ckpt
```

![Segmentation eval from eval_segmentation_gs_mask.py](res/readme/eval_seg_gs_mask_clip-001053.gif)

**`reconstruction_compare_demo.py`** → `outputs/recon_demo/<clip>_<stream>_w<N>/comparison.png`

```bash
python -m demos.reconstruction_compare_demo
```

![Reconstruction comparison from reconstruction_compare_demo.py](res/readme/reconstruction_compare.png)

**`interpolation_compare_demo.py`** → `outputs/interp_demo/<clip>_<stream>_w<N>/interpolation_comparison.png`

```bash
python -m demos.interpolation_compare_demo
```

![Interpolation comparison from interpolation_compare_demo.py](res/readme/interpolation_compare.png)

**`interpolation_velocity_regions_demo.py`** → `outputs/interp_velocity_regions/<clip>_<stream>/velocity_poster.png`

```bash
python -m demos.interpolation_velocity_regions_demo
```

![Velocity-region poster from interpolation_velocity_regions_demo.py](res/readme/velocity_poster.png)

### `evals/`

| Script | What it does | Run |
| --- | --- | --- |
| `benchmark_rec.py` | Current reconstruction + segmentation benchmark; supports `--mode reconstruction` and `--mode seg_compare` (hand_seg vs gs_mask). | `sbatch evals/benchmark_rec.sh` (reconstruction) or `sbatch evals/benchmark_seg_compare.sh` (head comparison) |
| `benchmark.py` | Older single-model reconstruction + segmentation benchmark. | `sbatch evals/jobscript.sh` |
| `bechmark_interpolation.py` | Keyframe-interpolation rendering core (held-out in-between frames). Not run directly — imported by `benchmark_rec.py` and the interpolation demos. | _(library module)_ |

These benchmarks emit **numeric** artifacts, not figures. `benchmark_rec.py`
writes per-run JSON/CSV under `runs/benchmarks/` (`config.json`,
`aggregate_<label>.json`, `per_clip_<label>.csv`, `comparison.json`, …) — see
[Benchmark implementation details](docs/BENCHMARKS.md) for the schema. For
visual comparisons, use the demos above.

### `hot3d/`

| Script | What it does | Run |
| --- | --- | --- |
| `optimized_download.py` | Downloads and pre-processes HOT3D clips into per-clip `clip-XXXXXX.npz` files (images + hand/object masks). | `python -m hot3d.optimized_download` |
| `extract_images.py` | Extracts per-stream images from a clip tar. | `python -m hot3d.extract_images` |
| `make_masks.py` | Renders MANO hand/object segmentation masks for a clip. | `python -m hot3d.make_masks` |
| `extract_training_data.py` | Packs extracted images + masks into NPZ and writes ready-sentinels for the dataset to ingest. | `python -m hot3d.extract_training_data` |
| `SimpleExtractTrainingData.py` | Simplified single-process variant of the training-data extractor. | `python -m hot3d.SimpleExtractTrainingData` |
| `precompute_features.py` | Pre-extracts frozen-backbone features to disk so training can skip the backbone (~85% of step time). | `python -m hot3d.precompute_features --data_root diffsynth/data/training_data --output_dir diffsynth/data/training_features --model_path models/NeoVerse/reconstructor.ckpt` |
| `dataloader.py` | Minimal WebDataset sanity-check over a HOT3D shard. | `python -m hot3d.dataloader` |



### `training/`

| Script | What it does | Run |
| --- | --- | --- |
| `hand_pred.py` | Trains the 2D `hand_pred_head` segmentation head (backbone + other heads frozen). | `sbatch training/train_classificationHead.sh` |
| `gs_mask.py` | Trains per-Gaussian mask logits, rendered through the rasterizer and supervised against GT masks (only the `gs_head` trains). | `sbatch training/train_gs_mask.sh` |
| `velocity_regularization.py` | Regularizes background (class 3) Gaussians toward ~0 velocity while leaving hand/object velocity free, using the frozen model as a teacher. | `sbatch training/train_vel_reg.sh` |

These scripts emit checkpoints to `models/NeoVerse/` and TensorBoard logs to
`runs/`, not figures. To visualize a trained checkpoint, point one of the demos
above at it (e.g. `interpolation_velocity_regions_demo.py` for a velocity
poster of a regularized model).

# Loading training data:
To generate the hand masks for the training dataset a MANO model has to be added to the repository, it can be found here: https://drive.google.com/drive/folders/1au4hhDPHVBV8G_FMBJ3i09lrl7jmoJvA?usp=sharing

# How to run

To run the evals and the demos, one of our trained models has to be loaded. The
checkpoints are included in this repository under `models/NeoVerse/` (so they are
already in place after cloning):

- `models/NeoVerse/reconstructor.ckpt` — base reconstructor
- `models/NeoVerse/hand_seg_model_opt_best.ckpt` — 2D hand segmentation head
- `models/NeoVerse/gs_mask_model_run20260510-175056_epoch006.ckpt` — per-Gaussian mask model
- `models/NeoVerse/velocity_regularization_20260604-001953_epoch1_step20099.ckpt` — velocity-regularized model

You can also browse them on GitHub:
https://github.com/lynet55/hands-on-neoverse/tree/main/models/NeoVerse

The simplest way to check out the models is to launch the **interactive
multi-model viewer**, which loads the checkpoints under `models/NeoVerse/` by
default, renders any subset of them side-by-side on the same clip, and lets you
toggle which classes (left/right hand, object, background) are rendered — all in
the browser. Run it on a GPU node; it prints a public share link by default:

```bash
bash demos/run_multi_model_demo.sh
```

The app caches forward passes and renders, so flipping the class toggles or
swapping render modes re-renders quickly. For a single-checkpoint reconstruction
view there is also `python -m demos.reconstruction_demo --low_vram`.

These scripts target the student cluster (SLURM) with a CUDA 12.8 toolchain.
Create and populate the `neoverse` virtualenv once, then submit the job scripts.

```bash
# 1. Load the matching CUDA module (every job script runs `module add cuda/12.8`)
. /etc/profile.d/modules.sh
module add cuda/12.8

# 2. Create and activate the venv (named `neoverse`, expected by every script)
python3.10 -m venv neoverse
source ./neoverse/bin/activate

# 3. Install dependencies + download the base reconstructor checkpoint
bash NeoVerse/setup.sh
```

After that, run any task by submitting its SLURM script, e.g.
`sbatch training/train_vel_reg.sh`. Each script re-loads `cuda/12.8` and
`source ./neoverse/bin/activate`, so the venv and CUDA module must exist and
match. To run a script interactively instead, activate the venv and invoke the
Python module directly (see the **Run** columns above).

# Docs

- [Benchmark implementation details](docs/BENCHMARKS.md) — how the reconstruction/segmentation and keyframe-interpolation suites compute their numbers.
- [Velocity-regularization experiment log](docs/VELOCITY_REGULARIZATION_LOG.md) — setup, frozen modules, loss terms, and the log of experiments run on vel-reg.

# Dependencies

Installed by [`NeoVerse/setup.sh`](NeoVerse/setup.sh) on top of Python 3.10 and
the `cuda/12.8` module:

| Package | Version | Source |
| --- | --- | --- |
| torch | 2.7.1 (cu128) | `download.pytorch.org/whl/cu128` |
| torchvision | 0.22.1 (cu128) | `download.pytorch.org/whl/cu128` |
| torch-scatter | matches torch-2.7.1+cu128 | `data.pyg.org` wheel |
| gsplat | git `main` | nerfstudio-project/gsplat |
| hand_tracking_toolkit | git `main` | facebookresearch |
| chumpy | git `main` | mattloper/chumpy |
| smplx | git `main` | vchoutas/smplx |

Pinned in [`NeoVerse/requirements.txt`](NeoVerse/requirements.txt):

| Package | Version |
| --- | --- |
| transformers | 4.57.6 |
| moviepy | 1.0.3 |
| deepspeed | 0.16.7 |
| safetensors, einops, numpy, Pillow, tqdm, sentencepiece | unpinned |
| imageio, imageio[ffmpeg], opencv-python, decord | unpinned |
| huggingface_hub, modelscope | unpinned |
| jaxtyping, scipy, matplotlib, evo, e3nn, addict | unpinned |
| gradio, trimesh | unpinned |
| accelerate, omegaconf, peft, tensorboard | unpinned |
| ftfy, pandas | unpinned |

Additional imports used by the `hot3d/` and `evals/` code that are **not** in
`requirements.txt` and must be installed separately:

| Package | Used by | Notes |
| --- | --- | --- |
| webdataset | `hot3d/dataloader.py`, download scripts | WebDataset streaming of HOT3D shards |
| torchmetrics | `evals/benchmark*.py` | PSNR / SSIM / LPIPS metrics |
</content>
