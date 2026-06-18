# NeoVerse Reconstructor Benchmarks

This document explains how the two evaluation suites compute their numbers:

1. **Reconstruction / segmentation benchmark** — driven by `benchmark_rec.sh`
   → `benchmark_rec.py`. Scores a single model on **input viewpoints** (no
   novel view): how well it re-renders the frames it was given and how well it
   segments them.
2. **Keyframe-interpolation benchmark** — `bechmark_interpolation.py`
   (the rendering core) consumed by `InterpolationEvaluator` in
   `benchmark_rec.py`. Scores **held-out in-between frames** that the model
   never saw, rendered purely from the predicted velocity field.

Both share one rule of aggregation: **metrics are computed per
`(clip, stream)` window, then averaged across `(clip, stream)` groups**, so a
long clip cannot dominate the headline number. Segmentation additionally keeps
a single global confusion matrix (see below) because per-batch averaging of IoU
is biased when class frequencies vary between windows.

The class set is fixed at four labels, in this channel order:

```
0 = right_hand   1 = left_hand   2 = object   3 = background
```

---

## 1. Reconstruction / Segmentation benchmark

### What it does, in one paragraph

`benchmark_rec.sh` selects a **model type** (`neoverse`, `hand_head`, or
`gaussian_mask`), a **checkpoint**, and a **metric set** (`reconstruction`,
`segmentation`, or `both`). The benchmark cuts every validation clip/stream into
non-overlapping windows of `window_size` stride-sampled frames, runs the
reconstructor once per window, and accumulates two metric families:

- **Segmentation** — predicted class label per pixel vs. the ground-truth mask,
  scored through a global confusion matrix → mIoU, per-class IoU, pixel
  accuracy, and boundary-F1.
- **Rendering** — the Gaussian splats re-rendered at the *input* cameras vs. the
  real input frames → PSNR, SSIM, LPIPS.

Segmentation is only meaningful for the two models that carry a segmentation
head (`hand_head`, `gaussian_mask`); `neoverse` is reconstruction-only.

### The three model types

| `--model-type` | Reconstructor loading | Segmentation source |
|----------------|-----------------------|---------------------|
| `neoverse` | the checkpoint loaded whole as the reconstructor | — (no seg head) |
| `hand_head` | `reconstructor.ckpt` + 2D `hand_pred_head` overlaid | `predictions["seg_labels"]` (a DPT head over image tokens) |
| `gaussian_mask` | `reconstructor.ckpt` + `gs_head`/`gs_head_dynamic` overlaid (strict) | argmax of the **rasterized per-Gaussian mask channels** |

The `--checkpoint` flag is routed to the right slot automatically based on
`--model-type`.

### Segmentation math

For one window the model yields a predicted label map
$\hat{y} \in \{0,1,2,3\}^{S \times H \times W}$ and the GT gives
$y$ of the same shape. Every pixel contributes to a $4\times 4$ confusion
matrix $C$, where $C_{ij}$ counts pixels whose **GT class is $i$** and
**predicted class is $j$**:

$$
C_{ij} = \sum_{s,h,w} \mathbf{1}[\,y_{shw}=i\,]\,\mathbf{1}[\,\hat{y}_{shw}=j\,].
$$

In code this is a single `bincount` over the flattened index
$y \cdot K + \hat{y}$ (with $K=4$), summed into a global accumulator `self.cm`
and into a per-`(clip,stream)` accumulator. From the **accumulated** matrix:

$$
\mathrm{TP}_c = C_{cc}, \quad
\mathrm{FP}_c = \sum_i C_{ic} - C_{cc}, \quad
\mathrm{FN}_c = \sum_j C_{cj} - C_{cc}.
$$

- **Per-class IoU** (Jaccard):
  $\;\mathrm{IoU}_c = \dfrac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c}$
- **Mean IoU**: $\;\mathrm{mIoU} = \frac{1}{4}\sum_{c} \mathrm{IoU}_c$
- **Pixel accuracy**: $\;\dfrac{\sum_c \mathrm{TP}_c}{\sum_{ij} C_{ij}}$
- **Per-class pixel accuracy (recall)**:
  $\;\dfrac{\mathrm{TP}_c}{\sum_j C_{cj}}$

A small $\varepsilon=10^{-9}$ guards the denominators. Because the global matrix
is built from raw pixel counts, the reported mIoU is a true pixel-weighted score
over the whole val set — **not** an average of noisy per-window mIoUs.

**Majority-class baseline.** From the GT pixel-frequency histogram
$f_c$ (also accumulated globally), the benchmark reports the score of the
trivial classifier that predicts the single most frequent class everywhere:
its IoU equals that class's frequency, all others are zero, so
$\mathrm{mIoU}_{\text{maj}} = f_{\max}/(4\sum_c f_c)$ and its pixel accuracy is
$f_{\max}/\sum_c f_c$.

**Boundary-F1 (BF1).** Measures how well *edges* line up, tolerant to a few
pixels of slop. For each class $c$, a boundary pixel is one that survives a
$3\times 3$ erosion gap (a pixel on the object that touches a non-object pixel);
`max_pool2d` implements the morphology. Boundaries are then dilated by the
tolerance $\tau$ (default 2 px), and

$$
\text{prec} = \frac{|P_b \cap \text{dil}_\tau(G_b)|}{|P_b|},\quad
\text{rec} = \frac{|G_b \cap \text{dil}_\tau(P_b)|}{|G_b|},\quad
\mathrm{BF1}_c = \frac{2\,\text{prec}\cdot\text{rec}}{\text{prec}+\text{rec}}
$$

with $P_b$/$G_b$ the predicted/GT boundary pixels. BF1 is averaged per window,
then per clip, then across clips.

### Two inference regimes (important for `gaussian_mask`)

The two segmentation heads must be run in the regimes they were trained in,
otherwise the numbers are not comparable:

- **`hand_head`** is read from a **multi-frame dynamic forward**
  (`is_inference=True`, `is_static=False`, `timestamp = [0,1,…,S-1]`,
  `use_motion=False`). Its DPT head sees the whole window as a sequence — the
  same regime `eval_segmentation.py` scores it in.
- **`gaussian_mask`** is run **one frame per forward, static**
  (`is_static=True`, `timestamp=0`, `is_inference=False`), then the per-Gaussian
  mask channels are rasterized at that frame's own camera. This mirrors
  `training_gs_mask.GsMaskReconstructor` and `eval_segmentation_gs_mask.py`.
  Feeding the gs_mask head a *dynamic* sequence routes it through the 4DGS motion
  path its mask channels never saw and collapses mIoU to near zero — so the
  benchmark deliberately runs gs_mask through the per-frame static path
  (`_predict_gs_mask_labels`) and rasterizes at the GT resolution so the
  confusion matrix lines up pixel-for-pixel.

### Rendering math

The predicted Gaussians are rasterized at the **input** cameras (the model's own
`rendered_extrinsics` / `rendered_intrinsics`, with `w2c = inverse(c2w)`),
producing $\hat{x} \in [0,1]^{S\times 3\times H\times W}$ to compare against the
real frames $x$. Metrics come from `torchmetrics`:

- **PSNR**: $\;10\log_{10}\!\big(1 / \mathrm{MSE}(\hat{x}, x)\big)$ with data
  range 1.0.
- **SSIM**: standard structural similarity (windowed luminance/contrast/structure).
- **LPIPS**: VGG perceptual distance, inputs in $[0,1]$ (`normalize=True`).

Two aggregates are reported: a **global** torchmetrics value (updated over every
window) and a **macro** mean ± std across `(clip, stream)` groups, plus per-clip
rows in the CSV.

### Windowing (`ClipWindowDataset`)

- The val split is reproduced from the **same sorted glob + fraction** as
  training (`get_val_clips`), so no training clip leaks in.
- Frames are stride-sampled (`--frame-stride`, default 3), then chopped into
  contiguous windows of `--window-size` (default 6). Windows shorter than 2
  frames are dropped.
- `batch_size=1`: one window per step, fed as `[1, S, 3, H, W]` with one-hot GT
  masks `[S, 4, H, W]`.

### Outputs

```
runs/benchmarks/<run_id>/
  config.json
  aggregate_<model_type>.json     # global mIoU / IoU-per-class / PSNR / SSIM / LPIPS …
  baselines_<model_type>.json     # majority-class mIoU / accuracy
  per_clip_<model_type>.csv       # one row per (clip, stream)
```

---

## 2. Keyframe-interpolation benchmark

### What it does, in one paragraph

This benchmark tests the **velocity field**, not static reconstruction. Each
window is a contiguous block of $2K-1$ stride-sampled frames, split into $K$
**keyframes** (even positions) and $K-1$ **hidden targets** (odd positions). The
model is shown *only* the keyframes; each hidden frame is reconstructed by
transitioning the keyframe Gaussians along their predicted velocity to the
**midpoint timestamp** between two keyframes, rendered at an **interpolated
camera**. The rendered target is scored against the real held-out frame. This is
the only benchmark where the 4DGS motion path actually drives the output.

### What a clip is, and what "held-out frame" means

A **clip** is a contiguous video sequence of frames (one per `(clip, stream)`).
The benchmark never scores a whole clip in one shot — it chops each clip into
**non-overlapping windows** and computes the metric per window, then averages
per clip, then macro-averages across clips. So "ran over one clip" really means
"ran over each window in that clip, then aggregated."

**One window, frame by frame.** With `--num-keyframes K` (default `K=3`), a
window is a contiguous block of $2K-1$ stride-sampled frames. For `K=3` that is
5 frames, indexed 0–4:

```
index:    0      1      2      3      4
role:     KF    HID     KF    HID     KF
           \____/  \____/  \____/  \____/
            a→  ←b   a→  ←b
```

- **Even positions (0, 2, 4) = keyframes (KF).** The only frames the model sees.
- **Odd positions (1, 3) = hidden / held-out targets (HID).** Withheld from the
  model entirely.

A **held-out frame** is therefore an in-between frame at an odd position that
the model was never given. The whole point is to test whether the predicted
**velocity field** can reconstruct a frame it never saw, rather than letting the
model reconstruct it directly (which would bypass motion).

**What happens at every held-out frame** (between keyframes $a$ and $b$):

1. **Show only the keyframes** to the model (`is_inference=True,
   use_motion=True`). Because just the keyframes go in, their timestamps form a
   clean integer sequence $[0,1,\dots,K-1]$ and each keyframe gets a predicted
   `velocity_fwd` / `velocity_bwd`.
2. **Synthesize the hidden frame's camera** at the midpoint of cameras $a$ and
   $b$ (slerp rotation, linear translation/intrinsics; see below), at timestamp
   $0.5, 1.5, \dots$.
3. **Transition the Gaussians along velocity** to that midpoint timestamp and
   rasterize at the synthesized camera (bidirectional by default: $a$ pushed
   forward and $b$ pushed backward, blended).
4. **Score the render against the real held-out frame** — the actual withheld
   pixels at index 1 or 3 — with PSNR / SSIM / LPIPS and the fg/bg PSNR split.

In one line: **a clip → windows of $2K-1$ frames → even frames shown as
keyframes, odd frames held out → each held-out frame is re-rendered solely by
moving the keyframe Gaussians along predicted velocity to the midpoint
camera/timestamp, then compared pixel-wise to the real frame.**

### Why keyframes-only

At `is_inference=True` the model's `prepare_contexts` returns early and
`is_target` is ignored, so every input frame is treated as a keyframe and chained
by velocity (`forward_timestamp` = next frame's timestamp). Feeding **only**
keyframes keeps the timestamps a clean monotonic integer sequence $[0,1,\dots,K-1]$
and lets us render the hidden frames at half-integer timestamps
$0.5, 1.5, \dots$ purely from the velocity field, with no target leakage.

### Camera interpolation (`midpoint_interpolated_cameras`)

The model only predicts cameras for the keyframes it sees. A hidden frame sits at
the midpoint of two keyframes $a,b$, so its camera is synthesized:

- **Rotation** — spherical-linear interpolation of the keyframe rotations at
  $t=0.5$:
  $\;R_{\text{mid}} = \mathrm{slerp}\big(q(R_a), q(R_b),\, 0.5\big)$
  (rotation → quaternion → slerp → rotation).
- **Translation** — linear midpoint:
  $\;t_{\text{mid}} = \tfrac12 (t_a + t_b)$.
- **Intrinsics** — linear midpoint: $\;K_{\text{mid}} = \tfrac12 (K_a + K_b)$.
- **Timestamp** — $\;t_{\text{mid}} = 0.5, 1.5, \dots, K-1.5$ (one per hidden
  frame).

The world-to-camera matrices for rasterization are `homo_matrix_inverse(c2w_mid)`.

### Rendering the hidden frames (`render_interpolated_targets`)

1. Forward the keyframes with `is_inference=True, use_motion=True` → splats +
   `velocity_fwd` / `velocity_bwd` + keyframe cameras.
2. Build the midpoint cameras above.
3. Rasterize the **velocity-transitioned** Gaussians at the midpoint
   timestamps/cameras.

**Bidirectional blending.** By default the rasterizer blends two transitions for
each midpoint: the **left** keyframe pushed *forward* (`velocity_fwd`) and the
**right** keyframe pushed *backward* (`velocity_bwd`) to the same midpoint. If the
two velocities disagree, a moving object appears twice ("ghosting"). Setting
`bidirection=False` renders each midpoint from the nearest keyframe only, which
collapses the doubling — a diagnostic for forward/backward velocity disagreement.

If the model emits no `velocity_fwd`, motion did not engage; the in-between
frames are then rendered statically and the evaluator prints a warning, because
the benchmark cannot see the velocity field in that case.

### Metrics (`InterpolationEvaluator`)

All metrics are over the **target (hidden) frames only**. Predictions and GT are
resized to a common resolution (bilinear for images, nearest for masks) if needed.

**Full-frame** — global torchmetrics PSNR / SSIM / LPIPS, exactly as in §1.

**Region-split PSNR** — the most sensitive signal for velocity regularization,
which only touches background Gaussians. Using the GT background channel
(class 3) as a mask $b\in\{0,1\}$, with foreground $\text{fg}=1-b$, the evaluator
accumulates **squared error and pixel counts globally** (not a mean of per-frame
PSNRs):

$$
\mathrm{SSE}_{\text{region}} = \sum (\hat{x}-x)^2 \odot m_{\text{region}}, \qquad
\mathrm{PSNR}_{\text{region}} = -10\log_{10}\!\left(\frac{\mathrm{SSE}_{\text{region}}}{N_{\text{region}}}\right)
$$

for `region ∈ {all, fg, bg}`, where $N_{\text{region}}$ is the number of
masked pixel·channels. Reported as `PSNR_pixelwise`, `PSNR_foreground`,
`PSNR_background`. A perfect (zero-error) region is clamped to 99 dB; a region
with no pixels in a window yields NaN and is dropped from the per-clip mean.

Per-`(clip,stream)` rows additionally carry masked PSNR computed per window
(`_masked_psnr`), averaged with NaN-dropping, and every aggregate also gets a
**macro mean ± std across clips**.

### Windowing (`InterpolationWindowDataset`)

- $K \ge 2$ keyframes; block length $2K-1$; default `--num-keyframes 3`
  → 5 frames/window (3 keyframes + 2 hidden targets).
- Non-overlapping blocks per `(clip, stream)`, stride-sampled by
  `--frame-stride`.
- One sample yields keyframe images `[K,3,H,W]`, hidden target images
  `[K-1,3,H,W]`, and their GT masks `[K-1,4,H,W]` (the masks drive the
  region split, not segmentation scoring).

### Outputs

```
runs/benchmarks/<run_id>/
  aggregate_interpolation.json    # PSNR/SSIM/LPIPS + PSNR_{pixelwise,foreground,background} + *_macro
  per_clip_interpolation.csv      # PSNR / SSIM / LPIPS / PSNR_fg / PSNR_bg per (clip, stream)
```

---

## Quick reference

| | Reconstruction / Segmentation | Interpolation |
|---|---|---|
| Frames scored | input frames (re-render) | hidden in-between frames |
| Motion path | off (`use_motion=False`) | on (`use_motion=True`) |
| Cameras | model's own input cameras | slerp/linear midpoint cameras |
| Seg metrics | mIoU, IoU/class, pixel acc, BF1 | — |
| Render metrics | PSNR, SSIM, LPIPS | PSNR, SSIM, LPIPS + fg/bg PSNR |
| Aggregation | global confusion matrix + macro-mean across clips | global SSE + macro-mean across clips |
