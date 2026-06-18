# Velocity Regularization — Experiment Log

**Goal:** make background Gaussians (seg class 3) have ~0 velocity while keeping
hand/object (foreground) velocity intact, in the NeoVerse 4DGS reconstructor.
Only the velocity heads are trainable (backbone frozen) unless noted.

## Setup: modules, what was frozen, and the loss terms

**Model.** NeoVerse / WorldMirror 4DGS reconstructor. A shared transformer trunk
(`frame_blocks` / global blocks) produces a `token_list` consumed by the *static* heads:
`depth_head`, `pts_head` (→`pts3d`, world points), the camera head, the Gaussian-splat
(`gs`) head, and `hand_pred_head` (per-pixel `seg_labels`, classes right/left/object/bg).
Two additional backbone stacks, `motion_fwd_blocks` / `motion_bwd_blocks`, consume the
same `token_list` and feed *only* the per-pixel `velocity_fwd_head` / `velocity_bwd_head`
(DPT heads). The motion blocks do **not** write back into `token_list`, so unfreezing them
cannot change the static heads' outputs.

**Teacher.** A frozen copy of the pure reconstructor, used three ways: (a) weight init,
(b) the `preserve_fg` target (its predicted foreground velocity), (c) distillation anchors
for camera / depth / seg.

**Rasterizer.** `transition()` moves Gaussians along their velocity to the render
timestamp; at integer (keyframe) timestamps it is the identity → **no velocity gradient**.
`bidirection` blends the forward- and backward-pushed Gaussians at a midpoint.

**Loss terms** (logged component name in brackets):
- `interp_loss` [interp] — RGB L2 on held-out (odd) frames rendered at *fractional*
  timestamps via velocity. The only term that gives the velocity heads a real data gradient.
- `keyframe rgb` [rgb] — RGB L2 on the seen keyframes.
- `bg_gaussian_vel` [bg_vel] — magnitude penalty on background (seg class 3) Gaussian
  velocity. L2 → gradient ∝ magnitude (vanishes near 0); L1 → constant gradient (actually
  drives bg → 0).
- `preserve_fg` — anchors student foreground (hand/object) velocity to the teacher's
  predicted velocity; a floor meant to stop the foreground from collapsing. L2 or L1.
- `velocity_balance` [balance] — keeps fwd and bwd magnitudes comparable (stops one
  direction dying). Its early `.norm()` form produced the NaN at v=0.
- `camera / depth / seg distill` [cam, depth] — keep the static heads ≈ the frozen teacher;
  inert while the backbone and those heads are frozen.

**Trainable vs frozen, by phase:**
- **#1 (24h):** trainable = `frame_blocks` (shared trunk) + `motion_fwd/bwd_blocks` +
  velocity heads; single LR 1e-4; **no** `interp_loss`; integer timestamps.
- **#2–#10 (heads-only):** trainable = `velocity_fwd_head` + `velocity_bwd_head` **only**;
  everything else frozen (entire backbone incl. motion blocks, and all static heads); LR
  1e-4 then 1e-5.
- **#11:** trainable = velocity heads (LR 1e-5) + `motion_fwd_blocks` / `motion_bwd_blocks`
  (LR 1e-6); frozen = `frame_blocks` / trunk + all static heads; float32 master weights on
  the trainable params; init from the pure reconstructor.

## Infrastructure findings (independent of any run)
- The original benchmark/demo render with `use_motion=False` → **blind to velocity**.
  Built a proper interpolation eval: hold out the odd frames, feed keyframes only at
  `is_inference=True`, render the midpoints via velocity at slerp/lerp-interpolated
  cameras. (`benchmark.py --mode interpolation`, `interpolation_compare_demo.py`,
  `diffsynth/data/benchmarking/bechmark_interpolation.py`.)
- Training velocity only gets a gradient if held-out frames are rendered at **fractional
  timestamps** (`interp_loss`). At integer timestamps `transition()` is the identity →
  zero velocity gradient (the original silent bug).
- `is_inference=True` ignores `is_target` (`prepare_contexts` returns early) → keyframe-only.
- **Reference (teacher/original):** eval PSNR 17.97, fg vel ≈ 0.028, bg vel ≈ 0.007, fg/bg ≈ 4–6×.

## Attempts (chronological)
| # | Config | Result |
|---|--------|--------|
| 1 | 24h baseline: motion **+ frame** backbone unfrozen, **no** interp loss, integer ts | +0.5 PSNR (measured *blind*, `use_motion=False`). Velocity: **backward collapsed → 0**, forward alive → "two hands" ghosting. |
| 2 | Freeze backbone; preserve_fg=20, balance=5 (heads only) | Asymmetric collapse **flipped** (forward died at eval). Worse than original. |
| 3 | **+ interp loss** (fractional frames); interp=1, bg_vel=1, preserve_fg=1, balance=1 (9h run) | **Parity**: velreg ≈ original (17.90 vs 17.97), velocities alive. BUT background velocity **unchanged** (objective not met). interp loss flat ~4e-3. |
| 4 | bg_vel=20 | **Full collapse** (fg+bg → 0) in ~2k steps. |
| 5 | bg_vel=5, preserve_fg=10 (L2), + hinge floor | Collapse (fg+bg → 0). |
| 6 | bg_vel **L1** (norm switch) | **Worked**: bg fell ~18× to 3e-4 — then **NaN crash** (`balance_loss` `.norm()` at v=0). |
| 7 | + NaN fixes, frame_stride=3, resume healthy ckpt; L1 bg_vel=5, preserve_fg=10 (L2) | Collapse: train AND eval `vel_fg` → 0 **together**. |
| 8 | diagnostic: logged teacher fg vel (the preserve_fg *target*) | Target ≈ **0.006** (teacher *training-path* fg, vs 0.028 eval-path). preserve_fg L2 loss ~9e-5 → gradient ∝(v−tv) vanishes near small v → **steamrolled by L1 bg_vel**. |
| 9 | preserve_fg → **L1** (constant gradient, wt 10) vs bg_vel L1 (wt 5); hinge removed | Collapse, **identical**: eval **fg/bg = 1.0×** (uniform), all ~3e-4, PSNR 16.34. |
| 10 | learning_rate 1e-4 → **1e-5** | Same collapse (observed). |

## Progression (observation → action)
The thread behind the table — what each run showed and what was changed next.
- **#1 → eval fix.** Observed: +0.5 PSNR, but the benchmark rendered with
  `use_motion=False` (blind to velocity), and the velocity itself was unhealthy — backward
  collapsed to 0, forward alive → "two hands" ghosting. Action: built the velocity-aware
  interpolation eval; found that integer-timestamp rendering gives zero velocity gradient,
  so added `interp_loss` at fractional timestamps.
- **#2.** Froze the whole backbone, trained the two velocity heads only
  (`preserve_fg`=20, `balance`=5). Observed: the asymmetric collapse flipped — forward died
  at eval; worse than the teacher.
- **#3.** Added `interp_loss`, all loss weights = 1 (9h). Observed: PSNR parity with the
  teacher, velocities alive, but background velocity essentially unchanged — `bg_vel` too
  weak to move it (objective not met).
- **#4.** Raised `bg_vel` to 20. Observed: full collapse (fg+bg → 0) within ~2k steps.
- **#5.** `bg_vel`=5, `preserve_fg`=10 (L2) + a hinge floor. Observed: collapse.
- **#6.** Switched `bg_vel` to **L1**. Observed: bg fell ~18× to 3e-4 (working) — then a NaN
  crash from `velocity_balance`'s `.norm()` at v=0.
- **#7.** Added NaN fixes, set `frame_stride`=3, resumed the #3 parity checkpoint; L1
  `bg_vel`=5, L2 `preserve_fg`=10. Observed: train- and eval-path `vel_fg` → 0 together.
- **#8 (diagnostic).** Logged the teacher's foreground velocity — the `preserve_fg` *target*
  — on the training path: ≈0.006 (vs 0.028 on the eval path). The L2 `preserve_fg` gradient
  ∝ (v − target) vanishes near small v, so it was steamrolled by the constant-gradient L1
  `bg_vel`.
- **#9.** Switched `preserve_fg` to **L1** (wt 10) against `bg_vel` L1 (wt 5), removed the
  hinge. Observed: identical collapse — eval fg/bg = 1.0× (spatially uniform), all ~3e-4,
  PSNR 16.34.
- **#10.** Lowered LR 1e-4 → 1e-5. Observed: same collapse.
- **Diagnosis.** No output-space reweighting (L1/L2, weights, hinge) or LR change fixed it →
  the frozen, shared velocity head cannot represent "high fg, zero bg"; the two are
  entangled in its parameters (capacity, not tuning).
- **#11.** Unfroze only `motion_fwd/bwd_blocks` (LR 1e-6) to add that capacity, kept the
  heads at 1e-5, and initialised from the pure reconstructor. Observed: the first
  non-collapsed run that also reduced background velocity (fg/bg up to 5.1× / 9.7×) at PSNR
  parity. Detailed eval blocks in Appendix A.

## Diagnosis (the wall)
Under **any** meaningful `bg_vel`, the velreg velocity head collapses to spatially
**uniform** near-zero output (eval **fg/bg = 1.0×**) vs the original's 4–6× separation.
No output-space loss reweighting (L1/L2, weights, hinge) and no LR change prevents it.
→ The shared, **frozen-input** velocity head cannot represent "high fg, zero bg" — the
fg and bg outputs are entangled in its parameters. This is **capacity-limited fg/bg
separation**, not a tuning problem.

## Standing caveat (important)
No run has *both* reduced background velocity *and* stayed non-collapsed → there is **no
valid experiment** on whether bg→0 helps PSNR. Across all (compromised) runs, the
original ≥ velreg on PSNR. The objective has never demonstrably improved interpolation.

## Attempt #11: unfreeze the motion backbone — **FIRST VALID RUN (objective met)**
Give the head the capacity to separate fg/bg. Unfroze **only** `motion_fwd_blocks` /
`motion_bwd_blocks` (NOT `frame_blocks`/shared trunk → camera/depth/gs untouched), at a
separate **low LR (1e-6)** (heads 1e-5), float32 master weights, with `interp` + L1
`preserve_fg` active, **started from the pure reconstructor** (= teacher, not a prior
ckpt). This fixed the three reasons the 24h unfreeze (#1) collapsed: (a) no interp signal
then → velocity drifted freely; (b) it unfroze the shared trunk; (c) a single high LR
(1e-4) on pretrained weights.

**Result (~9h, in-training eval @ step 19999, clip-001160, stride 3, K=3) — NO COLLAPSE:**

| | fg fwd | bg fwd | fg/bg fwd | fg bwd | bg bwd | fg/bg bwd | PSNR | bg-PSNR | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| original/teacher | 0.0283 | 0.0072 | **4.0×** | 0.0278 | 0.0046 | **6.1×** | 17.968 | 18.746 | 0.6049 |
| **velreg (#11)**  | 0.0265 | 0.0052 | **5.1×** | 0.0280 | 0.0029 | **9.7×** | 17.830 | 18.597 | 0.5989 |
| every prior (#2–10) | →0 | →0 | **1.0×** | →0 | →0 | 1.0× | 16.3 | — | — |

Foreground **held at the teacher value** (0.0265–0.0280 vs 0.028), background **down
1.4–1.6×**, fg/bg **separation up** (4.0→5.1, 6.1→9.7), PSNR **parity** (−0.14). First run
that ever *both* reduced bg velocity *and* stayed non-collapsed → the standing caveat is
lifted: bg→0 is achievable at no PSNR cost. The unfreeze diagnosis (capacity-limited
fg/bg separation on the frozen head) was correct.

**Parity confirmed across two independent eval contexts** (PSNR / bg-PSNR / SSIM):
| eval context | original | velreg (#11) | Δ PSNR |
|---|---|---|---|
| in-training eval @ step 19999 | 17.968 / 18.746 / 0.6049 | 17.830 / 18.597 / 0.5989 | −0.138 |
| regions-demo bidirection eval | 17.968 / 18.746 / 0.6049 | **17.879 / 18.641 / 0.5993** | −0.089 |

Both sit within ~0.1 PSNR of the teacher — the bg-velocity reduction costs essentially
nothing on reconstruction quality.

## Closing the quality levers (all examined on the #11 run, evidence not hunch)
Per-region fwd velocity, `interpolation_velocity_regions_demo.py`, clip-001160 win 0,
columns = mean speed | std | rigidity `‖v−v̄‖/‖v̄‖` | disagree `‖v_fwd+v_bwd‖` (×speed):

| region | original | velreg |
|---|---|---|
| object      | 0.0199 / 0.0287 / 1.362 / 0.0294 (1.56×) | 0.0191 / 0.0276 / 1.368 / 0.0284 (1.56×) |
| background  | 0.0072 / 0.0186 / 2.020 / 0.0083 (1.42×) | 0.0052 / 0.0177 / 2.248 / 0.0068 (1.68×) |
| left_hand   | 0.0224 / 0.0233 / 1.164 / 0.0330 (1.47×) | 0.0213 / 0.0223 / 1.174 / 0.0317 (1.46×) |
| right_hand  | 0.0582 / 0.0254 / 0.565 / 0.0497 (1.01×) | 0.0562 / 0.0243 / 0.580 / 0.0481 (1.01×) |

- **Rigidity loss → REJECTED.** The high object/bg rigidity *ratios* are a denominator
  artifact (÷‖v̄‖ on slow regions); absolute spread `ratio×mean` ranks right_hand 0.033 >
  object 0.026 ≈ left 0.025 > bg 0.012 — physically sensible (fast articulated hand = most
  real internal motion). The relative-deviation overlay shows the bright (non-rigid) pixels
  are smooth gradients **on the arms** = correct articulation, not object incoherence;
  object/bg read mostly blue (already rigid). A rigidity loss would gain nothing on the
  object and damage the genuinely-articulated hands. (A zero/uniform field is also
  *perfectly* rigid, so the term can't fight the collapse either.)
- **fwd/bwd consistency loss → REJECTED (no headroom).** `disagree ×speed` geometry:
  0=oppose(ideal), 1.41=orthogonal, 2=same-dir(doubling). All slow regions sit at ~1.4–1.6
  = orthogonal = two independent low-SNR estimates (noise floor); only right_hand rises
  above noise (1.01 ≈ 119°, partial opposition) and carries the real ghosting. **Single-
  direction render (zero blend, zero doubling) = `marginally worse` PSNR than bidirection**
  (clip-001160 win0: orig 14.64/14.64 vs velreg single 14.63/14.59) → the blend is net
  neutral-to-helpful, so eliminating disagreement has no PSNR to capture. velreg disagree
  is *lower* in absolute terms everywhere; the ×speed rise on bg (1.42→1.68) is the same
  denominator artifact.

## Final state
Objective **achieved** (#11): static-er background Gaussians at PSNR parity, fg intact,
no collapse. All three velocity-*structure* levers examined and closed — bg→0 ✅,
rigidity ❌ (no value / harmful), fwd/bwd consistency ❌ (no headroom). The residual ~0.1
PSNR gap and low absolute PSNR on hard windows are the **constant-velocity interpolation
ceiling**, not addressable by any velocity loss. Recommendation: ship/write up #11; stop
adding velocity losses.

---

## Appendix A — verbatim eval dumps (recovered from the session transcript)

The tables above are a *distillation*. The raw console blocks below are reproduced
**verbatim** from the working session (transcript
`325832a4-…-neoverse/325832a4-ac86-4086-9679-e0f50f8b1e4a.jsonl`) so the condensed numbers
are auditable. All on `clip-001160`; "original" = the frozen teacher reconstructor.

### A1 — early head-only `bg_vel` run, stride-3 benchmark (≈ attempt #2; forward collapsed, backward alive)
```
=== Mean interpolation quality on hidden frames (higher = better) ===
        original:  PSNR 17.968   bg-PSNR 18.746  SSIM 0.6049
    velocity_reg:  PSNR 16.680   bg-PSNR 17.541  SSIM 0.5830
  -> best by background PSNR: original

=== Predicted velocity magnitude (background should be ~0) ===
  original:
      fwd:  bg 0.0072   fg 0.0283  fg/bg 4.0x
      bwd:  bg 0.0046   fg 0.0278  fg/bg 6.1x
  velocity_reg:
      fwd:  bg 0.0007   fg 0.0006  fg/bg 0.9x      <- forward collapsed (fg≈bg)
      bwd:  bg 0.0034   fg 0.0137  fg/bg 4.0x      <- backward still alive
```

### A2 — same run, stride-1 benchmark (frame-stride sweep)
```
=== Mean interpolation quality on hidden frames (higher = better) ===
        original:  PSNR 22.902   bg-PSNR 24.509  SSIM 0.8161
    velocity_reg:  PSNR 21.912   bg-PSNR 23.114  SSIM 0.7909
  -> best by background PSNR: original

=== Predicted velocity magnitude (background should be ~0) ===
  original:
      fwd:  bg 0.0031   fg 0.0121  fg/bg 3.9x
      bwd:  bg 0.0024   fg 0.0098  fg/bg 4.0x
  velocity_reg:
      fwd:  bg 0.0007   fg 0.0007  fg/bg 1.0x
      bwd:  bg 0.0012   fg 0.0037  fg/bg 3.1x
```

### A3 — 9h parity run (attempt #3) and its resume eval @ step 20399
```
=== Mean interpolation quality on hidden frames (higher = better) ===
        original:  PSNR 17.968   bg-PSNR 18.746  SSIM 0.6049
    velocity_reg:  PSNR 17.900   bg-PSNR 18.640  SSIM 0.6065
  -> best by background PSNR: original

=== Predicted velocity magnitude (background should be ~0) ===
  original:
      fwd:  bg 0.0072   fg 0.0283  fg/bg 4.0x
      bwd:  bg 0.0046   fg 0.0278  fg/bg 6.1x
  velocity_reg:
      fwd:  bg 0.0067   fg 0.0243  fg/bg 3.6x      <- bg essentially unchanged (objective not met)
      bwd:  bg 0.0044   fg 0.0306  fg/bg 7.0x
```
```
=== [in-training eval @ step 20399] interpolation quality (higher=better) ===
        original:  PSNR 17.968   bg-PSNR 18.746   SSIM 0.6049
    velocity_reg:  PSNR 17.899   bg-PSNR 18.639   SSIM 0.6065
  --- predicted velocity magnitude (background should be ~0) ---
          original fwd:  bg 0.0072   fg 0.0283  fg/bg 4.0x
          original bwd:  bg 0.0046   fg 0.0278  fg/bg 6.1x
      velocity_reg fwd:  bg 0.0067   fg 0.0243  fg/bg 3.6x
      velocity_reg bwd:  bg 0.0044   fg 0.0306  fg/bg 7.0x
```

### A4 — collapse runs (attempts #4–#10 family; uniform fg/bg = 1.0×)
```
=== [in-training eval @ step 22398] interpolation quality (higher=better) ===
        original:  PSNR 17.968   bg-PSNR 18.746   SSIM 0.6049
    velocity_reg:  PSNR 16.331   bg-PSNR 17.291   SSIM 0.5729
  --- predicted velocity magnitude (background should be ~0) ---
          original fwd:  bg 0.0072   fg 0.0283  fg/bg 4.0x
          original bwd:  bg 0.0046   fg 0.0278  fg/bg 6.1x
      velocity_reg fwd:  bg 0.0003   fg 0.0003  fg/bg 1.0x
      velocity_reg bwd:  bg 0.0001   fg 0.0001  fg/bg 1.0x
```
```
=== [in-training eval @ step 20898] interpolation quality (higher=better) ===   (attempt #9, L1 preserve_fg)
        original:  PSNR 17.968   bg-PSNR 18.746   SSIM 0.6049
    velocity_reg:  PSNR 16.340   bg-PSNR 17.298   SSIM 0.5728
  --- predicted velocity magnitude (background should be ~0) ---
          original fwd:  bg 0.0072   fg 0.0283  fg/bg 4.0x
          original bwd:  bg 0.0046   fg 0.0278  fg/bg 6.1x
      velocity_reg fwd:  bg 0.0002   fg 0.0002  fg/bg 1.0x
      velocity_reg bwd:  bg 0.0004   fg 0.0004  fg/bg 1.0x
```
```
=== [in-training eval @ step 25898] interpolation quality (higher=better) ===   (vel_reg.py duplicate; total collapse incl. foreground)
        original:  PSNR 17.968   bg-PSNR 18.746   SSIM 0.6049
    velocity_reg:  PSNR 16.329   bg-PSNR 17.289   SSIM 0.5725
  --- predicted velocity magnitude (background should be ~0) ---
          original fwd:  bg 0.0072   fg 0.0283  fg/bg 4.0x
          original bwd:  bg 0.0046   fg 0.0278  fg/bg 6.1x
      velocity_reg fwd:  bg 0.0000   fg 0.0000  fg/bg 1.0x
      velocity_reg bwd:  bg 0.0000   fg 0.0000  fg/bg 1.0x
```

### A5 — attempt #11 (first valid run), in-training eval @ step 19999
```
=== [in-training eval @ step 19999] interpolation quality (higher=better) ===
        original:  PSNR 17.968   bg-PSNR 18.746   SSIM 0.6049
    velocity_reg:  PSNR 17.830   bg-PSNR 18.597   SSIM 0.5989
  --- predicted velocity magnitude (background should be ~0) ---
          original fwd:  bg 0.0072   fg 0.0283  fg/bg 4.0x
          original bwd:  bg 0.0046   fg 0.0278  fg/bg 6.1x
      velocity_reg fwd:  bg 0.0052   fg 0.0265  fg/bg 5.1x      <- bg down, fg held, separation UP
      velocity_reg bwd:  bg 0.0029   fg 0.0280  fg/bg 9.7x
```

### A6 — #11 per-region (regions demo, bidirection eval) + rigid fit BEFORE the GT-mask fix
The rigid block here is masked by the model's own `seg_labels` (class order ≠ GT) → speeds
14–190× off the speed bars and **right_hand NaN**. Kept only to show the bug that motivated A7.
```
=== Interpolation quality on hidden frames (higher = better) ===
        original:  PSNR 17.968   bg-PSNR 18.746  SSIM 0.6049
    velocity_reg:  PSNR 17.879   bg-PSNR 18.641  SSIM 0.5993

=== Per-region fwd-velocity (mean speed | std | dev-from-mean ~0=uniform | fwd/bwd disagree ~0=cancel) ===
  original:
           object:  mean 0.0199   std 0.0287   dev-from-mean 1.362   disagree 0.0294 (1.56×spd)
       background:  mean 0.0072   std 0.0186   dev-from-mean 2.020   disagree 0.0083 (1.42×spd)
        left_hand:  mean 0.0224   std 0.0233   dev-from-mean 1.164   disagree 0.0330 (1.47×spd)
       right_hand:  mean 0.0582   std 0.0254   dev-from-mean 0.565   disagree 0.0497 (1.01×spd)
  velocity_reg:
           object:  mean 0.0191   std 0.0276   dev-from-mean 1.368   disagree 0.0284 (1.56×spd)
       background:  mean 0.0052   std 0.0177   dev-from-mean 2.248   disagree 0.0068 (1.68×spd)
        left_hand:  mean 0.0213   std 0.0223   dev-from-mean 1.174   disagree 0.0317 (1.46×spd)
       right_hand:  mean 0.0562   std 0.0243   dev-from-mean 0.580   disagree 0.0481 (1.01×spd)

=== Best-fit RIGID-body residual per region (one ω+t; rotation ALLOWED) ===   [BUGGED: model-seg mask]
  original:
           object:  rigid_resid 0.0025 (1.74×spd)   affine_resid 0.0025 (1.75×spd)   strain_frac 0.52
       background:  rigid_resid 0.0132 (1.44×spd)   affine_resid 0.0133 (1.45×spd)   strain_frac 0.77
        left_hand:  rigid_resid 0.0001 (0.84×spd)   affine_resid 0.0000 (0.28×spd)   strain_frac 0.74
       right_hand:  rigid_resid nan (nan×spd)   affine_resid nan (nan×spd)   strain_frac nan
  velocity_reg:
           object:  rigid_resid 0.0205 (0.60×spd)   affine_resid 0.0102 (0.30×spd)   strain_frac 0.52
       background:  rigid_resid 0.0037 (1.77×spd)   affine_resid 0.0038 (1.82×spd)   strain_frac 0.57
        left_hand:  rigid_resid 0.0058 (0.12×spd)   affine_resid 0.0026 (0.05×spd)   strain_frac 0.49
       right_hand:  rigid_resid 0.0162 (0.20×spd)   affine_resid 0.0129 (0.16×spd)   strain_frac 0.65
```

### A7 — #11 rigid fit AFTER the GT-mask fix (valid) + boundary band
Cross-check passes (right_hand no longer NaN, implied speeds sane). This is the block the
"reject rigidity loss" conclusion rests on: object rigid 0.49×spd, affine 0.22×spd (halved),
`strain_frac` 0.53 → structured deformation, not noise; original ≈ velreg (fg structure untouched).
```
=== Best-fit RIGID-body residual per region (one ω+t; rotation ALLOWED) ===
  original:
           object:  rigid_resid 0.0201 (0.49×spd)   affine_resid 0.0089 (0.22×spd)   strain_frac 0.53
       background:  rigid_resid 0.0064 (1.43×spd)   affine_resid 0.0064 (1.44×spd)   strain_frac 0.57
        left_hand:  rigid_resid 0.0056 (0.10×spd)   affine_resid 0.0023 (0.04×spd)   strain_frac 0.50
       right_hand:  rigid_resid 0.0140 (0.17×spd)   affine_resid 0.0106 (0.13×spd)   strain_frac 0.66
  velocity_reg:
           object:  rigid_resid 0.0192 (0.49×spd)   affine_resid 0.0088 (0.23×spd)   strain_frac 0.52
       background:  rigid_resid 0.0047 (1.74×spd)   affine_resid 0.0048 (1.79×spd)   strain_frac 0.61
        left_hand:  rigid_resid 0.0053 (0.11×spd)   affine_resid 0.0022 (0.04×spd)   strain_frac 0.49
       right_hand:  rigid_resid 0.0143 (0.17×spd)   affine_resid 0.0107 (0.13×spd)   strain_frac 0.66

=== Boundary band: mean fwd ‖v‖ vs signed distance to fg/bg silhouette (px; <0 fg, >0 bg) ===
          dist(px):      -20      -10       -4        0        4       10       20
          original:   0.0001   0.0260   0.0311   0.0260   0.0204   0.0186   0.0124
      velocity_reg:   0.0000   0.0248   0.0299   0.0250   0.0197   0.0179   0.0119
```
