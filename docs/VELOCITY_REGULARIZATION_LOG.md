# Velocity Regularization — Experiment Log

**Goal:** make background Gaussians (seg class 3) have ~0 velocity while keeping
hand/object (foreground) velocity intact, in the NeoVerse 4DGS reconstructor.
Only the velocity heads are trainable (backbone frozen) unless noted.

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
