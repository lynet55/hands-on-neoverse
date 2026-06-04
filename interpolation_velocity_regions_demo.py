"""Per-region velocity demo for the NeoVerse reconstructor.

A variant of ``interpolation_compare_demo.py`` focused on the question:

    *Can the object be predicted with a single uniform velocity, as a rigid
    body? And the background?*

Same interpolation setup as the sibling demo — feed only the keyframes, render
the hidden in-between frames by velocity-transitioning the keyframe Gaussians to
the midpoint timestamp — but instead of treating the scene as "foreground vs
background", it splits the predicted velocity field into the four segmentation
regions and characterises each one *separately within the same image*:

    object   |   background   |   left hand   |   right hand

For every region (and model, and forward/backward direction) we record:

  * **mean speed**  ``mean ||v||`` over the region's pixels — the bulk motion.
  * **speed spread** ``std ||v||`` — error bars on the bar chart.
  * **rigidity residual** ``mean ||v - mean_vec|| / (||mean_vec|| + eps)`` — how
    far the per-pixel velocity *vectors* stray from a single shared velocity.
    ~0 means the whole region moves with one uniform velocity (a rigid
    translation); large means the velocity field is non-uniform and a rigid-body
    assumption would not reconstruct it.

The output image stacks the side-by-side renders (same PSNR / bg-PSNR / SSIM
metrics as the sibling demo) on top of two grouped bar charts — one for mean
speed per region, one for the rigidity residual per region — so the four
entities are visualised separately in one figure.

Like ``interpolation_compare_demo.py`` the renders can carry a turbo velocity
overlay (``--velocity-overlay``). Here it defaults to ``--velocity-mode
relative``: instead of absolute speed it maps each pixel's *relative* movement
``||v - v̄_region||`` — its departure from the region's single uniform velocity —
so a rigidly translating region reads ~0 (blue) and any internal/non-rigid
motion of the Gaussians lights up (red). Use ``--velocity-mode magnitude`` for
the plain absolute-speed overlay.

Examples
--------
    python interpolation_velocity_regions_demo.py \
        --velreg-path models/NeoVerse/velocity_regularization_best.ckpt

    python interpolation_velocity_regions_demo.py --clip clip-001160 \
        --stream stream1201-1 --window-index 2 --num-keyframes 3
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from diffsynth.data.SimpleHandObjectSegmentationDataset import STREAMS
from reconstruction_compare_demo import (
    get_test_clips,
    load_reconstructor,
    psnr,
    _ssim_fn,
    _to_img,
)
from diffsynth.data.benchmarking.bechmark_interpolation import render_interpolated_targets

# Reuse the window loader + masked-PSNR from the sibling demo so the two tools
# stay in lockstep (same test split, same data/split logic, same metric).
from interpolation_compare_demo import load_interp_window, masked_psnr

# Segmentation channel order in the GT masks ([4, H, W]): right, left, object, bg.
# Display order is the one the user asked for: object, background, left, right.
REGIONS = [
    ("object", 2, "#d62728"),
    ("background", 3, "#1f77b4"),
    ("left_hand", 1, "#2ca02c"),
    ("right_hand", 0, "#ff7f0e"),
]


# ===================================================================== #
#                       PER-REGION VELOCITY STATS                       #
# ===================================================================== #


def region_velocity_stats(vec: torch.Tensor, mag: torch.Tensor, mask: torch.Tensor) -> dict:
    """Characterise the velocity field inside one region for one frame.

    vec  : [h, w, 3]   per-pixel velocity vectors (render resolution)
    mag  : [h, w]      per-pixel velocity magnitudes
    mask : [h, w] bool region selector

    Returns mean/std speed and the rigidity residual (how far the per-pixel
    vectors deviate from the region's single mean velocity, relative to that mean
    speed). All NaN if the region is empty.
    """
    n = int(mask.sum().item())
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "rigidity": float("nan"), "n": 0}
    m = mask
    region_mag = mag[m]                                  # [n]
    out = {
        "mean": region_mag.mean().item(),
        "std": region_mag.std(unbiased=False).item(),
        "n": n,
    }
    region_vec = vec[m]                                  # [n, 3]
    mean_vec = region_vec.mean(dim=0)                    # [3]  the uniform/rigid guess
    resid = (region_vec - mean_vec).norm(dim=-1).mean().item()
    out["rigidity"] = resid / (mean_vec.norm().item() + 1e-8)
    return out


# ===================================================================== #
#                          VISUALIZATION                                #
# ===================================================================== #


def velocity_heatmap(vel_mag, vmax, size):
    """vel_mag [h,w] -> [3,H,W] turbo-colormapped RGB in [0,1], resized to ``size``."""
    v = vel_mag.detach().cpu().numpy()
    vmax = vmax if vmax and vmax > 0 else 1.0
    heat = plt.cm.turbo(np.clip(v / vmax, 0.0, 1.0))[..., :3]
    heat = torch.from_numpy(heat).permute(2, 0, 1).float()
    if heat.shape[-2:] != tuple(size):
        heat = F.interpolate(heat.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
    return heat


def blend_overlay(rgb, heat, alpha):
    return ((1.0 - alpha) * rgb + alpha * heat).clamp(0, 1)


def relative_deviation_map(vec: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Per-pixel departure from each region's *uniform* (rigid) velocity.

    vec   : [h, w, 3]   per-pixel velocity vectors (render resolution)
    masks : [4, h, w]   region masks (right, left, object, bg)

    Returns [h, w] where each pixel holds ``||v - v̄_region||`` — its velocity
    minus the mean velocity of the region it belongs to. A region translating as
    a rigid body has uniform ``v``, so its pixels read ~0; internal/relative
    motion (non-rigidity) shows up as the bright pixels. This is the spatial,
    per-pixel view of the ``rigidity`` residual in the bar chart.
    """
    dev = torch.zeros(vec.shape[:2], dtype=vec.dtype)
    for _, ch, _ in REGIONS:
        m = masks[ch] > 0.5
        if m.sum() == 0:
            continue
        mean_vec = vec[m].mean(dim=0)                      # region's rigid-velocity guess
        dev[m] = (vec[m] - mean_vec).norm(dim=-1)
    return dev


def disagreement_map(vec_fwd: torch.Tensor, vec_bwd: torch.Tensor) -> torch.Tensor:
    """Per-pixel forward/backward velocity disagreement ``||v_fwd + v_bwd||``.

    vec_fwd, vec_bwd : [h, w, 3]   per-pixel velocity vectors at the render grid.

    ``velocity_fwd`` pushes the left keyframe *forward* to the midpoint and
    ``velocity_bwd`` pushes the right keyframe *backward* to it, so for a point
    moving at constant velocity the two are negatives of each other and their sum
    is ~0. Where the sum is large the forward- and backward-transitioned Gaussians
    land in different places → the midpoint render shows the mover twice ("two
    hands"). This is the per-pixel field behind the bidirection ghosting.

    NOTE: ``v_fwd``/``v_bwd`` are sampled at the *same pixel* of two *different*
    keyframes, so for fast regions (pixel ≠ scene-point across the pair) this also
    picks up sampling mismatch — treat it as an upper bound there and trust it most
    on slow / background pixels.
    """
    return (vec_fwd + vec_bwd).norm(dim=-1)


def region_disagreement_stats(vec_fwd, vec_bwd, mag_fwd, mag_bwd, mask) -> dict:
    """Forward/backward disagreement inside one region for one frame.

    Returns ``disagree`` = mean ``||v_fwd + v_bwd||`` (absolute, in the same units
    as speed) and ``disagree_norm`` = that divided by the region's mean speed
    ``0.5*(||v_fwd|| + ||v_bwd||)``: 0 = the two directions perfectly oppose
    (consistent constant velocity), ~1 ≈ orthogonal, ~2 = they point the same way
    (full doubling). NaN if the region is empty.
    """
    n = int(mask.sum().item())
    if n == 0:
        return {"disagree": float("nan"), "disagree_norm": float("nan"), "n": 0}
    dis = (vec_fwd[mask] + vec_bwd[mask]).norm(dim=-1)          # [n]
    speed = 0.5 * (mag_fwd[mask] + mag_bwd[mask])              # [n]
    mean_dis = dis.mean().item()
    return {
        "disagree": mean_dis,
        "disagree_norm": mean_dis / (speed.mean().item() + 1e-8),
        "n": n,
    }


def rigid_fit_residual(points: torch.Tensor, vel: torch.Tensor) -> dict:
    """How well one rigid-body motion explains a region's world velocity field.

    points, vel : [N, 3]   world positions and world velocities of the masked
                           Gaussians (same world frame; from ``splats['means']``
                           and ``splats['world_velocity_fwd']``).

    A rigid body has a *single* angular velocity + translation, so its velocity is
    ``v(x) = t + ω × x`` — a smooth field linear in position (NOT uniform). This
    fits that 6-DoF model by least squares and reports the residual ``‖v − v_rigid‖``:
    the velocity left unexplained after allowing one rotation+translation. ~0 ⇒ the
    region already moves as one rigid body (a rigid loss would add nothing); large ⇒
    it genuinely deviates and a rigid prior would bite.

    To tell a *clean deformation* from *noise*, it also fits a general affine field
    ``v = A x + b`` and splits ``A`` into skew (rotation, rigid) and symmetric
    (strain / stretch, non-rigid). ``strain_frac`` = ‖sym‖/(‖sym‖+‖skew‖): high ⇒
    the non-rigid part is a structured stretch; if instead the affine residual is
    also large, the field is just incoherent/noisy. All residuals are also given
    normalized by the region's mean speed.
    """
    keys = ("rigid_resid", "rigid_resid_norm", "affine_resid",
            "affine_resid_norm", "strain_frac")
    n = points.shape[0]
    if n < 8:
        return {**{k: float("nan") for k in keys}, "n": n}
    # CPU double: the masked subset is small and CPU lstsq is robust/driver-agnostic.
    x = points.detach().double().cpu()
    v = vel.detach().double().cpu()
    speed = v.norm(dim=-1).mean().clamp_min(1e-9)

    # --- rigid fit: v = t + ω × x = t − [x]_× ω = [I | −[x]_×] · [t; ω] ---
    N = x.shape[0]
    zero = torch.zeros(N, device=x.device, dtype=x.dtype)
    Xx = torch.stack([                                    # [N,3,3] cross-product matrix [x]_×
        torch.stack([zero, -x[:, 2], x[:, 1]], dim=-1),
        torch.stack([x[:, 2], zero, -x[:, 0]], dim=-1),
        torch.stack([-x[:, 1], x[:, 0], zero], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=x.device, dtype=x.dtype).expand(N, 3, 3)
    design = torch.cat([I, -Xx], dim=-1).reshape(N * 3, 6)
    sol = torch.linalg.lstsq(design, v.reshape(N * 3, 1)).solution
    rigid_resid = (v - (design @ sol).reshape(N, 3)).norm(dim=-1).mean()

    # --- affine fit: v = A x + b ; split A into rotation (skew) + strain (sym) ---
    Xh = torch.cat([x, torch.ones(N, 1, device=x.device, dtype=x.dtype)], dim=-1)  # [N,4]
    sol_aff = torch.linalg.lstsq(Xh, v).solution          # [4,3]
    A_lin = sol_aff[:3].T                                  # [3,3]
    affine_resid = (v - Xh @ sol_aff).norm(dim=-1).mean()
    sym = 0.5 * (A_lin + A_lin.T)
    skew = 0.5 * (A_lin - A_lin.T)
    strain_frac = (sym.norm() / (sym.norm() + skew.norm() + 1e-12)).item()
    return {
        "rigid_resid": rigid_resid.item(),
        "rigid_resid_norm": (rigid_resid / speed).item(),
        "affine_resid": affine_resid.item(),
        "affine_resid_norm": (affine_resid / speed).item(),
        "strain_frac": strain_frac,
        "n": n,
    }


def boundary_velocity_profile(mag, masks, max_dist=40.0, bin_width=2.0):
    """Mean speed vs signed distance to the foreground/background silhouette.

    mag   : [V, h, w]      per-pixel speed (chosen direction).
    masks : [V, 4, h, w]   region masks (right, left, object, bg).

    Foreground = object ∪ left ∪ right hand; background = bg. Signed distance is
    **negative inside the foreground, positive inside the background**, 0 at the
    silhouette. Bins all pixels across the V frames by signed distance and returns
    ``(centers, mean_speed, counts)`` — the spatial ramp that exposes the slow
    boundary band: where physics wants a step (fg moves, bg static) the smooth head
    instead emits a ramp, and the near-silhouette background carries spurious
    velocity (a candidate explanation for the bg-velocity floor). Returns None if
    SciPy is unavailable.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return None
    centers = np.arange(-max_dist + bin_width / 2, max_dist, bin_width)
    edges = np.arange(-max_dist, max_dist + bin_width, bin_width)
    sums = np.zeros(len(centers))
    counts = np.zeros(len(centers))
    V = mag.shape[0]
    for i in range(V):
        fg = (masks[i, :3].sum(dim=0) > 0.5).cpu().numpy()   # object ∪ left ∪ right
        bg = (masks[i, 3] > 0.5).cpu().numpy()
        # signed distance to the fg/bg interface: + inside bg, − inside fg.
        d_bg = distance_transform_edt(bg)                    # 0 outside bg
        d_fg = distance_transform_edt(fg)
        signed = np.where(bg, d_bg, 0.0) - np.where(fg, d_fg, 0.0)
        valid = fg | bg                                      # ignore unlabeled pixels
        sd = signed[valid]
        sp = mag[i].cpu().numpy()[valid]
        idx = np.digitize(sd, edges) - 1
        ok = (idx >= 0) & (idx < len(centers))
        np.add.at(sums, idx[ok], sp[ok])
        np.add.at(counts, idx[ok], 1.0)
    mean_speed = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return centers, mean_speed, counts


def build_figure(gt_rgb, model_renders, region_summary, tgt_idxs, direction,
                 title, out_path, model_boundary=None):
    """Single image: render grid on top, per-region bar charts, boundary profile.

    gt_rgb         : [V, 3, H, W] hidden frames.
    model_renders  : name -> {"rgb","psnr","psnr_bg","ssim"}.
    region_summary : name -> region_name -> {"mean","std","rigidity","disagree",
                     "rigid_resid_norm",...} aggregated for ``direction``.
    model_boundary : name -> (centers, mean_speed, counts) for the fg/bg boundary
                     band profile, or None to omit that row.
    """
    names = list(model_renders.keys())
    V = gt_rgb.shape[0]
    n_cols = 1 + len(names)
    H, W = gt_rgb.shape[2], gt_rgb.shape[3]
    cell = 2.4
    has_boundary = bool(model_boundary) and any(v is not None for v in model_boundary.values())

    n_sub = 3 if has_boundary else 2
    h_ratios = [cell * V * (H / W), 4.0] + ([3.2] if has_boundary else [])
    fig = plt.figure(figsize=(max(cell * n_cols, 14.5),
                              cell * V * (H / W) + 4.6 + (3.4 if has_boundary else 0)))
    subs = fig.subfigures(n_sub, 1, height_ratios=h_ratios)
    top, bottom = subs[0], subs[1]
    boundary_sub = subs[2] if has_boundary else None

    # --- top: side-by-side renders (GT | model...) ---
    axes = top.subplots(V, n_cols, squeeze=False)
    col_titles = ["GT (hidden frame)", *names]
    for r in range(V):
        axes[r][0].imshow(_to_img(gt_rgb[r]))
        axes[r][0].set_ylabel(f"frame {tgt_idxs[r]}", fontsize=9)
        for c, name in enumerate(names, start=1):
            mr = model_renders[name]
            axes[r][c].imshow(_to_img(mr["rgb"][r]))
            label = f"PSNR {mr['psnr'][r]:.2f}   bg {mr['psnr_bg'][r]:.2f}"
            if mr["ssim"] is not None:
                label += f"\nSSIM {mr['ssim'][r]:.3f}"
            axes[r][c].set_xlabel(label, fontsize=8)
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=11)
    # Snug horizontally: drop the inter-image gap so the panels sit flush.
    top.subplots_adjust(wspace=0.01, hspace=0.06, left=0.04, right=0.99)

    # --- bottom: grouped bar charts, one group per region ---
    region_names = [rn for rn, _, _ in REGIONS]
    region_colors = {rn: col for rn, _, col in REGIONS}
    ax_speed, ax_rigid, ax_dis, ax_rfit = bottom.subplots(1, 4)
    x = np.arange(len(region_names))
    bw = 0.8 / max(len(names), 1)

    for j, name in enumerate(names):
        means = [region_summary[name][rn]["mean"] for rn in region_names]
        stds = [region_summary[name][rn]["std"] for rn in region_names]
        rigid = [region_summary[name][rn]["rigidity"] for rn in region_names]
        disagree = [region_summary[name][rn].get("disagree", float("nan")) for rn in region_names]
        rfit = [region_summary[name][rn].get("rigid_resid_norm", float("nan")) for rn in region_names]
        offs = x + (j - (len(names) - 1) / 2) * bw
        colors = [region_colors[rn] for rn in region_names]
        edge = "black" if j == 0 else "white"
        ax_speed.bar(offs, means, bw, yerr=stds, capsize=3, color=colors,
                     edgecolor=edge, linewidth=1.2, label=name)
        ax_rigid.bar(offs, rigid, bw, color=colors,
                     edgecolor=edge, linewidth=1.2, label=name)
        ax_dis.bar(offs, disagree, bw, color=colors,
                   edgecolor=edge, linewidth=1.2, label=name)
        ax_rfit.bar(offs, rfit, bw, color=colors,
                    edgecolor=edge, linewidth=1.2, label=name)

    for ax, ttl, ylab in (
        (ax_speed, f"Mean {direction}-velocity speed per region (± std)", "mean ‖v‖"),
        (ax_rigid, "Deviation-from-mean per region (penalizes rotation!)", "‖v − v̄‖ / ‖v̄‖"),
        (ax_dis, "Fwd/bwd disagreement per region (0 = directions cancel)", "‖v_fwd + v_bwd‖"),
        (ax_rfit, "Best-fit RIGID residual (0 = one ω+t explains it; rotation OK)",
         "‖v − v_rigid‖ / spd"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(region_names, fontsize=8, rotation=20)
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    # Per-model legend: bar fill encodes region, edge color encodes model.
    if len(names) > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.8",
                                 edgecolor="black" if j == 0 else "white", linewidth=1.5)
                   for j in range(len(names))]
        ax_speed.legend(handles, names, fontsize=8, title="model (bar edge)")

    # --- boundary band: mean speed vs signed distance to the fg/bg silhouette ---
    if boundary_sub is not None:
        ax_b = boundary_sub.subplots(1, 1)
        styles = ["-", "--", ":", "-."]
        for j, name in enumerate(names):
            prof = model_boundary.get(name)
            if prof is None:
                continue
            centers, mean_speed, counts = prof
            ax_b.plot(centers, mean_speed, styles[j % len(styles)], linewidth=1.8,
                      label=name)
        ax_b.axvline(0.0, color="0.4", linewidth=1.0)
        ax_b.text(0.01, 0.95, "← foreground (hands+object)", transform=ax_b.transAxes,
                  fontsize=8, va="top", ha="left", color="0.3")
        ax_b.text(0.99, 0.95, "background →", transform=ax_b.transAxes,
                  fontsize=8, va="top", ha="right", color="0.3")
        ax_b.set_xlabel("signed distance to fg/bg silhouette (px;  <0 inside fg,  >0 inside bg)",
                        fontsize=9)
        ax_b.set_ylabel(f"mean {direction} ‖v‖", fontsize=9)
        ax_b.set_title("Boundary band: velocity ramp across the silhouette "
                       "(physics wants a step at 0)", fontsize=10)
        ax_b.grid(alpha=0.3)
        ax_b.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===================================================================== #
#                                 CLI                                   #
# ===================================================================== #


def build_poster(results, window_order, names, direction, clip, stream, out_path,
                 frame_pick=None):
    """One poster figure for all windows.

    Top: an image grid — columns are windows, rows are methods (GT, original,
    velocity_reg). One hidden frame per cell (the velocity overlay is already
    blended into each model render). PSNR / bg-PSNR / SSIM are printed under each
    model cell with padding so the text never collides with the row below.

    Bottom: the boundary velocity ramp for every window in one axes — each window a
    colour, original dotted, velocity_reg solid (same colour as its window).
    """
    from matplotlib.lines import Line2D

    row_labels = ["GT", *names]
    nrow, ncol = len(row_labels), len(window_order)
    sample = results[window_order[0]][names[0]]["render"]["rgb"]
    V = sample.shape[0]
    fp = V // 2 if frame_pick is None else max(0, min(frame_pick, V - 1))
    H, W = sample.shape[-2:]
    cell = 3.0
    img_h = cell * nrow * (H / W)
    fig = plt.figure(figsize=(max(cell * ncol, 9.0), img_h + 4.2))
    gtop, gbot = fig.subfigures(2, 1, height_ratios=[img_h, 3.4])

    axes = gtop.subplots(nrow, ncol, squeeze=False)
    for r, rl in enumerate(row_labels):
        for c, w in enumerate(window_order):
            ax = axes[r][c]
            res_w = results[w]
            if rl == "GT":
                ref = res_w[names[0]]
                ax.imshow(_to_img(ref["tgt_images"][fp]))
                ax.set_xlabel(f"frame {ref['tgt_idxs'][fp]}", fontsize=9, labelpad=4)
            else:
                rd = res_w[rl]["render"]
                ax.imshow(_to_img(rd["rgb"][fp]))
                lbl = f"PSNR {rd['psnr'][fp]:.2f}   bg {rd['psnr_bg'][fp]:.2f}"
                if rd["ssim"] is not None:
                    lbl += f"\nSSIM {rd['ssim'][fp]:.3f}"
                ax.set_xlabel(lbl, fontsize=9, labelpad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"window {w}", fontsize=13)
            if c == 0:
                ax.set_ylabel(rl, fontsize=13)
    # Grid-like, poster-friendly spacing with room under each image for the caption.
    gtop.subplots_adjust(wspace=0.04, hspace=0.38, left=0.06, right=0.99,
                         top=0.92, bottom=0.06)

    # --- combined boundary ramp ---
    axb = gbot.subplots(1, 1)
    cmap = plt.cm.tab10
    style = {n: (":" if n == names[0] else "-") for n in names}   # original dotted, velreg solid
    for ci, w in enumerate(window_order):
        color = cmap(ci % 10)
        for n in names:
            prof = results[w][n]["boundary"]
            if prof is None:
                continue
            centers, mean_speed, _ = prof
            axb.plot(centers, mean_speed, style[n], color=color, linewidth=1.9)
    axb.axvline(0.0, color="0.4", linewidth=1.0)
    axb.text(0.01, 0.96, "← foreground (hands+object)", transform=axb.transAxes,
             fontsize=8, va="top", color="0.3")
    axb.text(0.99, 0.96, "background →", transform=axb.transAxes,
             fontsize=8, va="top", ha="right", color="0.3")
    win_handles = [Line2D([0], [0], color=cmap(ci % 10), lw=2.2, label=f"window {w}")
                   for ci, w in enumerate(window_order)]
    style_handles = [Line2D([0], [0], color="0.3", lw=2.2, ls=":", label=f"{names[0]} (dotted)")]
    if len(names) > 1:
        style_handles.append(Line2D([0], [0], color="0.3", lw=2.2, ls="-",
                                    label=f"{names[1]} (solid)"))
    leg1 = axb.legend(handles=win_handles, fontsize=8, loc="upper right", title="window (colour)")
    axb.add_artist(leg1)
    axb.legend(handles=style_handles, fontsize=8, loc="lower left", title="method (style)")
    axb.set_xlabel("signed distance to fg/bg silhouette (px;  <0 inside fg,  >0 inside bg)",
                   fontsize=10)
    axb.set_ylabel(f"mean {direction} ‖v‖", fontsize=10)
    axb.set_title("Boundary band: velocity ramp across the silhouette "
                  "(a step at 0 = clean; ramp = smeared)", fontsize=11)
    axb.grid(alpha=0.3)

    fig.suptitle(f"{clip} / {stream} — windows {window_order} ({direction})", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Per-region velocity (rigid-body) interpolation demo.")
    p.add_argument("--data-root", type=str,
                   default="/work/courses/3dv/team32/training_data_modal",
                   help="Same data_root as training/benchmark so the test split matches.")
    p.add_argument("--val-fraction", type=float, default=0.1)

    p.add_argument("--clip", type=str, default=None,
                   help="Clip stem, e.g. clip-001160. Default: first test-split clip.")
    p.add_argument("--stream", type=str, default=None,
                   help=f"Stream name. Default: first of {STREAMS}.")
    p.add_argument("--window-indices", type=int, nargs="+", default=[0, 1, 2],
                   help="Windows to run over and aggregate (default: 0 1 2). Each window gets its "
                        "own figure; the per-region and boundary stats are averaged across them for "
                        "stabler numbers than a single window gives.")
    p.add_argument("--window-index", type=int, default=None,
                   help="Shortcut for a single window; overrides --window-indices when set.")
    p.add_argument("--num-keyframes", type=int, default=3,
                   help="K: window is 2K-1 frames (K keyframes + K-1 hidden in-between targets).")
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--img-shape", type=int, nargs=2, default=(280, 280),
                   help="(H, W) — rasterizer render resolution.")

    p.add_argument("--original-path", type=str,
                   default="models/NeoVerse/reconstructor.ckpt")
    p.add_argument("--velreg-path", type=str,
                   default="models/NeoVerse/velocity_regularization_epoch1.ckpt")
    p.add_argument("--original", action=argparse.BooleanOptionalAction, default=True,
                   help="Include the original reconstructor.")
    p.add_argument("--velreg", action=argparse.BooleanOptionalAction, default=True,
                   help="Include the velocity-regularized reconstructor.")

    p.add_argument("--velocity-overlay", action=argparse.BooleanOptionalAction, default=True,
                   help="Overlay a turbo heatmap of the predicted velocity field on each render "
                        "(blue=low, red=high; on by default, --no-velocity-overlay to disable). "
                        "See --velocity-mode for what is mapped.")
    p.add_argument("--velocity-mode", type=str, default="relative",
                   choices=["magnitude", "relative", "disagreement"],
                   help="What the overlay heatmap encodes. 'magnitude': absolute speed ||v|| "
                        "(like interpolation_compare_demo). 'relative' (default): per-pixel "
                        "departure from the region's uniform velocity ||v - v̄_region|| — the "
                        "relative movement of the Gaussians, so a rigid region reads ~0 (blue) and "
                        "non-rigid motion stands out (red). 'disagreement': per-pixel forward/backward "
                        "disagreement ||v_fwd + v_bwd|| — the mismatch that makes the bidirection "
                        "render show the mover twice ('two hands'); ~0 (blue) = the directions cancel.")
    p.add_argument("--velocity-alpha", type=float, default=0.5,
                   help="Blend weight of the velocity heatmap over the render (0=off, 1=heatmap only).")
    p.add_argument("--velocity-max", type=float, default=None,
                   help="Velocity value mapped to the top of the colormap. Default: shared "
                        "95th percentile across the shown models.")
    p.add_argument("--velocity-direction", type=str, default="fwd", choices=["fwd", "bwd"],
                   help="Which velocity field to characterise in the bar charts / overlay.")
    p.add_argument("--poster-frame", type=int, default=None,
                   help="Which hidden frame (index into the K-1 targets) to show in the poster "
                        "grid. Default: the middle one.")
    p.add_argument("--single-direction", action=argparse.BooleanOptionalAction, default=False,
                   help="Render each midpoint from the nearest keyframe only (bidirection=False).")

    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="outputs/interp_velocity_regions")
    return p.parse_args()


def _agg_stats(stat_list, key):
    """Pixel-count-weighted mean of ``key`` over a list of per-frame stat dicts."""
    vals = [(s[key], s["n"]) for s in stat_list if s.get("n", 0) > 0 and s[key] == s[key]]
    if not vals:
        return float("nan")
    w = sum(n for _, n in vals)
    return sum(v * n for v, n in vals) / w


def process_window(reconstructor, name, window, args, ssim_metric, direction):
    """Render one model on one interpolation window; compute all per-region stats.

    Returns dict: render (rgb/psnr/psnr_bg/ssim), region (region→stat dict for the
    chosen direction, incl. disagreement + rigid), model_vel (overlay map [V,h,w] or
    None), boundary ((centers,mean,counts) or None), summary (JSON-friendly), plus the
    GT target images / indices for the figure.
    """
    kf_images, tgt_images, tgt_masks, kf_masks, kf_idxs, tgt_idxs = window
    H, W = args.img_shape
    pred_rgb, has_velocity, mag_fwd, mag_bwd, vec_fwd, vec_bwd, splat_extras = \
        render_interpolated_targets(
            reconstructor, kf_images.unsqueeze(0), args.device, W, H,
            return_vectors=True, return_splats=True,
            bidirection=not args.single_direction,
        )
    pred_rgb = pred_rgb.cpu()
    if not has_velocity:
        print("    WARNING: no 'velocity_fwd' — motion did not engage; in-between frames "
              "are rendered statically.", flush=True)

    # Match GT / masks to render resolution for fair metrics.
    gt, masks = tgt_images, tgt_masks
    if pred_rgb.shape[-2:] != gt.shape[-2:]:
        gt = F.interpolate(gt, size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False)
        masks = F.interpolate(masks, size=pred_rgb.shape[-2:], mode="nearest")

    def _mag_to_render(v):
        if v is None:
            return None
        return F.interpolate(v.unsqueeze(1), size=pred_rgb.shape[-2:],
                             mode="bilinear", align_corners=False)[:, 0].cpu()  # [V,h,w]

    def _vec_to_render(v):
        if v is None:
            return None
        v = v.permute(0, 3, 1, 2)                                  # [V,3,H,W]
        v = F.interpolate(v, size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False)
        return v.permute(0, 2, 3, 1).cpu()                        # [V,h,w,3]

    mag = {"fwd": _mag_to_render(mag_fwd), "bwd": _mag_to_render(mag_bwd)}
    vec = {"fwd": _vec_to_render(vec_fwd), "bwd": _vec_to_render(vec_bwd)}
    V = pred_rgb.shape[0]

    # Overlay map: speed, per-region deviation, or fwd/bwd disagreement.
    if args.velocity_mode == "disagreement" and vec["fwd"] is not None and vec["bwd"] is not None:
        model_vel = torch.stack([disagreement_map(vec["fwd"][i], vec["bwd"][i]) for i in range(V)])
    elif args.velocity_mode == "relative" and vec[direction] is not None:
        model_vel = torch.stack([relative_deviation_map(vec[direction][i], masks[i]) for i in range(V)])
    else:
        model_vel = mag[direction]

    per_psnr, per_psnr_bg = [], []
    per_ssim = [] if ssim_metric is not None else None
    per_region = {rn: {"fwd": [], "bwd": []} for rn, _, _ in REGIONS}
    per_region_dis = {rn: [] for rn, _, _ in REGIONS}

    for i in range(V):
        per_psnr.append(psnr(pred_rgb[i], gt[i]))
        bg_pix = masks[i, 3] > 0.5
        per_psnr_bg.append(masked_psnr(pred_rgb[i], gt[i], bg_pix[None].float()))
        for rn, ch, _ in REGIONS:
            region_mask = masks[i, ch] > 0.5
            for d in ("fwd", "bwd"):
                if vec[d] is not None:
                    per_region[rn][d].append(region_velocity_stats(vec[d][i], mag[d][i], region_mask))
            if vec["fwd"] is not None and vec["bwd"] is not None:
                per_region_dis[rn].append(region_disagreement_stats(
                    vec["fwd"][i], vec["bwd"][i], mag["fwd"][i], mag["bwd"][i], region_mask))
        if ssim_metric is not None:
            ssim_metric.reset()
            ssim_metric.update(pred_rgb[i:i+1].to(args.device), gt[i:i+1].to(args.device))
            per_ssim.append(ssim_metric.compute().item())

    # Best-fit rigid-body residual per region, on the model's *world* Gaussians at the
    # keyframes (positions splat_extras['means'], fwd world velocity), masked by the GT
    # keyframe masks (resized to the model output res) — NOT the model's seg labels,
    # whose class order need not match. Direction-agnostic; forward only.
    rigid_stats = {rn: {} for rn, _, _ in REGIONS}
    if (splat_extras is not None and splat_extras.get("means") is not None
            and splat_extras.get("wvel_fwd") is not None and splat_extras.get("hw") is not None):
        means = splat_extras["means"].cpu()      # [K, H*W, 3]
        wvel = splat_extras["wvel_fwd"].cpu()     # [K-1, H*W, 3]
        mh, mw = splat_extras["hw"]
        kfm = F.interpolate(kf_masks, size=(mh, mw), mode="nearest")          # [K,4,mh,mw]
        kfm_flat = kfm.reshape(kf_masks.shape[0], kf_masks.shape[1], -1)       # [K,4,mh*mw]
        for rn, ch, _ in REGIONS:
            per_kf = []
            for s in range(wvel.shape[0]):
                sel = kfm_flat[s, ch] > 0.5
                if int(sel.sum()) >= 8:
                    per_kf.append(rigid_fit_residual(means[s][sel], wvel[s][sel]))
            rigid_stats[rn] = {k: _agg_stats(per_kf, k) for k in
                               ("rigid_resid", "rigid_resid_norm", "affine_resid",
                                "affine_resid_norm", "strain_frac")}

    region = {}
    json_regions = {}
    for rn, _, _ in REGIONS:
        json_regions[rn] = {}
        for d in ("fwd", "bwd"):
            json_regions[rn][d] = {k: _agg_stats(per_region[rn][d], k)
                                   for k in ("mean", "std", "rigidity")}
        dis_agg = {k: _agg_stats(per_region_dis[rn], k) for k in ("disagree", "disagree_norm")}
        json_regions[rn]["disagreement"] = dis_agg
        json_regions[rn]["rigid"] = rigid_stats[rn]
        region[rn] = {**json_regions[rn][direction], **dis_agg, **rigid_stats[rn]}

    boundary = boundary_velocity_profile(mag[direction], masks) if mag[direction] is not None else None
    bg_vals = [v for v in per_psnr_bg if v == v]
    summary = {
        "mean_psnr": float(np.mean(per_psnr)),
        "mean_psnr_bg": float(np.mean(bg_vals)) if bg_vals else float("nan"),
        "mean_ssim": float(np.mean(per_ssim)) if per_ssim is not None else None,
        "per_frame_psnr": per_psnr,
        "per_frame_psnr_bg": per_psnr_bg,
        "regions": json_regions,
        "boundary_profile": (
            {"signed_distance": boundary[0].tolist(),
             "mean_speed": [None if v != v else float(v) for v in boundary[1]],
             "counts": boundary[2].tolist()} if boundary is not None else None),
    }
    return {"render": {"rgb": pred_rgb, "psnr": per_psnr, "psnr_bg": per_psnr_bg, "ssim": per_ssim},
            "region": region, "model_vel": model_vel, "boundary": boundary,
            "summary": summary, "tgt_images": tgt_images, "tgt_idxs": tgt_idxs}


def apply_velocity_overlay(model_renders, model_vel, args, direction):
    """Blend a turbo velocity heatmap onto each render (shared scale). Returns a title
    suffix, or '' if no model produced velocity."""
    all_vel = [v for v in model_vel.values() if v is not None]
    if not all_vel:
        return ""
    vmax = args.velocity_max or float(
        np.percentile(torch.cat([v.flatten() for v in all_vel]).numpy(), 95))
    for name in model_renders:
        v = model_vel.get(name)
        if v is None:
            continue
        rgb = model_renders[name]["rgb"]
        model_renders[name]["rgb"] = torch.stack([
            blend_overlay(rgb[i], velocity_heatmap(v[i], vmax, rgb.shape[-2:]), args.velocity_alpha)
            for i in range(rgb.shape[0])])
    mode_lbl = {"relative": "relative ‖v−v̄_region‖", "disagreement": "fwd/bwd ‖v_fwd+v_bwd‖",
                "magnitude": "speed ‖v‖"}[args.velocity_mode]
    return f"  |  {direction} {mode_lbl} overlay (turbo, vmax={vmax:.3g})"


def aggregate_regions(region_dicts):
    """Mean each region stat across windows (NaN-skipping). region_dicts: list of
    {region → {stat → val}}."""
    out = {}
    for rn, _, _ in REGIONS:
        keys = set().union(*[set(d.get(rn, {}).keys()) for d in region_dicts]) if region_dicts else set()
        out[rn] = {}
        for k in keys:
            vals = [d[rn][k] for d in region_dicts
                    if rn in d and k in d[rn] and d[rn][k] == d[rn][k]]
            out[rn][k] = float(np.mean(vals)) if vals else float("nan")
    return out


def aggregate_boundary(profiles):
    """Count-weighted combine of (centers, mean_speed, counts) profiles across windows."""
    profiles = [p for p in profiles if p is not None]
    if not profiles:
        return None
    centers = profiles[0][0]
    tot = np.zeros_like(profiles[0][1], dtype=float)
    cnt = np.zeros_like(profiles[0][2], dtype=float)
    for _, m, n in profiles:
        m = np.nan_to_num(np.asarray(m, dtype=float), nan=0.0)
        n = np.asarray(n, dtype=float)
        tot += m * n
        cnt += n
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return centers, mean, cnt


def print_region_reports(region_summary, model_boundary, names, direction):
    """Console dump of the per-region velocity / rigid / boundary tables."""
    print(f"\n=== Per-region {direction}-velocity "
          f"(mean speed | std | dev-from-mean ~0=uniform | fwd/bwd disagree ~0=cancel) ===")
    for name in names:
        print(f"  {name}:")
        for rn, _, _ in REGIONS:
            r = region_summary[name][rn]
            print(f"      {rn:>11}:  mean {r['mean']:.4f}   std {r['std']:.4f}   "
                  f"dev-from-mean {r['rigidity']:.3f}   "
                  f"disagree {r.get('disagree', float('nan')):.4f} "
                  f"({r.get('disagree_norm', float('nan')):.2f}×spd)")

    print(f"\n=== Best-fit RIGID-body residual per region (one ω+t; rotation ALLOWED) ===")
    print("    rigid_resid (abs & ×spd) = velocity left unexplained by one rigid motion; ~0 ⇒ "
          "already rigid (a rigid loss adds nothing).")
    print("    strain_frac = symmetric/(sym+skew) of the affine fit: high + low affine_resid ⇒ "
          "real stretch (non-rigid); high affine_resid ⇒ just noise.")
    for name in names:
        print(f"  {name}:")
        for rn, _, _ in REGIONS:
            r = region_summary[name][rn]
            print(f"      {rn:>11}:  rigid_resid {r.get('rigid_resid', float('nan')):.4f} "
                  f"({r.get('rigid_resid_norm', float('nan')):.2f}×spd)   "
                  f"affine_resid {r.get('affine_resid', float('nan')):.4f} "
                  f"({r.get('affine_resid_norm', float('nan')):.2f}×spd)   "
                  f"strain_frac {r.get('strain_frac', float('nan')):.2f}")

    if any(model_boundary.get(n) is not None for n in names):
        print(f"\n=== Boundary band: mean {direction} ‖v‖ vs signed distance to fg/bg silhouette "
              "(px; <0 fg, >0 bg) ===")
        probe = [-20, -10, -4, 0, 4, 10, 20]
        print(f"      {'dist(px)':>12}:" + "".join(f"{d:>9}" for d in probe))
        for name in names:
            prof = model_boundary.get(name)
            if prof is None:
                continue
            centers, mean_speed, _ = prof
            vals = "".join(f"{mean_speed[int(np.argmin(np.abs(centers - d)))]:>9.4f}" for d in probe)
            print(f"      {name:>12}:{vals}")
        print("    A clean step would jump at 0; a wide ramp + nonzero speed at >0 (inside bg) "
              "is the spurious near-silhouette background velocity.")


def main():
    args = parse_args()

    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            f"--device is '{args.device}' but a working CUDA GPU is required: the bf16 model and the "
            "Gaussian-splat rasterizer are CUDA-only. Run on a GPU node (check `nvidia-smi`)."
        )

    # --- resolve clip / stream from the test split ---
    test_clips = get_test_clips(args.data_root, args.val_fraction)
    clip = args.clip or test_clips[0]
    if args.clip is not None and args.clip not in test_clips:
        print(f"WARNING: {args.clip} is not in the test split ({len(test_clips)} clips). "
              "Using it anyway.", flush=True)
    stream = args.stream or STREAMS[0]
    print(f"Test clips: {len(test_clips)} | using clip={clip} stream={stream} "
          f"window={args.window_index} num_keyframes={args.num_keyframes}", flush=True)

    runs = []
    if args.original:
        runs.append(("original", args.original_path))
    if args.velreg:
        if Path(args.velreg_path).exists():
            runs.append(("velocity_reg", args.velreg_path))
        else:
            print(f"WARNING: velreg checkpoint {args.velreg_path} not found; skipping.", flush=True)
    if not runs:
        raise RuntimeError("No models selected (use --original / --velreg).")

    H, W = args.img_shape
    ssim_metric = _ssim_fn(args.device)
    direction = args.velocity_direction
    window_indices = ([args.window_index] if args.window_index is not None
                      else list(dict.fromkeys(args.window_indices)))
    print(f"Windows: {window_indices}", flush=True)

    # Load each window once; run each model over all windows (model-outer / window-inner
    # so the checkpoint is read a single time per model).
    windows = {}
    for w_idx in window_indices:
        try:
            windows[w_idx] = load_interp_window(
                args.data_root, clip, stream, args.num_keyframes, args.frame_stride, w_idx)
        except Exception as e:
            print(f"  skipping window {w_idx}: {e}", flush=True)
    if not windows:
        raise RuntimeError("No valid windows to process.")

    results = {w: {} for w in windows}        # results[w_idx][name] = process_window(...)
    for name, path in runs:
        print(f"[{name}] loading checkpoint ...", flush=True)
        reconstructor = load_reconstructor(path, args.device)
        for w_idx, window in windows.items():
            print(f"  window {w_idx}: keyframes {window[4]} targets {window[5]}", flush=True)
            results[w_idx][name] = process_window(
                reconstructor, name, window, args, ssim_metric, direction)
        del reconstructor
        torch.cuda.empty_cache()

    names = [n for n, _ in runs]
    window_order = list(results.keys())

    # --- blend the velocity overlay into every render with a SHARED scale across all
    # windows + models, so colours are comparable across the whole poster ---
    if args.velocity_overlay:
        all_vel = [results[w][n]["model_vel"] for w in results for n in results[w]
                   if results[w][n]["model_vel"] is not None]
        if all_vel:
            vmax = args.velocity_max or float(
                np.percentile(torch.cat([v.flatten() for v in all_vel]).numpy(), 95))
            for w in results:
                for n in results[w]:
                    v = results[w][n]["model_vel"]
                    if v is None:
                        continue
                    rgb = results[w][n]["render"]["rgb"]
                    results[w][n]["render"]["rgb"] = torch.stack([
                        blend_overlay(rgb[i], velocity_heatmap(v[i], vmax, rgb.shape[-2:]),
                                      args.velocity_alpha) for i in range(rgb.shape[0])])
            print(f"velocity overlay: shared vmax={vmax:.3g} ({args.velocity_mode})", flush=True)
        else:
            print("velocity overlay requested but no model produced velocity; skipped.", flush=True)

    # --- one combined poster (windows × methods grid + shared boundary ramp) ---
    out_dir = Path(args.output_dir) / f"{clip}_{stream}"
    poster_path = out_dir / "velocity_poster.png"
    build_poster(results, window_order, names, direction, clip, stream, poster_path,
                 frame_pick=args.poster_frame)
    for w_idx, per_name in results.items():
        with open(out_dir / f"summary_w{w_idx}.json", "w") as f:
            json.dump({"clip": clip, "stream": stream, "window_index": w_idx,
                       "direction": direction,
                       "models": {n: per_name[n]["summary"] for n in per_name}}, f, indent=2)
    print(f"\nPoster: {poster_path}", flush=True)

    # --- per-window PSNR + aggregate region/rigid/boundary over windows ---
    print("\n=== Interpolation quality per window (higher = better) ===")
    for w_idx, per_name in results.items():
        cells = "   ".join(
            f"{n} PSNR {per_name[n]['summary']['mean_psnr']:.3f}/bg "
            f"{per_name[n]['summary']['mean_psnr_bg']:.3f}" for n in per_name)
        print(f"  window {w_idx}:  {cells}")
    print(f"\n  mean over {len(results)} window(s):")
    for n in names:
        ps = [results[w][n]["summary"]["mean_psnr"] for w in results]
        bg = [results[w][n]["summary"]["mean_psnr_bg"] for w in results]
        print(f"      {n:>14}:  PSNR {np.mean(ps):.3f}   bg-PSNR {np.nanmean(bg):.3f}")

    agg_region = {n: aggregate_regions([results[w][n]["region"] for w in results]) for n in names}
    agg_boundary = {n: aggregate_boundary([results[w][n]["boundary"] for w in results]) for n in names}
    print(f"\n################  AGGREGATE over windows {list(results.keys())}  ################")
    print_region_reports(agg_region, agg_boundary, names, direction)
    print(f"\nPoster + per-window summary_w*.json under: "
          f"{Path(args.output_dir)}/{clip}_{stream}/")


if __name__ == "__main__":
    main()
