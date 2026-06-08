"""Keyframe-interpolation rendering for the NeoVerse reconstructor.

Shared by ``benchmark.py`` (numeric interpolation benchmark) and
``interpolation_compare_demo.py`` (visual side-by-side) so both tools evaluate
in *exactly* the same way: feed the model only the keyframes, then render each
hidden in-between frame by velocity-transitioning the keyframe Gaussians to the
midpoint timestamp at an interpolated camera.

Why keyframes-only: at ``is_inference=True`` the model's ``prepare_contexts``
returns early and ``is_target`` is ignored, so every input frame is treated as a
keyframe and chained by velocity (``forward_timestamp = next frame's timestamp``).
Feeding only keyframes keeps those timestamps monotonic (``[0,1,…,K-1]``) and lets
us render the hidden frames at half-integer timestamps purely from the velocity
field — no target leakage.
"""

import torch

from NeoVerse.diffsynth.auxiliary_models.worldmirror.utils.render_utils import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    slerp_quaternions,
)
from NeoVerse.diffsynth.utils.auxiliary import homo_matrix_inverse


def midpoint_interpolated_cameras(c2w: torch.Tensor, Ks: torch.Tensor):
    """Cameras halfway between consecutive keyframes.

    c2w: [K, 4, 4] camera-to-world, Ks: [K, 3, 3]. Returns (c2w_mid [K-1,4,4],
    Ks_mid [K-1,3,3], ts_mid [K-1] = 0.5,1.5,…). Rotation is slerp'd, translation
    and intrinsics are linearly interpolated. We need this because the model only
    predicts cameras for frames it sees (the keyframes); the hidden frames sit at
    the midpoints, so we synthesize their cameras here.
    """
    R_a, R_b = c2w[:-1, :3, :3], c2w[1:, :3, :3]
    q = slerp_quaternions(
        rotation_matrix_to_quaternion(R_a), rotation_matrix_to_quaternion(R_b), 0.5
    )
    R_mid = quaternion_to_rotation_matrix(q)                 # [K-1, 3, 3]
    t_mid = 0.5 * (c2w[:-1, :3, 3] + c2w[1:, :3, 3])         # [K-1, 3]

    M = c2w.shape[0] - 1
    c2w_mid = torch.eye(4, dtype=c2w.dtype, device=c2w.device).repeat(M, 1, 1)
    c2w_mid[:, :3, :3] = R_mid
    c2w_mid[:, :3, 3] = t_mid
    Ks_mid = 0.5 * (Ks[:-1] + Ks[1:])
    ts_mid = torch.arange(M, dtype=torch.float32, device=c2w.device) + 0.5
    return c2w_mid, Ks_mid, ts_mid


@torch.no_grad()
def render_interpolated_targets(reconstructor, kf_images, device, res_w, res_h,
                                return_velocity=False, bidirection=True,
                                return_vectors=False, return_splats=False):
    """Feed only the keyframes, then render the in-between frames by transitioning the
    keyframe Gaussians along their velocity to the midpoint timestamps.

    kf_images: [1, K, 3, H, W].

    Returns
    -------
    return_velocity=False : (pred [K-1, 3, res_h, res_w] in [0,1] on device, has_velocity)
    return_velocity=True  : (pred, has_velocity, vel_fwd_mag, vel_bwd_mag)
        where vel_fwd_mag / vel_bwd_mag are [K-1, H, W] per-pixel predicted velocity
        magnitudes (raw, before the renderer's dynamic/static thresholding). For the
        i-th hidden frame, ``velocity_fwd[i]`` pushes the left keyframe forward and
        ``velocity_bwd[i]`` pushes the right keyframe backward; if the two disagree the
        midpoint render shows the moving object twice. Either is None if the model
        produced no velocity.

    bidirection : when False, the rasterizer renders each midpoint from the nearest
        keyframe only (forward at the exact midpoint) instead of blending the
        forward- and backward-transitioned Gaussians. Use it to check whether ghosting
        ("two hands") comes from forward/backward velocity disagreement: the doubling
        should collapse to a single instance.
    """
    kf_images = kf_images.to(device, non_blocking=True)
    B, K, _, H, W = kf_images.shape
    views = {
        "img": kf_images,
        # is_target is ignored at inference (prepare_contexts returns early), but we pass
        # all-False so the model treats every input frame as a keyframe with a monotonic
        # integer timestamp — required so forward_timestamp >= timestamp holds.
        "is_target": torch.zeros((B, K), dtype=torch.bool, device=device),
        "is_static": torch.zeros((B, K), dtype=torch.bool, device=device),
        "timestamp": torch.arange(K, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1),
        "valid_mask": torch.ones((B, K, H, W), dtype=torch.bool, device=device),
    }

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = reconstructor(views, is_inference=True, use_motion=True)

    has_velocity = "velocity_fwd" in predictions
    gaussians = predictions["splats"]
    c2w = predictions["rendered_extrinsics"][0]    # [K, 4, 4]
    Ks = predictions["rendered_intrinsics"][0]     # [K, 3, 3]
    c2w_mid, Ks_mid, ts_mid = midpoint_interpolated_cameras(c2w, Ks)
    w2c_mid = homo_matrix_inverse(c2w_mid)

    rasterizer = reconstructor.gs_renderer.rasterizer
    prev_bidirection = rasterizer.bidirection
    rasterizer.bidirection = bidirection
    try:
        out = rasterizer.forward(
            gaussians,
            render_viewmats=[w2c_mid],
            render_Ks=[Ks_mid],
            render_timestamps=[ts_mid],
            sh_degree=0,
            width=res_w,
            height=res_h,
            render_classes=[0, 1, 2, 3],
        )
    finally:
        rasterizer.bidirection = prev_bidirection

    target_rgb = out[0] if isinstance(out, (tuple, list)) else out
    pred = target_rgb[0].permute(0, 3, 1, 2).clamp(0, 1).float()

    # World-frame positions / velocities for geometric (e.g. rigid-body) analysis.
    # ``pts3d`` is world points [B,K,H,W,3]; ``velocity_fwd/bwd`` is *camera*-frame
    # [B,K-1,H,W,3], rotated to world by the keyframe camera2world (same convention
    # the rasterizer uses: world_vel[s] = R_c2w[s] · vel_cam[s]). Flattened to
    # [*, H*W, 3] with the output (H, W) returned so callers can align GT masks
    # (the model's own seg_labels are NOT used here — their class order need not
    # match the GT mask channels).
    splat_extras = None
    if return_splats:
        pts = predictions.get("pts3d")            # [B,K,H,W,3] world
        cam = predictions.get("camera_poses")     # [B,K,4,4] camera-to-world
        hw = (pts.shape[2], pts.shape[3]) if pts is not None else None

        def _world_vel(key):
            vc = predictions.get(key)             # [B,K-1,H,W,3] camera frame
            if vc is None or cam is None:
                return None
            S_ = vc.shape[1]
            R = cam[:, :S_, :3, :3].to(vc.dtype)  # [B,K-1,3,3]
            wv = torch.einsum("bsij,bshwj->bshwi", R, vc)
            return wv[0].reshape(S_, -1, 3).float()              # [K-1, H*W, 3]

        splat_extras = {
            "means": (pts[0].reshape(pts.shape[1], -1, 3).float()  # [K, H*W, 3]
                      if pts is not None else None),
            "wvel_fwd": _world_vel("velocity_fwd"),               # [K-1, H*W, 3] or None
            "wvel_bwd": _world_vel("velocity_bwd"),
            "hw": hw,                                              # (H, W) of the above
        }

    if return_velocity or return_vectors:
        def _vec(key):
            v = predictions.get(key)               # [B, K-1, H, W, 3] or None
            return v[0].float() if v is not None else None
        def _mag(vec):
            return vec.norm(dim=-1) if vec is not None else None
        vec_fwd, vec_bwd = _vec("velocity_fwd"), _vec("velocity_bwd")
        if return_vectors:
            # (pred, has_velocity, mag_fwd, mag_bwd, vec_fwd [K-1,H,W,3], vec_bwd[, splat_extras])
            base = (pred, has_velocity, _mag(vec_fwd), _mag(vec_bwd), vec_fwd, vec_bwd)
            return base + (splat_extras,) if return_splats else base
        base = (pred, has_velocity, _mag(vec_fwd), _mag(vec_bwd))
        return base + (splat_extras,) if return_splats else base
    return (pred, has_velocity, splat_extras) if return_splats else (pred, has_velocity)
