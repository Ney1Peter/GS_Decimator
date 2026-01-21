# Gaussian Splat Decimation Tool
# Author: Felix Hirt (base) + modifications
# License: MIT

import argparse
import time
import os
import re
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label
from plyfile import PlyData, PlyElement


import torch
import torch_scatter

import helpers.PointModel as GS


# ---------------------------
# Filename-based camera parsing
# ---------------------------
def parse_ply_name_camera(ply_path: str):
    """
    Parse focal, width, height from filename:
      {base}_{focal}_{W}_{H}.ply
    We parse the LAST 3 underscore-separated tokens of the stem as integers.
    Returns: base_name(str), focal(int), W(int), H(int)
    """
    p = Path(ply_path)
    stem = p.stem  # filename without suffix
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"PLY filename must be like {{name}}_{{focal}}_{{W}}_{{H}}.ply, got: {p.name}"
        )

    focal_s, w_s, h_s = parts[-3], parts[-2], parts[-1]
    if not (focal_s.isdigit() and w_s.isdigit() and h_s.isdigit()):
        raise ValueError(
            f"Last 3 tokens must be integers: focal,W,H. Got: {focal_s},{w_s},{h_s} in {p.name}"
        )

    focal = int(focal_s)
    W = int(w_s)
    H = int(h_s)

    base = "_".join(parts[:-3])
    if base == "":
        base = "model"
    return base, focal, W, H


def ensure_output_name(save_path: str, focal: int, W: int, H: int) -> str:
    """
    Ensure output filename follows {name}_{focal}_{W}_{H}.ply.
    If save_path already ends with _f_W_H.ply, keep it.
    Otherwise append _f_W_H before .ply.
    """
    p = Path(save_path)
    if p.suffix.lower() != ".ply":
        raise ValueError("--save_path must end with .ply")

    stem = p.stem
    m = re.match(r"^(.*)_(\d+)_(\d+)_(\d+)$", stem)
    if m:
        # already has 3 integers at end; keep as is
        return str(p)

    new_name = f"{stem}_{int(focal)}_{int(W)}_{int(H)}.ply"
    return str(p.with_name(new_name))


def build_K_from_filename(focal_px: int, W: int, H: int) -> np.ndarray:
    """
    Simple pinhole K from focal in pixels and image size.
    Principal point set to image center.
    Using (W-1)/2 is common for pixel-center convention.
    """
    fx = float(focal_px)
    fy = float(focal_px)
    cx = (float(W) - 1.0) * 0.5
    cy = (float(H) - 1.0) * 0.5
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


# ---------------------------
# Math helpers
# ---------------------------
def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions (w, x, y, z).
    R: (B,3,3)
    """
    B = R.shape[0]
    q = torch.zeros((B, 4), device=R.device, dtype=R.dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    mask = trace > 0
    if mask.any():
        t = trace[mask]
        s = torch.sqrt(t + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    mask2 = ~mask
    if mask2.any():
        R2 = R[mask2]
        diag = torch.stack([R2[:, 0, 0], R2[:, 1, 1], R2[:, 2, 2]], dim=1)
        idx = torch.argmax(diag, dim=1)

        q2 = torch.zeros((R2.shape[0], 4), device=R.device, dtype=R.dtype)

        m0 = idx == 0
        if m0.any():
            s = torch.sqrt(1.0 + R2[m0, 0, 0] - R2[m0, 1, 1] - R2[m0, 2, 2]) * 2.0
            q2[m0, 0] = (R2[m0, 2, 1] - R2[m0, 1, 2]) / s
            q2[m0, 1] = 0.25 * s
            q2[m0, 2] = (R2[m0, 0, 1] + R2[m0, 1, 0]) / s
            q2[m0, 3] = (R2[m0, 0, 2] + R2[m0, 2, 0]) / s

        m1 = idx == 1
        if m1.any():
            s = torch.sqrt(1.0 + R2[m1, 1, 1] - R2[m1, 0, 0] - R2[m1, 2, 2]) * 2.0
            q2[m1, 0] = (R2[m1, 0, 2] - R2[m1, 2, 0]) / s
            q2[m1, 1] = (R2[m1, 0, 1] + R2[m1, 1, 0]) / s
            q2[m1, 2] = 0.25 * s
            q2[m1, 3] = (R2[m1, 1, 2] + R2[m1, 2, 1]) / s

        m2 = idx == 2
        if m2.any():
            s = torch.sqrt(1.0 + R2[m2, 2, 2] - R2[m2, 0, 0] - R2[m2, 1, 1]) * 2.0
            q2[m2, 0] = (R2[m2, 1, 0] - R2[m2, 0, 1]) / s
            q2[m2, 1] = (R2[m2, 0, 2] + R2[m2, 2, 0]) / s
            q2[m2, 2] = (R2[m2, 1, 2] + R2[m2, 2, 1]) / s
            q2[m2, 3] = 0.25 * s

        q[mask2] = q2

    q = q / (q.norm(dim=1, keepdim=True) + 1e-12)
    return q


def quaternion_mean_markley(quats, cluster_ids, num_clusters, chunk: int = 20000):
    """
    Markley quaternion averaging (vectorized).
    quats: (N,4) normalized, cluster_ids: (N,)
    """
    outer = quats.unsqueeze(2) * quats.unsqueeze(1)  # (N,4,4)
    M_sum = torch_scatter.scatter_sum(outer, cluster_ids, dim=0, dim_size=num_clusters)

    ones = torch.ones((quats.size(0), 1, 1), device=quats.device, dtype=quats.dtype).expand(-1, 4, 4)
    counts = torch_scatter.scatter_sum(ones, cluster_ids, dim=0, dim_size=num_clusters).clamp_min(1.0)
    M = M_sum / counts

    M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
    M = M + torch.eye(4, device=M.device, dtype=M.dtype).unsqueeze(0) * 1e-8
    mask_invalid = torch.isnan(M).any(dim=(1, 2)) | torch.isinf(M).any(dim=(1, 2))
    if mask_invalid.any():
        M[mask_invalid] = torch.eye(4, device=M.device, dtype=M.dtype).unsqueeze(0)

    eigvals_list, eigvecs_list = [], []
    for i in range(0, M.size(0), chunk):
        M_chunk = torch.nan_to_num(M[i:i+chunk], nan=0.0, posinf=1e6, neginf=-1e6)
        M_chunk = M_chunk + torch.eye(4, device=M.device, dtype=M.dtype).unsqueeze(0) * 1e-8
        ev, V = torch.linalg.eigh(M_chunk)
        eigvals_list.append(ev); eigvecs_list.append(V)
    eigvals = torch.cat(eigvals_list, 0)
    eigvecs = torch.cat(eigvecs_list, 0)

    max_ids = eigvals.argmax(dim=1)
    mean_quats = eigvecs[torch.arange(num_clusters, device=quats.device), :, max_ids]
    mean_quats = mean_quats / (mean_quats.norm(dim=1, keepdim=True) + 1e-12)
    return mean_quats


# ---------------------------
# StageA (Original) merge
# ---------------------------
def compute_density_aware_radius_fast(xyz, base_radius, voxel_size=None, show_progress=False):
    if voxel_size is None:
        voxel_size = base_radius

    voxel_idx = torch.floor(xyz / voxel_size).long()
    voxel_keys = voxel_idx[:, 0] + voxel_idx[:, 1] * 1000 + voxel_idx[:, 2] * 1000000

    unique_keys, counts = torch.unique(voxel_keys, return_counts=True)
    inv_idx = torch.bucketize(voxel_keys, unique_keys)
    density_map = counts[inv_idx - 1].float()

    median_density = torch.median(counts.float())
    density_scale = (median_density / density_map).clamp(0.3, 3.0)
    return base_radius * density_scale


def decimate_original(base_radius, gaussian_model, density_aware=False):
    xyz = gaussian_model._xyz
    scaling = gaussian_model.get_scaling
    rotation = gaussian_model.get_rotation
    features_dc = gaussian_model._features_dc
    features_rest = gaussian_model._features_rest
    opacity = gaussian_model.get_opacity

    if density_aware:
        radii = compute_density_aware_radius_fast(xyz, base_radius)
    else:
        radii = torch.full((xyz.shape[0],), float(base_radius), device=xyz.device)

    voxel_idx = torch.floor(xyz / radii.unsqueeze(-1)).long()
    voxel_keys = (voxel_idx * torch.tensor([73856093, 19349663, 83492791], device=xyz.device)).sum(1)
    unique_keys, inverse, _ = torch.unique(voxel_keys, return_inverse=True, return_counts=True)

    print(f"Merging {len(xyz)} splats into {len(unique_keys)} clusters...")

    mean_xyz = torch_scatter.scatter_mean(xyz, inverse, dim=0)

    dists = torch.norm(xyz - mean_xyz[inverse], dim=1)
    max_dists, _ = torch_scatter.scatter_max(dists, inverse, dim=0)

    diffs = xyz - mean_xyz[inverse]
    outer = diffs.unsqueeze(-1) * diffs.unsqueeze(-2)

    cov_sum = torch_scatter.scatter_sum(outer, inverse, dim=0)
    counts = torch_scatter.scatter_sum(
        torch.ones(xyz.size(0), device=xyz.device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3),
        inverse, dim=0
    )
    cov = cov_sum / counts.clamp_min(1.0)

    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
    cov = cov + torch.eye(3, device=cov.device).unsqueeze(0) * 1e-8
    mask_invalid = torch.isnan(cov).any(dim=(1, 2)) | torch.isinf(cov).any(dim=(1, 2))
    if mask_invalid.any():
        cov[mask_invalid] = torch.eye(3, device=cov.device).unsqueeze(0)

    eigvals_list = []
    chunk = 20000
    for i in range(0, cov.size(0), chunk):
        cov_chunk = cov[i:i+chunk]
        cov_chunk = torch.nan_to_num(cov_chunk, nan=0.0, posinf=1e6, neginf=-1e6)
        cov_chunk = cov_chunk + torch.eye(3, device=cov.device).unsqueeze(0) * 1e-8
        ev, _ = torch.linalg.eigh(cov_chunk)
        eigvals_list.append(ev)
    eigvals = torch.cat(eigvals_list, 0)
    scale_from_cov = torch.sqrt(torch.clamp(eigvals, min=1e-6))

    new_scaling = torch.maximum(
        torch_scatter.scatter_mean(scaling, inverse, dim=0),
        scale_from_cov.max(dim=1).values.unsqueeze(1),
    )
    new_scaling = torch.maximum(new_scaling, max_dists.unsqueeze(1) * 0.5)

    rotation = rotation / (torch.norm(rotation, dim=1, keepdim=True) + 1e-12)
    mean_rot = quaternion_mean_markley(rotation, inverse, len(unique_keys))

    w = opacity.unsqueeze(-1)
    den = torch_scatter.scatter_sum(w, inverse, dim=0).clamp_min(1e-12)
    mean_features_dc = torch_scatter.scatter_sum(features_dc * w, inverse, dim=0) / den

    if gaussian_model.max_sh_degree and gaussian_model.max_sh_degree > 0:
        mean_features_rest = torch_scatter.scatter_sum(features_rest * w, inverse, dim=0) / den
    else:
        mean_features_rest = features_rest

    one_minus_alpha = 1 - opacity.squeeze(1)
    prod = torch_scatter.scatter_mul(one_minus_alpha, inverse, dim=0)
    new_opacity = (1 - prod).unsqueeze(1).clamp(1e-6, 1 - 1e-6)

    gaussian_model._xyz = mean_xyz
    gaussian_model._scaling = gaussian_model.scaling_inverse_activation(new_scaling)
    gaussian_model._rotation = mean_rot
    gaussian_model._features_dc = mean_features_dc
    gaussian_model._features_rest = mean_features_rest
    gaussian_model._opacity = gaussian_model.inverse_opacity_activation(new_opacity)

    return gaussian_model


def progressive_decimate_original(model, target_count, base_radius, growth, max_iter):
    r = float(base_radius)
    for _ in range(int(max_iter)):
        if model._xyz.shape[0] <= target_count:
            break
        decimate_original(r, model, density_aware=False)
        r *= float(growth)
    return model


# ---------------------------
# StageB (New) merge: PCA + medoid center + cover_sigma + inflate
# ---------------------------
def decimate_voxel_pca(model: GS.PointModel, radius: float, cover_sigma: float, inflate: float, min_count_for_pca: int = 3):
    device = model._xyz.device
    xyz = model._xyz
    scaling_act = model.get_scaling
    opacity_act = model.get_opacity
    features_dc = model._features_dc
    features_rest = model._features_rest

    N = xyz.shape[0]
    if N == 0:
        return model

    radii = torch.full((N,), float(radius), device=device)
    voxel_idx = torch.floor(xyz / radii.unsqueeze(-1)).long()
    voxel_keys = (voxel_idx * torch.tensor([73856093, 19349663, 83492791], device=device)).sum(1)
    _, inverse, counts = torch.unique(voxel_keys, return_inverse=True, return_counts=True)
    C = int(counts.numel())

    mean_xyz = torch_scatter.scatter_mean(xyz, inverse, dim=0)
    d_to_mean = torch.norm(xyz - mean_xyz[inverse], dim=1)
    _, argmin = torch_scatter.scatter_min(d_to_mean, inverse, dim=0)
    center_xyz = xyz[argmin]

    dists = torch.norm(xyz - center_xyz[inverse], dim=1)
    max_dists, _ = torch_scatter.scatter_max(dists, inverse, dim=0)

    diffs = xyz - center_xyz[inverse]
    outer = diffs.unsqueeze(-1) * diffs.unsqueeze(-2)
    cov_sum = torch_scatter.scatter_sum(outer, inverse, dim=0)

    ones33 = torch.ones((N, 1, 1), device=device, dtype=torch.float32).expand(-1, 3, 3)
    cnt33 = torch_scatter.scatter_sum(ones33, inverse, dim=0).clamp_min(1.0)
    cov = cov_sum / cnt33

    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
    cov = cov + torch.eye(3, device=device).unsqueeze(0) * 1e-8
    bad = torch.isnan(cov).any(dim=(1, 2)) | torch.isinf(cov).any(dim=(1, 2))
    if bad.any():
        cov[bad] = torch.eye(3, device=device).unsqueeze(0)

    eigvals_list, eigvecs_list = [], []
    chunk = 20000
    for i in range(0, C, chunk):
        cov_chunk = cov[i:i+chunk]
        cov_chunk = torch.nan_to_num(cov_chunk, nan=0.0, posinf=1e6, neginf=-1e6)
        cov_chunk = cov_chunk + torch.eye(3, device=device).unsqueeze(0) * 1e-8
        ev, V = torch.linalg.eigh(cov_chunk)
        eigvals_list.append(ev); eigvecs_list.append(V)
    eigvals = torch.cat(eigvals_list, 0)
    eigvecs = torch.cat(eigvecs_list, 0)

    det = torch.det(eigvecs)
    flip = det < 0
    if flip.any():
        eigvecs[flip, :, 2] *= -1.0

    pca_quat = rotmat_to_quat(eigvecs)

    sigma = torch.sqrt(torch.clamp(eigvals, min=1e-6))
    pca_scale = sigma * float(cover_sigma)

    mean_scaling = torch_scatter.scatter_mean(scaling_act, inverse, dim=0)
    new_scaling = torch.maximum(mean_scaling, pca_scale)
    new_scaling = torch.maximum(new_scaling, max_dists.unsqueeze(1) * 0.9)
    new_scaling = new_scaling * float(inflate)

    if min_count_for_pca and min_count_for_pca > 0:
        small = counts < int(min_count_for_pca)
        if small.any():
            rot_act = model.get_rotation
            rot_act = rot_act / (rot_act.norm(dim=1, keepdim=True) + 1e-12)
            mean_rot = quaternion_mean_markley(rot_act, inverse, C)
            pca_quat[small] = mean_rot[small]
            new_scaling[small] = torch.maximum(mean_scaling[small], max_dists[small].unsqueeze(1) * 0.9) * float(inflate)

    w = opacity_act.unsqueeze(-1)
    denom = torch_scatter.scatter_sum(w, inverse, dim=0).clamp_min(1e-12)
    mean_features_dc = torch_scatter.scatter_sum(features_dc * w, inverse, dim=0) / denom

    if model.max_sh_degree and model.max_sh_degree > 0:
        mean_features_rest = torch_scatter.scatter_sum(features_rest * w, inverse, dim=0) / denom
    else:
        mean_features_rest = features_rest

    one_minus_alpha = 1 - opacity_act.squeeze(1)
    prod = torch_scatter.scatter_mul(one_minus_alpha, inverse, dim=0)
    new_opacity_act = (1 - prod).unsqueeze(1).clamp(1e-6, 1 - 1e-6)

    model._xyz = center_xyz
    model._scaling = model.scaling_inverse_activation(new_scaling)
    model._rotation = pca_quat
    model._features_dc = mean_features_dc
    model._features_rest = mean_features_rest
    model._opacity = model.inverse_opacity_activation(new_opacity_act)
    return model


def progressive_decimate_pca(model: GS.PointModel, target_count: int, base_radius: float, growth: float, max_iter: int, cover_sigma: float, inflate: float):
    r = float(base_radius)
    for _ in range(int(max_iter)):
        if model._xyz.shape[0] <= target_count:
            break
        decimate_voxel_pca(model, r, cover_sigma=cover_sigma, inflate=inflate)
        r *= float(growth)
    return model


# ---------------------------
# Image smooth mask + connected components
# ---------------------------
def to_gray_float01(img_rgb: np.ndarray) -> np.ndarray:
    x = img_rgb.astype(np.float32) / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def conv2d_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    H, W = x.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            out[i, j] = float(np.sum(xp[i:i+kh, j:j+kw] * k))
    return out


def box_blur(x: np.ndarray, ksize: int) -> np.ndarray:
    ksize = int(ksize)
    if ksize < 1:
        return x
    if ksize % 2 == 0:
        ksize += 1
    k = np.ones((ksize, ksize), dtype=np.float32) / float(ksize * ksize)
    return conv2d_same(x, k)


def make_smooth_mask(img_rgb: np.ndarray, smooth_win: int, thr_percentile: float) -> np.ndarray:
    gray = to_gray_float01(img_rgb).astype(np.float32)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / 8.0
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32) / 8.0

    gx = conv2d_same(gray, kx)
    gy = conv2d_same(gray, ky)
    grad = np.sqrt(gx * gx + gy * gy)

    smoothness = box_blur(grad, smooth_win)
    thr = float(np.percentile(smoothness, thr_percentile))
    return smoothness < thr


# ---------------------------
# Assign region_id to points (projection)
# Extrinsic is identity by design (camera at original view).
# ---------------------------
def assign_region_id_to_points_numpy(xyz_np: np.ndarray, K: np.ndarray, region_map: np.ndarray, chunk: int = 300_000):
    """
    xyz_np: (N,3) camera/world (assumed camera coords if E=I)
    region_map: (H,W) int32, 0=non-smooth, 1..R
    return: rid_per_point (N,) int32
    """
    H, W = region_map.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    N = xyz_np.shape[0]
    out = np.zeros((N,), dtype=np.int32)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Xc = xyz_np[s:e].astype(np.float32)
        Z = Xc[:, 2]
        valid = Z > 1e-6
        if not np.any(valid):
            continue

        Xv = Xc[valid]
        Zv = Z[valid]
        u = fx * (Xv[:, 0] / Zv) + cx
        v = fy * (Xv[:, 1] / Zv) + cy

        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)

        in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not np.any(in_img):
            continue

        ui2 = ui[in_img]
        vi2 = vi[in_img]
        rid = region_map[vi2, ui2].astype(np.int32)

        idx_valid = np.nonzero(valid)[0]
        idx_in_img = idx_valid[in_img]
        out[s:e][idx_in_img] = rid

    return out


# ---------------------------
# Write clean vertex-only PLY (no normals, no metadata elements)
# Desired header fields:
# x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0..2 rot_0..3
# ---------------------------
def save_clean_ply_vertex_only(path: str, model: GS.PointModel):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = model._xyz.detach().cpu().numpy().astype(np.float32)
    # features_dc in PointModel is (N,1,3) or (N,?,?) depending on transpose
    # In your load_ply, _features_dc is stored as (N,1,3) after transpose(1,2).
    fdc = model._features_dc.detach().cpu().numpy().astype(np.float32).reshape(xyz.shape[0], -1)
    if fdc.shape[1] != 3:
        # try alternate reshape
        fdc = fdc[:, :3]

    opacity = model._opacity.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
    scale = model._scaling.detach().cpu().numpy().astype(np.float32)
    rot = model._rotation.detach().cpu().numpy().astype(np.float32)

    if scale.shape[1] != 3:
        raise ValueError(f"Expected scale to have 3 components, got {scale.shape}")
    if rot.shape[1] != 4:
        raise ValueError(f"Expected rotation to have 4 components, got {rot.shape}")

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    v = np.empty((xyz.shape[0],), dtype=dtype)

    v["x"] = xyz[:, 0]; v["y"] = xyz[:, 1]; v["z"] = xyz[:, 2]
    v["f_dc_0"] = fdc[:, 0]; v["f_dc_1"] = fdc[:, 1]; v["f_dc_2"] = fdc[:, 2]
    v["opacity"] = opacity[:, 0]
    v["scale_0"] = scale[:, 0]; v["scale_1"] = scale[:, 1]; v["scale_2"] = scale[:, 2]
    v["rot_0"] = rot[:, 0]; v["rot_1"] = rot[:, 1]; v["rot_2"] = rot[:, 2]; v["rot_3"] = rot[:, 3]

    el = PlyElement.describe(v, "vertex")
    PlyData([el], text=False).write(path)


# ---------------------------
# Main pipeline: StageA global -> StageB smooth regions
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="StageA (original) global decimation + StageB (image-smooth guided) PCA decimation (filename camera)."
    )
    parser.add_argument("--path_to_model", type=str, required=True, help="Input Gaussian PLY (vertex-only).")
    parser.add_argument("--save_path", type=str, required=True, help="Output PLY path.")

    # StageA (original)
    parser.add_argument("--stagea_enable", action="store_true", help="Enable StageA (original) global decimation.")
    parser.add_argument("--stagea_radius", type=float, default=0.01, help="StageA base radius.")
    parser.add_argument("--stagea_target_ratio", type=float, default=None,
                        help="If set, progressive StageA to target_count = N*ratio (e.g. 0.5).")
    parser.add_argument("--stagea_growth", type=float, default=1.25)
    parser.add_argument("--stagea_max_iter", type=int, default=12)

    # StageB (image guided)
    parser.add_argument("--stageb_enable", action="store_true", help="Enable StageB (smooth-region PCA) decimation.")
    parser.add_argument("--image_path", type=str, default=None, help="Input image used to build smooth regions.")

    parser.add_argument("--smooth_win", type=int, default=61)
    parser.add_argument("--thr_percentile", type=float, default=25.0)
    parser.add_argument("--min_region_pixels", type=int, default=800)

    parser.add_argument("--base_radius_b", type=float, default=0.003, help="StageB base radius (within smooth regions).")
    parser.add_argument("--growth_b", type=float, default=1.22)
    parser.add_argument("--max_iter_b", type=int, default=16)

    parser.add_argument("--k_small", type=int, default=60)
    parser.add_argument("--k_mid", type=int, default=28)
    parser.add_argument("--k_large", type=int, default=7)
    parser.add_argument("--small_n", type=int, default=5000)
    parser.add_argument("--large_n", type=int, default=50000)
    parser.add_argument("--min_keep", type=int, default=900)

    # StageB merge params
    parser.add_argument("--cover_sigma", type=float, default=1.95, help="PCA sigma coverage (lower=less smooth).")
    parser.add_argument("--inflate", type=float, default=1.04, help="Final scale inflate (lower=less smooth).")

    args = parser.parse_args()

    t0 = time.time()

    base_name, focal_px, W, H = parse_ply_name_camera(args.path_to_model)
    out_path = ensure_output_name(args.save_path, focal_px, W, H)

    print(f"Parsed camera from filename: focal={focal_px}px, size={W}x{H}, base='{base_name}'")
    print("Loading model...")
    model = GS.PointModel()
    model.load_ply(args.path_to_model)
    N0 = int(model._xyz.shape[0])
    print(f"Input points: {N0}")

    # -------- StageA: original --------
    if args.stagea_enable:
        print("\n[StageA] Original global decimation...")
        if args.stagea_target_ratio is not None:
            target = max(1, int(round(N0 * float(args.stagea_target_ratio))))
            print(f"StageA target_count={target} (ratio={args.stagea_target_ratio})")
            progressive_decimate_original(
                model,
                target_count=target,
                base_radius=args.stagea_radius,
                growth=args.stagea_growth,
                max_iter=args.stagea_max_iter,
            )
        else:
            decimate_original(args.stagea_radius, model, density_aware=False)

        print(f"StageA done. Points now: {int(model._xyz.shape[0])}")

    # -------- StageB: smooth regions PCA --------
    if args.stageb_enable:
        if args.image_path is None:
            raise ValueError("--image_path is required when --stageb_enable is set")

        print("\n[StageB] Image-smooth guided PCA decimation...")
        K = build_K_from_filename(focal_px, W, H)

        img = Image.open(args.image_path).convert("RGB").resize((W, H), Image.BILINEAR)
        rgb = np.array(img, dtype=np.uint8)

        smooth = make_smooth_mask(rgb, args.smooth_win, args.thr_percentile)

        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.int32)
        labels, num = cc_label(smooth.astype(np.uint8), structure=structure)

        if num > 0:
            counts_pix = np.bincount(labels.reshape(-1))
            keep_ids = [i for i in range(1, len(counts_pix)) if counts_pix[i] >= args.min_region_pixels]
            keep_mask = np.isin(labels, keep_ids)
            labels = labels * keep_mask
            labels, num = cc_label(labels.astype(np.uint8), structure=structure)

        print(f"Smooth regions (after filter): {num}")

        xyz_np = model._xyz.detach().cpu().numpy().astype(np.float32)
        rid_np = assign_region_id_to_points_numpy(xyz_np, K, labels.astype(np.int32))
        rid = torch.from_numpy(rid_np).to(torch.long).to(model._xyz.device)

        non_idx = torch.nonzero(rid == 0, as_tuple=False).squeeze(1)
        keep_idx_all = [non_idx]
        merged_models = []

        for r in range(1, num + 1):
            idx = torch.nonzero(rid == r, as_tuple=False).squeeze(1)
            n = int(idx.numel())
            if n == 0:
                continue

            if n < args.small_n:
                k = args.k_small
            elif n < args.large_n:
                k = args.k_mid
            else:
                k = args.k_large

            target = max(args.min_keep, int(round(n / float(k))))
            if target >= n:
                keep_idx_all.append(idx)
                continue

            sub = GS.PointModel(sh_degree=model.max_sh_degree)
            sub.max_sh_degree = model.max_sh_degree
            sub._xyz = model._xyz[idx]
            sub._features_dc = model._features_dc[idx]
            sub._features_rest = model._features_rest[idx]
            sub._scaling = model._scaling[idx]
            sub._rotation = model._rotation[idx]
            sub._opacity = model._opacity[idx]

            progressive_decimate_pca(
                sub,
                target_count=target,
                base_radius=args.base_radius_b,
                growth=args.growth_b,
                max_iter=args.max_iter_b,
                cover_sigma=args.cover_sigma,
                inflate=args.inflate,
            )

            merged_models.append(sub)
            print(f"Region {r}: n={n} -> target={target}, out={int(sub._xyz.shape[0])} (k={k})")

        keep_idx = torch.cat(keep_idx_all, dim=0) if keep_idx_all else torch.empty((0,), device=model._xyz.device, dtype=torch.long)

        xyz_out = [model._xyz[keep_idx]]
        scaling_out = [model._scaling[keep_idx]]
        rot_out = [model._rotation[keep_idx]]
        dc_out = [model._features_dc[keep_idx]]
        op_out = [model._opacity[keep_idx]]
        rest_out = [model._features_rest[keep_idx]]

        for sub in merged_models:
            xyz_out.append(sub._xyz)
            scaling_out.append(sub._scaling)
            rot_out.append(sub._rotation)
            dc_out.append(sub._features_dc)
            op_out.append(sub._opacity)
            rest_out.append(sub._features_rest)

        model._xyz = torch.cat(xyz_out, dim=0)
        model._scaling = torch.cat(scaling_out, dim=0)
        model._rotation = torch.cat(rot_out, dim=0)
        model._features_dc = torch.cat(dc_out, dim=0)
        model._opacity = torch.cat(op_out, dim=0)
        model._features_rest = torch.cat(rest_out, dim=0)

        print(f"StageB done. Points now: {int(model._xyz.shape[0])}")

    # -------- Save clean vertex-only PLY (filename camera) --------
    save_clean_ply_vertex_only(out_path, model)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
