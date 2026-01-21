# batch_decimate.py
# Gaussian Splat Decimation Tool (Batch)
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
      {index}_{focal}_{W}_{H}.ply  (your batch case)
      or {base}_{focal}_{W}_{H}.ply
    We parse the LAST 3 underscore-separated tokens of the stem as integers.
    Returns: base_name(str), focal(int), W(int), H(int)
    """
    p = Path(ply_path)
    stem = p.stem
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

    base = "_".join(parts[:-3]) or "model"
    return base, focal, W, H


def parse_index_from_ply_name(ply_path: str) -> str:
    """
    Your naming rule: {index}_{focal}_{W}_{H}.ply
    We take the FIRST token as index (string).
    """
    p = Path(ply_path)
    parts = p.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Bad ply name: {p.name}")
    return parts[0]  # keep as string (e.g. "0")


def build_K_from_filename(focal_px: int, W: int, H: int) -> np.ndarray:
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

    print(f"  Merging {len(xyz)} splats into {len(unique_keys)} clusters...")

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

    # medoid center
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
# Assign region_id to points (projection, extrinsic=I)
# ---------------------------
def assign_region_id_to_points_numpy(xyz_np: np.ndarray, K: np.ndarray, region_map: np.ndarray, chunk: int = 300_000):
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
# ---------------------------
def save_clean_ply_vertex_only(path: str, model: GS.PointModel):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = model._xyz.detach().cpu().numpy().astype(np.float32)
    fdc = model._features_dc.detach().cpu().numpy().astype(np.float32).reshape(xyz.shape[0], -1)
    if fdc.shape[1] < 3:
        raise ValueError(f"_features_dc shape looks wrong: {fdc.shape}")
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
# Utility: clone model (avoid re-loading 3 times)
# ---------------------------
def clone_model(src: GS.PointModel) -> GS.PointModel:
    dst = GS.PointModel(sh_degree=src.max_sh_degree)
    dst.max_sh_degree = src.max_sh_degree
    # Tensors: clone to detach storage
    dst._xyz = src._xyz.clone()
    dst._features_dc = src._features_dc.clone()
    dst._features_rest = src._features_rest.clone() if src._features_rest.numel() > 0 else src._features_rest
    dst._scaling = src._scaling.clone()
    dst._rotation = src._rotation.clone()
    dst._opacity = src._opacity.clone()
    return dst


def find_image_for_index(img_dir: str, idx: str) -> str:
    """
    Find image named {idx}.png or {idx}.jpg/jpeg (case-insensitive).
    """
    d = Path(img_dir)
    candidates = [
        d / f"{idx}.png",
        d / f"{idx}.jpg",
        d / f"{idx}.jpeg",
        d / f"{idx}.PNG",
        d / f"{idx}.JPG",
        d / f"{idx}.JPEG",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # fallback: any file starting with idx + "." (rare)
    for p in d.glob(f"{idx}.*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            return str(p)
    return ""


# ---------------------------
# Stage runners
# ---------------------------
def run_stageA(model: GS.PointModel, stagea_radius: float):
    print(f"  [A] radius={stagea_radius}")
    decimate_original(stagea_radius, model, density_aware=False)
    return model


def run_stageB(model: GS.PointModel, image_path: str, focal_px: int, W: int, H: int,
               smooth_win: int, thr_percentile: float, min_region_pixels: int,
               base_radius_b: float, growth_b: float, max_iter_b: int,
               k_small: int, k_mid: int, k_large: int, small_n: int, large_n: int, min_keep: int,
               cover_sigma: float, inflate: float):
    print(f"  [B] image={Path(image_path).name}, base_radius_b={base_radius_b}, cover_sigma={cover_sigma}, inflate={inflate}")

    K = build_K_from_filename(focal_px, W, H)

    img = Image.open(image_path).convert("RGB").resize((W, H), Image.BILINEAR)
    rgb = np.array(img, dtype=np.uint8)

    smooth = make_smooth_mask(rgb, smooth_win, thr_percentile)

    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.int32)
    labels, num = cc_label(smooth.astype(np.uint8), structure=structure)

    # filter tiny regions
    if num > 0:
        counts_pix = np.bincount(labels.reshape(-1))
        keep_ids = [i for i in range(1, len(counts_pix)) if counts_pix[i] >= min_region_pixels]
        keep_mask = np.isin(labels, keep_ids)
        labels = labels * keep_mask
        labels, num = cc_label(labels.astype(np.uint8), structure=structure)

    print(f"    Smooth regions: {num}")

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

        if n < small_n:
            k = k_small
        elif n < large_n:
            k = k_mid
        else:
            k = k_large

        target = max(min_keep, int(round(n / float(k))))
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
            base_radius=base_radius_b,
            growth=growth_b,
            max_iter=max_iter_b,
            cover_sigma=cover_sigma,
            inflate=inflate,
        )

        merged_models.append(sub)
        print(f"    Region {r}: n={n} -> target={target}, out={int(sub._xyz.shape[0])} (k={k})")

    keep_idx = torch.cat(keep_idx_all, dim=0) if keep_idx_all else torch.empty((0,), device=model._xyz.device, dtype=torch.long)

    # concat final tensors
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

    return model


# ---------------------------
# Batch main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch run A-only / B-only / AB on folders of PLY + images.")

    ap.add_argument("--ply_dir", required=True, help="Folder with PLYs named like 0_f_W_H.ply")
    ap.add_argument("--img_dir", required=True, help="Folder with images named like 0.png / 0.jpg")
    ap.add_argument("--out_dir", required=True, help="Output root folder. Will create A_ply, B_ply, AB_ply.")
    ap.add_argument("--device", default=None, help="Force device: cuda or cpu. Default: auto.")

    # A params
    ap.add_argument("--stagea_radius", type=float, default=0.003, help="A radius (default 0.003)")

    # B params (defaults = your current tuned values)
    ap.add_argument("--smooth_win", type=int, default=61)
    ap.add_argument("--thr_percentile", type=float, default=25.0)
    ap.add_argument("--min_region_pixels", type=int, default=800)

    ap.add_argument("--base_radius_b", type=float, default=0.003)
    ap.add_argument("--growth_b", type=float, default=1.22)
    ap.add_argument("--max_iter_b", type=int, default=16)

    ap.add_argument("--k_small", type=int, default=60)
    ap.add_argument("--k_mid", type=int, default=28)
    ap.add_argument("--k_large", type=int, default=7)
    ap.add_argument("--small_n", type=int, default=5000)
    ap.add_argument("--large_n", type=int, default=50000)
    ap.add_argument("--min_keep", type=int, default=900)

    ap.add_argument("--cover_sigma", type=float, default=1.95)
    ap.add_argument("--inflate", type=float, default=1.04)

    ap.add_argument("--ext", default=".ply", help="PLY extension filter, default .ply")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N files (debug).")

    args = ap.parse_args()

    # device control
    if args.device is not None:
        if args.device not in ["cpu", "cuda"]:
            raise ValueError("--device must be cpu or cuda")
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ply_dir = Path(args.ply_dir)
    img_dir = Path(args.img_dir)
    out_root = Path(args.out_dir)

    outA = out_root / "A_ply"
    outB = out_root / "B_ply"
    outAB = out_root / "AB_ply"
    outA.mkdir(parents=True, exist_ok=True)
    outB.mkdir(parents=True, exist_ok=True)
    outAB.mkdir(parents=True, exist_ok=True)

    ply_paths = sorted([p for p in ply_dir.iterdir() if p.is_file() and p.suffix.lower() == args.ext.lower()])
    if args.limit and args.limit > 0:
        ply_paths = ply_paths[: int(args.limit)]

    print(f"Found {len(ply_paths)} PLY files in {ply_dir}")

    t_all = time.time()
    n_ok = 0
    n_skip = 0

    for i, ply_path in enumerate(ply_paths, 1):
        stem = ply_path.stem
        try:
            idx = parse_index_from_ply_name(str(ply_path))
            base_name, focal_px, W, H = parse_ply_name_camera(str(ply_path))
        except Exception as e:
            print(f"[{i}/{len(ply_paths)}] SKIP {ply_path.name}: cannot parse name ({e})")
            n_skip += 1
            continue

        img_path = find_image_for_index(str(img_dir), idx)
        if img_path == "":
            print(f"[{i}/{len(ply_paths)}] SKIP {ply_path.name}: missing image for index {idx} in {img_dir}")
            n_skip += 1
            continue

        print(f"\n[{i}/{len(ply_paths)}] {ply_path.name}  <->  {Path(img_path).name}")
        print(f"  Camera: focal={focal_px}, W={W}, H={H}")

        try:
            # load once
            model0 = GS.PointModel()
            model0.load_ply(str(ply_path))
            # PointModel chooses device internally; optionally move to forced device
            if device == "cpu" and model0._xyz.is_cuda:
                model0._xyz = model0._xyz.cpu()
                model0._features_dc = model0._features_dc.cpu()
                model0._features_rest = model0._features_rest.cpu()
                model0._scaling = model0._scaling.cpu()
                model0._rotation = model0._rotation.cpu()
                model0._opacity = model0._opacity.cpu()
            elif device == "cuda" and (not model0._xyz.is_cuda):
                model0._xyz = model0._xyz.cuda()
                model0._features_dc = model0._features_dc.cuda()
                model0._features_rest = model0._features_rest.cuda()
                model0._scaling = model0._scaling.cuda()
                model0._rotation = model0._rotation.cuda()
                model0._opacity = model0._opacity.cuda()

            # ---------- A-only ----------
            mA = clone_model(model0)
            run_stageA(mA, args.stagea_radius)
            save_clean_ply_vertex_only(str(outA / ply_path.name), mA)
            print(f"  -> wrote A:  {outA / ply_path.name}")

            # ---------- B-only ----------
            mB = clone_model(model0)
            run_stageB(
                mB, img_path, focal_px, W, H,
                args.smooth_win, args.thr_percentile, args.min_region_pixels,
                args.base_radius_b, args.growth_b, args.max_iter_b,
                args.k_small, args.k_mid, args.k_large, args.small_n, args.large_n, args.min_keep,
                args.cover_sigma, args.inflate
            )
            save_clean_ply_vertex_only(str(outB / ply_path.name), mB)
            print(f"  -> wrote B:  {outB / ply_path.name}")

            # ---------- AB ----------
            mAB = clone_model(model0)
            run_stageA(mAB, args.stagea_radius)
            run_stageB(
                mAB, img_path, focal_px, W, H,
                args.smooth_win, args.thr_percentile, args.min_region_pixels,
                args.base_radius_b, args.growth_b, args.max_iter_b,
                args.k_small, args.k_mid, args.k_large, args.small_n, args.large_n, args.min_keep,
                args.cover_sigma, args.inflate
            )
            save_clean_ply_vertex_only(str(outAB / ply_path.name), mAB)
            print(f"  -> wrote AB: {outAB / ply_path.name}")

            n_ok += 1

        except Exception as e:
            print(f"[{i}/{len(ply_paths)}] ERROR processing {ply_path.name}: {e}")
            n_skip += 1

        # optional: free GPU cache between samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")
    print(f"OK: {n_ok}, Skipped/Failed: {n_skip}")
    print(f"Total time: {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
