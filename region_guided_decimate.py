import argparse
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scipy.ndimage import label as cc_label

import torch
import torch_scatter

import helpers.PointModel as GS

# ---------------- smooth mask ----------------
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
    mask = smoothness < thr
    return mask

# ---------------- PLY meta ----------------
def read_meta(ply: PlyData):
    img_el = ply["image_size"].data
    img_field = img_el.dtype.names[0]
    img_size = np.array(img_el[img_field], dtype=np.int64).reshape(-1)
    W0, H0 = int(img_size[0]), int(img_size[1])

    intr_el = ply["intrinsic"].data
    intr_field = intr_el.dtype.names[0]
    intr = np.array(intr_el[intr_field], dtype=np.float32).reshape(-1)
    K = intr.reshape(3, 3)

    extr_el = ply["extrinsic"].data
    extr_field = extr_el.dtype.names[0]
    extr = np.array(extr_el[extr_field], dtype=np.float32).reshape(-1)
    E = extr.reshape(4, 4)
    return (W0, H0), K, E

def write_ply_with_same_meta(ply_src: PlyData, new_vertex_data, out_path: str):
    vertex_el = PlyElement.describe(new_vertex_data, "vertex")
    other_els = [el for el in ply_src.elements if el.name != "vertex"]
    PlyData([vertex_el] + other_els, text=False).write(out_path)

# ---------------- projection: point -> region_id ----------------
def assign_region_id_to_points(xyz_np, K, E, region_map, chunk=300_000):
    """
    xyz_np: (N,3) float32 world
    region_map: (H,W) int32, 0=non-smooth, 1..R
    return: region_id_per_point int32 shape (N,)  (0 means not in smooth)
    """
    H, W = region_map.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    N = xyz_np.shape[0]
    out = np.zeros((N,), dtype=np.int32)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        X = xyz_np[s:e].astype(np.float32)
        ones = np.ones((X.shape[0], 1), dtype=np.float32)
        Xw = np.concatenate([X, ones], axis=1)          # (M,4)
        Xc = (Xw @ E.T)[:, :3]                          # (M,3)

        Z = Xc[:, 2]
        valid = Z > 1e-6
        if not np.any(valid):
            continue

        Xcv = Xc[valid]
        Zv = Z[valid]

        u = fx * (Xcv[:, 0] / Zv) + cx
        v = fy * (Xcv[:, 1] / Zv) + cy
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

def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions (w, x, y, z).
    R: (B,3,3)
    return: (B,4) normalized
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


# ---------------- decimation core (subset) ----------------
def quaternion_mean_markley(quats, cluster_ids, num_clusters, chunk: int = 20000):
    # Same as你的 decimate.py 版本
    outer = quats.unsqueeze(2) * quats.unsqueeze(1)  # (N,4,4)
    M_sum = torch_scatter.scatter_sum(outer, cluster_ids, dim=0, dim_size=num_clusters)

    ones = torch.ones((quats.size(0),1,1), device=quats.device, dtype=quats.dtype).expand(-1,4,4)
    counts = torch_scatter.scatter_sum(ones, cluster_ids, dim=0, dim_size=num_clusters)
    M = M_sum / counts.clamp_min(1.0)

    M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
    M = M + torch.eye(4, device=M.device, dtype=M.dtype).unsqueeze(0) * 1e-8

    eigvals_list, eigvecs_list = [], []
    for i in range(0, M.size(0), chunk):
        M_chunk = torch.nan_to_num(M[i:i+chunk], nan=0.0, posinf=1e6, neginf=-1e6)
        M_chunk += torch.eye(4, device=M.device, dtype=M.dtype).unsqueeze(0) * 1e-8
        ev, V = torch.linalg.eigh(M_chunk)
        eigvals_list.append(ev); eigvecs_list.append(V)
    eigvals = torch.cat(eigvals_list, 0)
    eigvecs = torch.cat(eigvecs_list, 0)

    max_ids = eigvals.argmax(dim=1)
    mean_quats = eigvecs[torch.arange(num_clusters, device=quats.device), :, max_ids]
    mean_quats = mean_quats / (mean_quats.norm(dim=1, keepdim=True) + 1e-12)
    return mean_quats

def decimate_voxel_medoid(
    model: GS.PointModel,
    radius: float,
    cover_sigma: float = 2.2,
    inflate: float = 1.12,
    min_count_for_pca: int = 3,
):
    """
    Voxel clustering merge.
    Key improvements to reduce holes:
      - Center uses medoid (closest original point to mean).
      - Rotation uses PCA (cov eigenvectors), not mean quaternion.
      - Scale uses sigma coverage (sqrt eigenvals) with cover_sigma + inflate.
      - Robust sanitize + chunked eigh to avoid cusolver invalid-value.
    """
    device = model._xyz.device
    xyz = model._xyz                                  # (N,3)
    scaling_act = model.get_scaling                   # (N,3) activated
    features_dc = model._features_dc                  # (N,1,3)
    features_rest = model._features_rest              # (N,?,3) maybe empty for sh0
    opacity_act = model.get_opacity                   # (N,1) in [0,1]

    N = xyz.shape[0]
    if N == 0:
        return model

    # --- voxel keys ---
    radii = torch.full((N,), float(radius), device=device)
    voxel_idx = torch.floor(xyz / radii.unsqueeze(-1)).long()
    voxel_keys = (voxel_idx * torch.tensor([73856093, 19349663, 83492791], device=device)).sum(1)
    unique_keys, inverse, counts = torch.unique(voxel_keys, return_inverse=True, return_counts=True)
    C = unique_keys.shape[0]

    # --- mean center ---
    mean_xyz = torch_scatter.scatter_mean(xyz, inverse, dim=0)  # (C,3)

    # --- medoid center (closest original point to mean) ---
    d_to_mean = torch.norm(xyz - mean_xyz[inverse], dim=1)      # (N,)
    dmin, argmin = torch_scatter.scatter_min(d_to_mean, inverse, dim=0)  # argmin indexes into N
    center_xyz = xyz[argmin]                                     # (C,3)

    # --- per cluster max distance to center (coverage lower bound) ---
    dists = torch.norm(xyz - center_xyz[inverse], dim=1)
    max_dists, _ = torch_scatter.scatter_max(dists, inverse, dim=0)      # (C,)

    # --- covariance around center (for PCA) ---
    diffs = xyz - center_xyz[inverse]                                   # (N,3)
    outer = diffs.unsqueeze(-1) * diffs.unsqueeze(-2)                   # (N,3,3)
    cov_sum = torch_scatter.scatter_sum(outer, inverse, dim=0)          # (C,3,3)

    ones33 = torch.ones((N,1,1), device=device, dtype=torch.float32).expand(-1,3,3)
    cnt33 = torch_scatter.scatter_sum(ones33, inverse, dim=0).clamp_min(1.0)
    cov = cov_sum / cnt33                                               # (C,3,3)

    # sanitize
    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
    cov = cov + torch.eye(3, device=device).unsqueeze(0) * 1e-8
    mask_invalid = torch.isnan(cov).any(dim=(1,2)) | torch.isinf(cov).any(dim=(1,2))
    if mask_invalid.any():
        cov[mask_invalid] = torch.eye(3, device=device).unsqueeze(0)

    # chunked eigh (get eigvals & eigvecs)
    eigvals_list, eigvecs_list = [], []
    chunk = 20000
    for i in range(0, C, chunk):
        cov_chunk = cov[i:i+chunk]
        cov_chunk = torch.nan_to_num(cov_chunk, nan=0.0, posinf=1e6, neginf=-1e6)
        cov_chunk = cov_chunk + torch.eye(3, device=device).unsqueeze(0) * 1e-8
        ev, V = torch.linalg.eigh(cov_chunk)
        eigvals_list.append(ev)
        eigvecs_list.append(V)
    eigvals = torch.cat(eigvals_list, dim=0)   # (C,3)
    eigvecs = torch.cat(eigvecs_list, dim=0)   # (C,3,3)

    # --- PCA rotation (convert eigvecs -> quat) ---
    # Note: eigenvectors are orthonormal, but may be left-handed (det=-1). Fix it.
    det = torch.det(eigvecs)
    flip = det < 0
    if flip.any():
        eigvecs[flip, :, 2] *= -1.0  # flip one axis to make det positive

    pca_quat = rotmat_to_quat(eigvecs)  # (C,4)

    # --- scale from eigenvalues (sigma coverage) ---
    sigma = torch.sqrt(torch.clamp(eigvals, min=1e-6))           # (C,3)
    pca_scale = sigma * float(cover_sigma)                       # (C,3)

    mean_scaling = torch_scatter.scatter_mean(scaling_act, inverse, dim=0)  # (C,3)

    # combine conservative estimates
    new_scaling = torch.maximum(mean_scaling, pca_scale)

    # also enforce a coverage lower bound from max distance (tune factor)
    new_scaling = torch.maximum(new_scaling, max_dists.unsqueeze(1) * 0.9)

    # inflate to reduce holes further
    new_scaling = new_scaling * float(inflate)

    # if cluster has too few points, PCA is not reliable: fall back to mean rotation & mean scaling
    # (but still keep coverage lower bound)
    if min_count_for_pca is not None and min_count_for_pca > 0:
        small = counts < int(min_count_for_pca)
        if small.any():
            # keep original mean rotation (Markley) for small clusters (optional but safer)
            # We compute it only once; if you want, you can skip and just keep identity.
            # Here: reuse your existing quat mean function:
            # model.get_rotation gives normalized; we can average within cluster:
            rotation_act = model.get_rotation
            rotation_act = rotation_act / (rotation_act.norm(dim=1, keepdim=True) + 1e-12)
            mean_rot_small = quaternion_mean_markley(rotation_act, inverse, C)
            pca_quat[small] = mean_rot_small[small]

            new_scaling[small] = torch.maximum(mean_scaling[small], max_dists[small].unsqueeze(1) * 0.9) * float(inflate)

    # --- features (opacity-weighted mean) ---
    w = opacity_act.unsqueeze(-1)  # (N,1,1) broadcast w/ features
    mean_features_dc = torch_scatter.scatter_sum(features_dc * w, inverse, dim=0) / \
                       torch_scatter.scatter_sum(w, inverse, dim=0).clamp_min(1e-12)

    if model.max_sh_degree and model.max_sh_degree > 0:
        mean_features_rest = torch_scatter.scatter_sum(features_rest * w, inverse, dim=0) / \
                             torch_scatter.scatter_sum(w, inverse, dim=0).clamp_min(1e-12)
    else:
        mean_features_rest = features_rest  # likely empty

    # --- opacity fusion (cover-like) ---
    one_minus_alpha = 1 - opacity_act.squeeze(1)                        # (N,)
    prod = torch_scatter.scatter_mul(one_minus_alpha, inverse, dim=0)   # (C,)
    new_opacity_act = (1 - prod).unsqueeze(1)                           # (C,1)

    # --- update model params ---
    model._xyz = center_xyz
    model._scaling = model.scaling_inverse_activation(new_scaling)
    model._rotation = pca_quat
    model._features_dc = mean_features_dc
    model._features_rest = mean_features_rest
    model._opacity = model.inverse_opacity_activation(new_opacity_act.clamp(1e-6, 1-1e-6))

    return model


def progressive_decimate_subset(sub_model: GS.PointModel, target_count: int, base_radius: float, growth: float, max_iter: int):
    r = float(base_radius)
    for it in range(max_iter):
        if sub_model._xyz.shape[0] <= target_count:
            break
        decimate_voxel_medoid(sub_model, r)
        r *= float(growth)
    return sub_model

def make_submodel_from_indices(model: GS.PointModel, idx: torch.Tensor) -> GS.PointModel:
    sub = GS.PointModel(sh_degree=model.max_sh_degree)
    sub.max_sh_degree = model.max_sh_degree
    sub._xyz = model._xyz[idx]
    sub._features_dc = model._features_dc[idx]
    sub._features_rest = model._features_rest[idx]
    sub._scaling = model._scaling[idx]
    sub._rotation = model._rotation[idx]
    sub._opacity = model._opacity[idx]
    return sub

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out_ply", required=True)

    ap.add_argument("--use_inv_extrinsic", action="store_true")

    # smooth mask params
    ap.add_argument("--smooth_win", type=int, default=61)
    ap.add_argument("--thr_percentile", type=float, default=25.0)
    ap.add_argument("--min_region_pixels", type=int, default=800)

    # decimation control
    ap.add_argument("--base_radius", type=float, default=0.01, help="base radius for region decimation")
    ap.add_argument("--growth", type=float, default=1.25)
    ap.add_argument("--max_iter", type=int, default=8)

    # points-per-merge control (how aggressive per region)
    ap.add_argument("--k_small", type=int, default=80, help="small region: roughly n/k points kept")
    ap.add_argument("--k_mid", type=int, default=40)
    ap.add_argument("--k_large", type=int, default=20)
    ap.add_argument("--small_n", type=int, default=5000)
    ap.add_argument("--large_n", type=int, default=50000)
    ap.add_argument("--min_keep", type=int, default=200, help="minimum points to keep per region")

    ap.add_argument("--chunk", type=int, default=300000)
    args = ap.parse_args()

    # load ply + meta
    ply = PlyData.read(args.ply)
    (W0, H0), K, E = read_meta(ply)
    if args.use_inv_extrinsic:
        E = np.linalg.inv(E)

    # load image aligned to ply image_size
    img = Image.open(args.image).convert("RGB").resize((W0, H0), Image.BILINEAR)
    rgb = np.array(img, dtype=np.uint8)

    # smooth mask -> connected components
    smooth_mask = make_smooth_mask(rgb, args.smooth_win, args.thr_percentile)
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=np.int32)
    labels, num = cc_label(smooth_mask.astype(np.uint8), structure=structure)

    # filter tiny regions by pixel area
    if num > 0:
        pix_counts = np.bincount(labels.reshape(-1))
        keep_ids = [i for i in range(1, len(pix_counts)) if pix_counts[i] >= args.min_region_pixels]
        keep_mask = np.isin(labels, keep_ids)
        labels = labels * keep_mask
        labels, num = cc_label(labels.astype(np.uint8), structure=structure)

    print(f"Smooth regions (after filter): {num}")

    # point -> region_id
    vtx = ply["vertex"].data
    xyz_np = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1).astype(np.float32)
    rid_per_point = assign_region_id_to_points(xyz_np, K, E, labels.astype(np.int32), chunk=args.chunk)
    rid_per_point_t = torch.from_numpy(rid_per_point).to(torch.long)

    # load into PointModel (vertex params only)
    model = GS.PointModel(sh_degree=0)
    model.load_ply(args.ply)

    device = model._xyz.device

    # collect outputs
    keep_idx_all = []
    merged_models = []

    # non-smooth points stay
    non_smooth_idx = torch.nonzero(rid_per_point_t == 0, as_tuple=False).squeeze(1).to(device)
    keep_idx_all.append(non_smooth_idx)

    # per region
    for rid in range(1, num + 1):
        idx = torch.nonzero(rid_per_point_t == rid, as_tuple=False).squeeze(1).to(device)
        n = int(idx.numel())
        if n == 0:
            continue

        # choose aggressiveness by region point count
        if n < args.small_n:
            k = args.k_small
        elif n < args.large_n:
            k = args.k_mid
        else:
            k = args.k_large

        target = max(args.min_keep, int(round(n / float(k))))
        if target >= n:
            # no need to merge
            keep_idx_all.append(idx)
            continue

        sub = make_submodel_from_indices(model, idx)
        sub = progressive_decimate_subset(sub, target, args.base_radius, args.growth, args.max_iter)
        merged_models.append(sub)

        print(f"Region {rid}: n={n} -> target={target}, out={sub._xyz.shape[0]} (k={k})")

    # build final model tensors
    keep_idx = torch.cat(keep_idx_all, dim=0) if keep_idx_all else torch.empty((0,), device=device, dtype=torch.long)

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

    # save vertex-only to temp, then copy meta back
    tmp_out = args.out_ply + ".tmp_vertex_only.ply"
    model.save_ply(tmp_out)
    ply_tmp = PlyData.read(tmp_out)

    write_ply_with_same_meta(ply, ply_tmp["vertex"].data, args.out_ply)
    print("Wrote:", args.out_ply)

if __name__ == "__main__":
    main()



# python region_guided_decimate.py \
#   --image /home/zheng/Gaussian_Decimator/data/test1.png \
#   --ply /home/zheng/Gaussian_Decimator/data/test1.ply \
#   --out_ply /home/zheng/Gaussian_Decimator/output/region_decimated.ply \
#   --use_inv_extrinsic \
#   --base_radius 0.01 \
#   --smooth_win 61 \
#   --thr_percentile 25
