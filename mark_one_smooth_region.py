import argparse
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scipy.ndimage import label as cc_label


# ---------- smooth mask (same idea as你之前的脚本：Sobel + 大窗口平均) ----------
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


def make_smooth_mask(img_rgb: np.ndarray, smooth_win: int = 61, thr_percentile: float = 25.0) -> np.ndarray:
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


# ---------- read meta & project ----------
def read_meta(ply: PlyData):
    img_el = ply["image_size"].data
    img_field = img_el.dtype.names[0]
    img_size = np.array(img_el[img_field], dtype=np.int64).reshape(-1)
    if img_size.size != 2:
        raise ValueError(f"image_size has {img_size.size} values, expected 2")
    W0, H0 = int(img_size[0]), int(img_size[1])

    intr_el = ply["intrinsic"].data
    intr_field = intr_el.dtype.names[0]
    intr = np.array(intr_el[intr_field], dtype=np.float32).reshape(-1)
    if intr.size != 9:
        raise ValueError(f"intrinsic has {intr.size} floats, expected 9")
    K = intr.reshape(3, 3)

    extr_el = ply["extrinsic"].data
    extr_field = extr_el.dtype.names[0]
    extr = np.array(extr_el[extr_field], dtype=np.float32).reshape(-1)
    if extr.size != 16:
        raise ValueError(f"extrinsic has {extr.size} floats, expected 16")
    E = extr.reshape(4, 4)

    return (W0, H0), K, E


def compute_point_mask_by_region(xyz, K, E, region_map, target_region_id: int, chunk=300_000):
    """
    xyz: (N,3)
    region_map: (H,W) int32, 0=non-smooth, 1..num regions
    returns boolean mask (N,) selecting points that project into target region
    """
    H, W = region_map.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    N = xyz.shape[0]
    out = np.zeros((N,), dtype=bool)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        X = xyz[s:e].astype(np.float32)

        ones = np.ones((X.shape[0], 1), dtype=np.float32)
        Xw = np.concatenate([X, ones], axis=1)  # (M,4)
        Xc = (Xw @ E.T)[:, :3]  # (M,3)

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
        hit = (region_map[vi2, ui2] == target_region_id)

        idx_valid = np.nonzero(valid)[0]
        idx_in_img = idx_valid[in_img]
        idx_sel = idx_in_img[hit]

        out[s:e][idx_sel] = True

    return out


def write_ply_with_same_meta(ply_src: PlyData, new_vertex_data, out_path: str):
    vertex_el = PlyElement.describe(new_vertex_data, "vertex")
    other_els = [el for el in ply_src.elements if el.name != "vertex"]
    out = PlyData([vertex_el] + other_els, text=False)
    out.write(out_path)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out_vis", default="regions_vis.png", help="region visualization png")
    ap.add_argument("--out_marked_image", default="marked_region.png")
    ap.add_argument("--out_marked_ply", default="marked_region.ply")
    ap.add_argument("--smooth_win", type=int, default=61)
    ap.add_argument("--thr_percentile", type=float, default=25.0)
    ap.add_argument("--min_region_pixels", type=int, default=800, help="discard tiny regions in image")
    ap.add_argument("--region_id", type=int, default=None, help="which region id (from vis) to mark")
    ap.add_argument("--auto_largest", action="store_true", help="auto pick the largest region")
    ap.add_argument("--use_inv_extrinsic", action="store_true")
    ap.add_argument("--set_opacity", action="store_true")
    ap.add_argument("--chunk", type=int, default=300000)
    args = ap.parse_args()

    ply = PlyData.read(args.ply)
    (W0, H0), K, E = read_meta(ply)
    if args.use_inv_extrinsic:
        E = np.linalg.inv(E)

    # resize image to PLY image_size so pixels align
    img = Image.open(args.image).convert("RGB").resize((W0, H0), Image.BILINEAR)
    rgb = np.array(img, dtype=np.uint8)

    # smooth mask + connected components
    smooth_mask = make_smooth_mask(rgb, smooth_win=args.smooth_win, thr_percentile=args.thr_percentile)
    # 4-neighborhood CC
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=np.int32)
    labels, num = cc_label(smooth_mask.astype(np.uint8), structure=structure)

    # remove tiny regions
    if num > 0:
        counts = np.bincount(labels.reshape(-1))
        # counts[0] is background
        keep_ids = [i for i in range(1, len(counts)) if counts[i] >= args.min_region_pixels]
        keep_mask = np.isin(labels, keep_ids)
        labels = labels * keep_mask
        # re-label compactly to 1..K
        labels2, num2 = cc_label(labels.astype(np.uint8), structure=structure)
        labels = labels2
        num = num2

    print(f"Found smooth regions: {num} (after filtering tiny < {args.min_region_pixels} pixels)")

    # region stats
    if num > 0:
        counts = np.bincount(labels.reshape(-1))
        region_info = [(rid, int(counts[rid])) for rid in range(1, len(counts))]
        region_info.sort(key=lambda x: x[1], reverse=True)
        print("Top regions by pixel count (region_id, pixels):")
        for rid, c in region_info[:10]:
            print(f"  {rid}: {c}")

    # choose region
    if args.auto_largest:
        if num == 0:
            raise RuntimeError("No smooth regions found.")
        counts = np.bincount(labels.reshape(-1))
        rid = int(np.argmax(counts[1:]) + 1)
        target_rid = rid
        print("Auto picked largest region_id:", target_rid)
    else:
        if args.region_id is None:
            print("You did not provide --region_id. Please open the visualization and pick one, then rerun.")
            target_rid = None
        else:
            target_rid = int(args.region_id)

    # visualization: random color per region
    rng = np.random.default_rng(0)
    vis = rgb.copy().astype(np.float32)
    if num > 0:
        colors = rng.integers(0, 255, size=(num + 1, 3), dtype=np.uint8)
        colors[0] = np.array([0, 0, 0], dtype=np.uint8)
        overlay = colors[labels]  # HWC
        # alpha blend
        alpha = 0.45
        vis = (vis * (1 - alpha) + overlay.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(vis.astype(np.uint8)).save(args.out_vis)
    print("Wrote region visualization:", args.out_vis)

    if target_rid is None:
        return

    # mark image region to black
    marked_img = rgb.copy()
    marked_img[labels == target_rid] = 0
    Image.fromarray(marked_img).save(args.out_marked_image)
    print("Wrote marked image:", args.out_marked_image)

    # mark ply points
    vtx = ply["vertex"].data
    xyz = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1).astype(np.float32)

    point_mask = compute_point_mask_by_region(xyz, K, E, labels.astype(np.int32), target_rid, chunk=args.chunk)
    sel_n = int(point_mask.sum())
    print(f"Selected points projecting into region {target_rid}: {sel_n} / {xyz.shape[0]}")

    vtx2 = vtx.copy()
    vtx2["f_dc_0"][point_mask] = 0.0
    vtx2["f_dc_1"][point_mask] = 0.0
    vtx2["f_dc_2"][point_mask] = 0.0
    if args.set_opacity:
        vtx2["opacity"][point_mask] = np.maximum(vtx2["opacity"][point_mask], 0.95)

    write_ply_with_same_meta(ply, vtx2, args.out_marked_ply)
    print("Wrote marked ply:", args.out_marked_ply)


if __name__ == "__main__":
    main()


# python mark_one_smooth_region.py \
#   --image /home/zheng/Gaussian_Decimator/data/test1.png \
#   --ply /home/zheng/Gaussian_Decimator/data/test1.ply \
#   --out_vis /home/zheng/Gaussian_Decimator/output/regions_vis.png \
#   --use_inv_extrinsic


# python mark_one_smooth_region.py \
#   --image /home/zheng/Gaussian_Decimator/data/test1.png \
#   --ply /home/zheng/Gaussian_Decimator/data/test1.ply \
#   --out_vis /home/zheng/Gaussian_Decimator/output/regions_vis.png \
#   --out_marked_image /home/zheng/Gaussian_Decimator/output/marked_region.png \
#   --out_marked_ply /home/zheng/Gaussian_Decimator/output/marked_region.ply \
#   --region_id 3 \
#   --set_opacity \
#   --use_inv_extrinsic
