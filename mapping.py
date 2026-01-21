import argparse
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def read_meta(ply: PlyData):
    # image_size
    img_el = ply["image_size"].data
    img_field = img_el.dtype.names[0]
    img_size = np.array(img_el[img_field], dtype=np.int64).reshape(-1)
    if img_size.size != 2:
        raise ValueError(f"image_size has {img_size.size} values, expected 2")
    W0, H0 = int(img_size[0]), int(img_size[1])

    # intrinsic (3x3)
    intr_el = ply["intrinsic"].data
    intr_field = intr_el.dtype.names[0]
    intr = np.array(intr_el[intr_field], dtype=np.float32).reshape(-1)
    if intr.size != 9:
        raise ValueError(f"intrinsic has {intr.size} floats, expected 9")
    K = intr.reshape(3, 3)

    # extrinsic (4x4)
    extr_el = ply["extrinsic"].data
    extr_field = extr_el.dtype.names[0]
    extr = np.array(extr_el[extr_field], dtype=np.float32).reshape(-1)
    if extr.size != 16:
        raise ValueError(f"extrinsic has {extr.size} floats, expected 16")
    E = extr.reshape(4, 4)

    return (W0, H0), K, E


def project_mask_for_rect(xyz, K, E, W, H, rect, chunk=300_000):
    """
    xyz: (N,3) float32
    rect: (x0,y0,x1,y1) in pixel coords on W x H image
    Returns: boolean mask (N,) where point projects into rect and is valid.
    """
    x0, y0, x1, y1 = rect
    N = xyz.shape[0]
    out = np.zeros((N,), dtype=bool)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # chunked for memory
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        X = xyz[s:e]  # (M,3)
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

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_img):
            continue

        u2 = u[in_img]
        v2 = v[in_img]
        in_rect = (u2 >= x0) & (u2 < x1) & (v2 >= y0) & (v2 < y1)

        # Map back to original indices
        idx_valid = np.nonzero(valid)[0]
        idx_img = idx_valid[in_img]
        idx_sel = idx_img[in_rect]

        out[s:e][idx_sel] = True

    return out


def write_ply_with_same_meta(ply_src: PlyData, new_vertex_data, out_path: str):
    vertex_el = PlyElement.describe(new_vertex_data, "vertex")
    other_els = [el for el in ply_src.elements if el.name != "vertex"]
    out = PlyData([vertex_el] + other_els, text=False)
    out.write(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--ply", required=True, help="input ply path (with metadata)")
    ap.add_argument("--rect", required=True, help="x0,y0,x1,y1 in pixels (on PLY image_size)")
    ap.add_argument("--out_image", default="marked.png")
    ap.add_argument("--out_ply", default="marked.ply")
    ap.add_argument("--use_inv_extrinsic", action="store_true",
                    help="use inv(extrinsic) for projection if your E direction is opposite")
    ap.add_argument("--set_opacity", action="store_true",
                    help="also set selected points opacity high (more visible)")
    ap.add_argument("--chunk", type=int, default=300000)
    args = ap.parse_args()

    ply = PlyData.read(args.ply)
    (W0, H0), K, E = read_meta(ply)

    if args.use_inv_extrinsic:
        E = np.linalg.inv(E)

    # Load and resize image to PLY image size so coordinates match
    img = Image.open(args.image).convert("RGB")
    img_rs = img.resize((W0, H0), Image.BILINEAR)
    img_arr = np.array(img_rs)

    x0, y0, x1, y1 = map(int, args.rect.split(","))
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    x0 = max(0, min(W0, x0)); x1 = max(0, min(W0, x1))
    y0 = max(0, min(H0, y0)); y1 = max(0, min(H0, y1))
    rect = (x0, y0, x1, y1)

    # Mark image region to black
    img_arr[y0:y1, x0:x1, :] = 0
    Image.fromarray(img_arr).save(args.out_image)
    print("Wrote marked image:", args.out_image, f"(size {W0}x{H0})")

    # Read xyz
    vtx = ply["vertex"].data
    xyz = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1).astype(np.float32)

    # Compute mask: points projecting into rect
    mask = project_mask_for_rect(xyz, K, E, W0, H0, rect, chunk=args.chunk)
    sel_n = int(mask.sum())
    print(f"Selected points in rect: {sel_n} / {xyz.shape[0]}")

    # Modify vertex colors (f_dc -> black)
    vtx2 = vtx.copy()
    if "f_dc_0" in vtx2.dtype.names and "f_dc_1" in vtx2.dtype.names and "f_dc_2" in vtx2.dtype.names:
        vtx2["f_dc_0"][mask] = 0.0
        vtx2["f_dc_1"][mask] = 0.0
        vtx2["f_dc_2"][mask] = 0.0
    else:
        raise ValueError("vertex does not have f_dc_0/1/2 fields")

    # Optional: make them very visible by increasing opacity (note: your ply stores opacity already in [0,1])
    if args.set_opacity and "opacity" in vtx2.dtype.names:
        vtx2["opacity"][mask] = np.maximum(vtx2["opacity"][mask], 0.95)

    # Write out ply with same metadata
    write_ply_with_same_meta(ply, vtx2, args.out_ply)
    print("Wrote marked ply:", args.out_ply)


if __name__ == "__main__":
    main()
