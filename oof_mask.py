import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def parse_ply_name_camera(ply_path: str):
    p = Path(ply_path)
    parts = p.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"PLY filename must be {{name}}_{{focal}}_{{W}}_{{H}}.ply, got: {p.name}")
    focal_s, w_s, h_s = parts[-3], parts[-2], parts[-1]
    if not (focal_s.isdigit() and w_s.isdigit() and h_s.isdigit()):
        raise ValueError(f"Last 3 tokens must be integers focal,W,H. Got {focal_s},{w_s},{h_s}")
    base = "_".join(parts[:-3]) or "model"
    return base, int(focal_s), int(w_s), int(h_s)


def build_K(focal_px: int, W: int, H: int):
    fx = float(focal_px)
    fy = float(focal_px)
    cx = (float(W) - 1.0) * 0.5
    cy = (float(H) - 1.0) * 0.5
    return fx, fy, cx, cy


def project_points_to_pixels(xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    valid = Z > 1e-6
    u = np.full((xyz.shape[0],), np.nan, dtype=np.float32)
    v = np.full((xyz.shape[0],), np.nan, dtype=np.float32)
    u[valid] = fx * (X[valid] / Z[valid]) + cx
    v[valid] = fy * (Y[valid] / Z[valid]) + cy
    return u, v, valid


def main():
    ap = argparse.ArgumentParser(description="Mark or drop points that project outside the image frame (out-of-frame).")
    ap.add_argument("--ply", required=True, help="Input PLY path (name_f_W_H.ply)")
    ap.add_argument("--out_ply", required=True, help="Output ply path")

    # actions
    ap.add_argument("--drop_oof", action="store_true", help="Drop out-of-frame points (remove them from the output ply).")
    ap.add_argument("--paint_black", action="store_true", help="Set f_dc to 0 for out-of-frame points")
    ap.add_argument("--set_opacity_zero", action="store_true", help="Set opacity=0 for out-of-frame points")

    ap.add_argument("--keep_invalid_z", action="store_true",
                    help="If set, do NOT treat Z<=0 as out-of-frame. (Default: invalid Z is out-of-frame.)")

    ap.add_argument("--chunk", type=int, default=300000, help="Chunk size for projection")
    ap.add_argument("--debug_image", type=str, default=None,
                    help="If set, write a debug image with projected inside-frame points in green.")
    args = ap.parse_args()

    # default action if none specified: drop
    if not args.drop_oof and not args.paint_black and not args.set_opacity_zero:
        args.drop_oof = True

    base, focal, W, H = parse_ply_name_camera(args.ply)
    fx, fy, cx, cy = build_K(focal, W, H)
    print(f"Parsed camera: focal={focal}px, size={W}x{H}, cx={cx:.2f}, cy={cy:.2f}")

    ply = PlyData.read(args.ply)
    vtx = ply["vertex"].data
    N = int(vtx.shape[0])
    print(f"PLY points: {N}")

    xyz = np.stack([np.asarray(vtx["x"], dtype=np.float32),
                    np.asarray(vtx["y"], dtype=np.float32),
                    np.asarray(vtx["z"], dtype=np.float32)], axis=1)

    selected = np.zeros((N,), dtype=bool)
    chunk = int(args.chunk)

    debug_canvas = None
    if args.debug_image is not None:
        debug_canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        u, v, valid = project_points_to_pixels(xyz[s:e], fx, fy, cx, cy)

        # default: invalid Z considered OOF
        if args.keep_invalid_z:
            sel = np.zeros((e - s,), dtype=bool)
        else:
            sel = ~valid

        if np.any(valid):
            ui = np.floor(u[valid]).astype(np.int32)
            vi = np.floor(v[valid]).astype(np.int32)
            in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

            # valid but outside
            sel_valid = ~in_img
            idx_valid = np.nonzero(valid)[0]
            sel[idx_valid] = sel_valid

            if debug_canvas is not None and np.any(in_img):
                ui_in = ui[in_img]
                vi_in = vi[in_img]
                debug_canvas[vi_in, ui_in] = np.array([0, 255, 0], dtype=np.uint8)

        selected[s:e] = sel

    nsel = int(selected.sum())
    print(f"Out-of-frame points: {nsel} / {N} ({nsel / max(1, N) * 100:.2f}%)")

    if args.drop_oof:
        keep = ~selected
        vtx_out = np.array(vtx[keep], copy=True)
        print(f"Dropped {nsel} points. Remaining: {int(vtx_out.shape[0])}")
    else:
        vtx_out = np.array(vtx, copy=True)
        if args.paint_black:
            if "f_dc_0" in vtx_out.dtype.names:
                vtx_out["f_dc_0"][selected] = 0.0
                vtx_out["f_dc_1"][selected] = 0.0
                vtx_out["f_dc_2"][selected] = 0.0
            else:
                print("Warning: no f_dc_0..2 found, skipping paint_black")

        if args.set_opacity_zero:
            if "opacity" in vtx_out.dtype.names:
                vtx_out["opacity"][selected] = 0.0
            else:
                print("Warning: no opacity found, skipping set_opacity_zero")

    el = PlyElement.describe(vtx_out, "vertex")
    PlyData([el], text=False).write(args.out_ply)
    print(f"Wrote: {args.out_ply}")

    if args.debug_image is not None:
        Image.fromarray(debug_canvas).save(args.debug_image)
        print(f"Wrote debug image: {args.debug_image} (green=projected inside frame)")


if __name__ == "__main__":
    main()
