import argparse
import numpy as np
from PIL import Image


def to_gray_float01(img_rgb: np.ndarray) -> np.ndarray:
    # img_rgb: uint8 HWC
    x = img_rgb.astype(np.float32) / 255.0
    # Rec.709 luma
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def conv2d_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Simple 2D conv (same) for numpy arrays. x: HxW, k: khxkw"""
    H, W = x.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros((H, W), dtype=np.float32)
    # naive conv (ok for one image; not super fast but fine)
    for i in range(H):
        for j in range(W):
            patch = xp[i:i+kh, j:j+kw]
            out[i, j] = float(np.sum(patch * k))
    return out


def box_blur(x: np.ndarray, ksize: int) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    k = np.ones((ksize, ksize), dtype=np.float32) / float(ksize * ksize)
    return conv2d_same(x, k)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation with square structuring element radius r."""
    if r <= 0:
        return mask
    H, W = mask.shape
    rp = int(r)
    mp = np.pad(mask, ((rp, rp), (rp, rp)), mode="edge")
    out = np.zeros_like(mask, dtype=bool)
    for i in range(H):
        for j in range(W):
            out[i, j] = bool(np.any(mp[i:i+2*rp+1, j:j+2*rp+1]))
    return out


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion with square structuring element radius r."""
    if r <= 0:
        return mask
    H, W = mask.shape
    rp = int(r)
    mp = np.pad(mask, ((rp, rp), (rp, rp)), mode="edge")
    out = np.zeros_like(mask, dtype=bool)
    for i in range(H):
        for j in range(W):
            out[i, j] = bool(np.all(mp[i:i+2*rp+1, j:j+2*rp+1]))
    return out


def open_close(mask: np.ndarray, r_open: int, r_close: int) -> np.ndarray:
    # opening = erode -> dilate (remove small noise)
    m = dilate(erode(mask, r_open), r_open) if r_open > 0 else mask
    # closing = dilate -> erode (fill small holes)
    m = erode(dilate(m, r_close), r_close) if r_close > 0 else m
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image")
    ap.add_argument("--out_mask", default="smooth_mask.png", help="output binary mask (white=smooth)")
    ap.add_argument("--out_vis", default="smooth_vis.png", help="output visualization overlay")
    ap.add_argument("--resize_max", type=int, default=1200, help="resize long side for speed (0=disable)")
    ap.add_argument("--smooth_win", type=int, default=41, help="window size for smoothness averaging (odd, e.g. 31/41/61)")
    ap.add_argument("--thr", type=float, default=None,
                    help="threshold on averaged gradient (0..1). If not set, use percentile.")
    ap.add_argument("--thr_percentile", type=float, default=35.0,
                    help="if --thr not set, choose threshold as this percentile of smoothness map (lower=more strict)")
    ap.add_argument("--open_r", type=int, default=2, help="morph open radius")
    ap.add_argument("--close_r", type=int, default=6, help="morph close radius")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = img.size

    # optional resize for speed/robustness
    if args.resize_max and max(orig_w, orig_h) > args.resize_max:
        scale = args.resize_max / float(max(orig_w, orig_h))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img_small = img.resize((new_w, new_h), Image.BILINEAR)
    else:
        img_small = img

    rgb = np.array(img_small, dtype=np.uint8)
    gray = to_gray_float01(rgb).astype(np.float32)

    # Sobel gradients
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / 8.0
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32) / 8.0

    gx = conv2d_same(gray, kx)
    gy = conv2d_same(gray, ky)
    grad = np.sqrt(gx * gx + gy * gy)

    # averaged edge strength (lower -> smoother)
    smoothness = box_blur(grad, args.smooth_win)

    # choose threshold
    if args.thr is None:
        thr = float(np.percentile(smoothness, args.thr_percentile))
    else:
        thr = float(args.thr)

    mask = smoothness < thr

    # morph cleanup
    mask = open_close(mask, args.open_r, args.close_r)

    # upscale mask back to original size
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
    mask_np = (np.array(mask_img) > 127)

    # save binary mask
    Image.fromarray(mask_np.astype(np.uint8) * 255).save(args.out_mask)

    # visualization: dim non-smooth, keep smooth
    vis = np.array(img, dtype=np.uint8).copy()
    dim = (vis.astype(np.float32) * 0.35).astype(np.uint8)
    vis[~mask_np] = dim[~mask_np]
    Image.fromarray(vis).save(args.out_vis)

    print("Wrote:", args.out_mask)
    print("Wrote:", args.out_vis)
    print(f"Smooth pixels ratio: {mask_np.mean():.3f}")
    print(f"Used threshold: {thr:.6f} (on resized image domain)")


if __name__ == "__main__":
    main()
