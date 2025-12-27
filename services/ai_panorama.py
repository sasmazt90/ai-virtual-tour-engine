# ai-virtual-tour-engine/services/ai_panorama.py
from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
import os

from services.seam import synthetic_panorama_from_single, make_seamless_horizontal
from services.ai_fill import ai_fill_panorama


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _resize_max(img: np.ndarray, max_dim: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _orb_homography(a: np.ndarray, b: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Try find homography mapping b -> a (so we warp b into a/canvas space).
    Returns (ok, H).
    """
    orb = cv2.ORB_create(nfeatures=2500)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return False, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(db, da, k=2)  # b->a
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 40:
        return False, None

    pts_b = np.float32([kb[m.queryIdx].pt for m in good])
    pts_a = np.float32([ka[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 4.0)
    if H is None:
        return False, None

    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < 30:
        return False, None

    return True, H


def _blend_over(canvas: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """
    Simple feather blend where warped has non-black pixels.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    m = (gray > 8).astype(np.float32)

    if m.sum() < 50:
        return canvas

    # Feather edges
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=3.0)
    m = np.clip(m, 0.0, 1.0)[..., None]

    out = (canvas.astype(np.float32) * (1.0 - m) + warped.astype(np.float32) * m)
    return out.astype(np.uint8)


def generate_ai_panorama(image_paths: List[str]) -> np.ndarray:
    """
    FAKE-360 panorama generator for non-ideal multi-view room photos.

    Goal:
    - Not a strict geometric panorama
    - Produces a wide "360-ish" strip
    - Uses best-effort ORB warp; if overlap is weak -> falls back to synthetic from single
    - Always runs ai_fill_panorama + seamless horizontal wrap

    This is exactly what you want for "AI boş açıları uydursun".
    """
    paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(paths) == 0:
        raise ValueError("No images for AI panorama")

    # If too few, just do a synthetic 360 from single best image
    if len(paths) == 1:
        img = _resize_max(_read(paths[0]))
        pano = synthetic_panorama_from_single(img)
        pano = ai_fill_panorama(pano)
        pano = make_seamless_horizontal(pano, blend_width=min(140, pano.shape[1] // 10))
        return pano

    # Read & resize
    imgs = [_resize_max(_read(p)) for p in paths]

    # Choose a base: median brightness + sharpness-ish
    scores = []
    for im in imgs:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(g, cv2.CV_32F).var()
        bright = float(g.mean())
        scores.append(sharp * 0.7 + bright * 0.3)
    base_idx = int(np.argmax(scores))
    base = imgs[base_idx]

    h0, w0 = base.shape[:2]
    canvas_h = h0
    canvas_w = int(w0 * 3.2)  # wide strip target
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place base in the middle
    cx = canvas_w // 2 - w0 // 2
    canvas[:, cx:cx + w0] = base

    # Warp others onto canvas using homography to base-ish space
    placed = 0
    for i, im in enumerate(imgs):
        if i == base_idx:
            continue

        ok, H = _orb_homography(base, im)
        if not ok:
            continue

        # We need to shift coords because base is at cx
        T = np.array([[1.0, 0.0, float(cx)],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        Hw = T @ H

        warped = cv2.warpPerspective(im, Hw, (canvas_w, canvas_h))
        canvas = _blend_over(canvas, warped)
        placed += 1

    # If we couldn't place enough, fallback to synthetic (prevents ugly collages)
    if placed < 1:
        pano = synthetic_panorama_from_single(base)
        pano = ai_fill_panorama(pano)
        pano = make_seamless_horizontal(pano, blend_width=min(140, pano.shape[1] // 10))
        return pano

    # Fill gaps + make loopable
    canvas = ai_fill_panorama(canvas)
    canvas = make_seamless_horizontal(canvas, blend_width=min(180, canvas.shape[1] // 12))
    return canvas
