from _future_ import annotations

import math
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


# =========================================================
# Core panorama builder
# =========================================================

def build_panorama(
    images_bgr: List[np.ndarray],
    *,
    try_scans_fallback: bool = True,
    inpaint_black: bool = True,
    enforce_360_seam: bool = True,
) -> np.ndarray:
    """
    Build panorama from a list of BGR images.

    Pipeline:
      1) Stitch (OpenCV Stitcher)
      2) Crop black borders
      3) Fill black holes (inpaint)
      4) 360 seam enforcement (wrap-aware seam selection + edge blending)

    Returns: panorama BGR image
    """
    if not images_bgr or len(images_bgr) < 2:
        raise ValueError("build_panorama requires at least 2 images")

    imgs = [_ensure_bgr_uint8(i) for i in images_bgr]
    pano = _stitch_opencv(imgs, try_scans_fallback=try_scans_fallback)

    pano = _crop_black_borders(pano)

    if inpaint_black:
        pano = _inpaint_black_regions(pano)

    pano = _crop_black_borders(pano)

    if enforce_360_seam:
        pano = _make_360_seamless(pano)

    return pano


# =========================================================
# Stitching
# =========================================================

def _stitch_opencv(images_bgr: List[np.ndarray], *, try_scans_fallback: bool) -> np.ndarray:
    """
    Uses OpenCV Stitcher in PANORAMA mode; optionally retries SCANS.
    """
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images_bgr)

    if status == cv2.Stitcher_OK and pano is not None:
        return pano

    if try_scans_fallback:
        stitcher2 = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        status2, pano2 = stitcher2.stitch(images_bgr)
        if status2 == cv2.Stitcher_OK and pano2 is not None:
            return pano2

    raise RuntimeError(f"OpenCV stitching failed (status={status}).")


# =========================================================
# Black border cropping
# =========================================================

def _crop_black_borders(img_bgr: np.ndarray, *, threshold: int = 8) -> np.ndarray:
    """
    Crops outer black borders by finding bounding box of non-black pixels.
    """
    if img_bgr is None:
        raise ValueError("img is None")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morph close to remove tiny holes
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return img_bgr  # nothing to crop

    x, y, w, h = cv2.boundingRect(coords)
    # safety
    x2 = min(x + w, img_bgr.shape[1])
    y2 = min(y + h, img_bgr.shape[0])
    cropped = img_bgr[y:y2, x:x2].copy()

    return cropped if cropped.size else img_bgr


# =========================================================
# Filling black holes (free/local)
# =========================================================

def _inpaint_black_regions(img_bgr: np.ndarray, *, black_threshold: int = 10) -> np.ndarray:
    """
    Fills internal black regions using OpenCV inpaint.
    Good for holes created by warping/stitching.

    black pixels: near (0,0,0) => inpaint mask
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # mask for near-black
    mask = (gray <= black_threshold).astype(np.uint8) * 255

    # clean mask: remove very thin noise, connect small gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # If mask is mostly black => do nothing
    black_ratio = float(np.count_nonzero(mask)) / float(h * w)
    if black_ratio < 0.001:
        return img_bgr

    # inpaint
    # TELEA generally looks more natural for texture continuation
    out = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return out


# =========================================================
# 360 Seam: left/right match
# =========================================================

def _make_360_seamless(
    pano_bgr: np.ndarray,
    *,
    search_steps: int = 48,
    overlap_ratio: float = 0.08,
    max_overlap_px: int = 240,
) -> np.ndarray:
    """
    Makes panorama more suitable for 360 by:
      1) Finding best horizontal roll so seam lands in lowest-difference area
      2) Blending left/right edges over an overlap strip

    Note: This is "best effort" for a generic stitched pano. True 360 capture
    works best when photos cover full rotation with consistent exposure.
    """
    pano = pano_bgr.copy()
    h, w = pano.shape[:2]

    # Choose overlap width
    overlap = int(min(max_overlap_px, max(32, w * overlap_ratio)))
    overlap = min(overlap, w // 4)

    # Downsample for seam cost search
    small = pano
    scale = 1.0
    if w > 1400:
        scale = 1400.0 / w
        small = cv2.resize(pano, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    best_shift = _find_best_seam_shift(small, steps=search_steps, overlap=int(overlap * scale))
    # Apply roll to full-res pano
    pano = np.roll(pano, shift=best_shift, axis=1)

    # Edge blend on full-res
    pano = _blend_left_right_edges(pano, overlap=overlap)

    return pano


def _find_best_seam_shift(img_bgr: np.ndarray, *, steps: int, overlap: int) -> int:
    """
    Find horizontal shift that minimizes difference between left and right edges.
    We evaluate multiple candidate shifts and pick smallest edge-SSD over overlap strip.
    """
    h, w = img_bgr.shape[:2]
    if overlap <= 0 or overlap >= w // 2:
        return 0

    # use luminance for faster cost
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # sample candidates
    # steps -> candidate shifts across width
    # include 0
    candidates = [int(round(i * w / steps)) for i in range(steps)]
    candidates = sorted(set(candidates))

    best_cost = float("inf")
    best_shift = 0

    # precompute left/right strips for each shifted version efficiently:
    # We'll roll and compute cost. On small image it’s fine.
    for s in candidates:
        rolled = np.roll(gray, shift=s, axis=1)

        left = rolled[:, :overlap]
        right = rolled[:, -overlap:]

        # compare left vs right (reverse right to align directionally)
        right_rev = right[:, ::-1]

        diff = left - right_rev
        cost = float(np.mean(diff * diff))

        if cost < best_cost:
            best_cost = cost
            best_shift = s

    return best_shift


def _blend_left_right_edges(img_bgr: np.ndarray, *, overlap: int) -> np.ndarray:
    """
    Blend left edge and right edge over overlap region.
    We blend left strip with mirrored right strip to reduce seam.
    """
    h, w = img_bgr.shape[:2]
    if overlap <= 0 or overlap >= w // 2:
        return img_bgr

    out = img_bgr.copy()

    left = out[:, :overlap].astype(np.float32)
    right = out[:, -overlap:].astype(np.float32)

    # mirror right to align features directionally
    right_m = right[:, ::-1]

    # alpha ramp 0..1 across overlap
    alpha = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[None, :, None]

    blended = (1.0 - alpha) * right_m + alpha * left
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # write back
    out[:, :overlap] = blended
    out[:, -overlap:] = blended[:, ::-1]  # un-mirror back

    return out


# =========================================================
# Utils
# =========================================================

def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img
