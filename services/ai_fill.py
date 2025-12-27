# ai-virtual-tour-engine/services/ai_fill.py
from __future__ import annotations

import cv2
import numpy as np


def _make_hole_mask(img: np.ndarray) -> np.ndarray:
    """
    Detect "empty" / "invalid" areas to fill.
    Typical stitcher outputs have black borders / warped gaps.
    """
    if img is None or img.size == 0:
        raise ValueError("ai_fill_panorama: empty image")

    # Anything near-black is considered hole
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 8).astype(np.uint8) * 255

    # Also catch very low-saturation very dark pixels (optional)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask2 = ((v < 18) & (s < 25)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(mask, mask2)

    # Clean mask a bit
    k = max(3, (min(img.shape[:2]) // 200) | 1)  # odd kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def _edge_crop_to_content(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop excessive black borders if a big portion is empty.
    Keeps content region; avoids huge empty margins.
    """
    h, w = img.shape[:2]
    inv = cv2.bitwise_not(mask)
    coords = cv2.findNonZero(inv)
    if coords is None:
        return img

    x, y, ww, hh = cv2.boundingRect(coords)

    # Avoid over-cropping: keep at least 70% in each dimension
    if ww < int(0.70 * w) or hh < int(0.70 * h):
        return img

    cropped = img[y:y+hh, x:x+ww]
    return cropped


def _wrap_seam_and_inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    For "360-ish" panoramas: seam at left/right edges.
    We soften by padding with wrap-around then inpaint.
    """
    h, w = img.shape[:2]
    pad = max(32, min(256, w // 12))

    # Wrap padding
    left = img[:, :pad].copy()
    right = img[:, -pad:].copy()
    padded = np.concatenate([right, img, left], axis=1)

    # Expand mask similarly
    mleft = mask[:, :pad].copy()
    mright = mask[:, -pad:].copy()
    mpadded = np.concatenate([mright, mask, mleft], axis=1)

    # Inpaint on padded image
    radius = max(3, min(9, min(h, w) // 180))
    filled = cv2.inpaint(padded, mpadded, radius, cv2.INPAINT_TELEA)

    # Remove padding
    filled = filled[:, pad:pad+w]

    return filled


def ai_fill_panorama(img: np.ndarray) -> np.ndarray:
    """
    "AI-like" fill without external models:
    - Detect empty/black gaps
    - Crop huge borders if safe
    - Inpaint holes
    - Wrap seam left/right to avoid hard edge seam
    """
    if img is None or img.size == 0:
        raise ValueError("ai_fill_panorama: empty image")

    img = img.copy()
    mask = _make_hole_mask(img)

    # If mask is tiny, do nothing
    hole_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if hole_ratio < 0.002:
        return img

    # Crop huge borders if safe
    img2 = _edge_crop_to_content(img, mask)
    if img2.shape != img.shape:
        img = img2
        mask = _make_hole_mask(img)

    # Inpaint holes
    h, w = img.shape[:2]
    radius = max(3, min(9, min(h, w) // 180))
    filled = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)

    # Seam wrap + inpaint again (helps 360 looping feel)
    mask2 = _make_hole_mask(filled)
    filled = _wrap_seam_and_inpaint(filled, mask2)

    # Light smoothing to reduce artifacts
    filled = cv2.GaussianBlur(filled, (0, 0), sigmaX=0.6)

    return filled
