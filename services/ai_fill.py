# ai-virtual-tour-engine/services/ai_fill.py
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def _find_empty_mask(pano: np.ndarray) -> np.ndarray:
    """
    Detect empty / black regions in a stitched panorama.
    These usually come from warping gaps.
    """
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)

    # very dark pixels = empty
    mask = gray < 8

    # clean noise
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _simple_edge_fill(pano: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Deterministic fallback:
    Extend nearest valid pixels horizontally to fill gaps.
    This avoids broken output if AI is unavailable.
    """
    filled = pano.copy()
    h, w = pano.shape[:2]

    for y in range(h):
        row_mask = mask[y]
        if not row_mask.any():
            continue

        valid_x = np.where(row_mask == 0)[0]
        if len(valid_x) == 0:
            continue

        left = valid_x[0]
        right = valid_x[-1]

        # fill left gap
        for x in range(0, left):
            filled[y, x] = filled[y, left]

        # fill right gap
        for x in range(right + 1, w):
            filled[y, x] = filled[y, right]

    return filled


def ai_fill_panorama(pano: np.ndarray) -> np.ndarray:
    """
    Fills missing panorama regions using AI (or safe fallback).

    CONTRACT:
    - Does NOT hallucinate objects
    - Does NOT assume room type
    - Only extends visible surfaces logically
    """

    mask = _find_empty_mask(pano)

    # If no empty area → nothing to do
    if mask.sum() == 0:
        return pano

    # --- PLACEHOLDER FOR REAL AI MODEL ---
    # Here is where a real inpainting / diffusion / completion model
    # would be called using:
    #   - pano
    #   - mask
    #   - instruction: "continue visible surfaces realistically"
    #
    # Since we are currently model-free, we use a safe fallback.

    try:
        # Attempt OpenCV inpainting (fast, non-hallucinatory)
        inpainted = cv2.inpaint(
            pano,
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )
        return inpainted
    except Exception:
        # Final safety net (never break pipeline)
        return _simple_edge_fill(pano, mask)
