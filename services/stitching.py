# ai-virtual-tour-engine/services/stitching.py
from __future__ import annotations

from typing import List
import os
import cv2
import numpy as np


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def stitch_images(image_paths: List[str]) -> np.ndarray:
    """
    STRICT OpenCV panorama stitching for REAL panoramas (not collages).

    Rules:
    - Minimum 4 images (below that is unreliable for room panoramas)
    - Uses PANORAMA mode
    - Hard-fails on bad geometry (so pipeline can AI-fallback)
    """

    paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(paths) < 4:
        raise ValueError("Need at least 4 images for reliable OpenCV stitching")

    imgs = [_read(p) for p in paths]

    # --- Resize images for stability (VERY important)
    resized = []
    MAX_DIM = 1600
    for img in imgs:
        h, w = img.shape[:2]
        scale = MAX_DIM / max(h, w) if max(h, w) > MAX_DIM else 1.0
        if scale != 1.0:
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )
        resized.append(img)

    # --- Create stitcher (Panorama mode = spherical assumptions)
    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    else:
        stitcher = cv2.createStitcher(False)

    # Be more strict than OpenCV defaults
    try:
        stitcher.setPanoConfidenceThresh(0.6)
    except Exception:
        pass  # older OpenCV versions may not support this

    status, pano = stitcher.stitch(resized)

    if status != cv2.Stitcher_OK or pano is None or pano.size == 0:
        raise RuntimeError(f"OpenCV stitch failed with status={status}")

    # --- Sanity checks to avoid fake panoramas / wall-collages
    h, w = pano.shape[:2]

    # Panorama must be clearly wider than tall
    if w < h * 1.5:
        raise RuntimeError(
            f"Invalid panorama aspect ratio (w={w}, h={h})"
        )

    # Reject extremely small outputs (bad warp)
    if w < 800 or h < 300:
        raise RuntimeError(
            f"Panorama too small (w={w}, h={h})"
        )

    return pano
