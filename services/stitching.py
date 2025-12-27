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


def _crop_black_borders(img: np.ndarray) -> np.ndarray:
    """
    Removes black / empty borders caused by OpenCV warping.
    This is CRITICAL for 360 feeling (no fake edges).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # consider near-black as empty
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        raise RuntimeError("Panorama is empty after stitching")

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    if cropped.size == 0:
        raise RuntimeError("Invalid crop result")

    return cropped


def stitch_images(image_paths: List[str]) -> np.ndarray:
    """
    STRICT OpenCV panorama stitching for REAL panoramas (not collages).

    Rules:
    - Minimum 4 images
    - PANORAMA (spherical) mode
    - Fails hard on bad geometry
    - Crops warp artifacts (black borders)
    """

    paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(paths) < 4:
        raise ValueError("Need at least 4 images for reliable OpenCV stitching")

    imgs = [_read(p) for p in paths]

    # --- Resize for stability (important for memory & feature matching)
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

    # --- Create stitcher (spherical panorama assumptions)
    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    else:
        stitcher = cv2.createStitcher(False)

    # Make OpenCV stricter (avoid garbage panoramas)
    try:
        stitcher.setPanoConfidenceThresh(0.6)
    except Exception:
        pass

    status, pano = stitcher.stitch(resized)

    if status != cv2.Stitcher_OK or pano is None or pano.size == 0:
        raise RuntimeError(f"OpenCV stitch failed with status={status}")

    # --- Crop black borders caused by warping
    pano = _crop_black_borders(pano)

    # --- Sanity checks (kill fake panoramas)
    h, w = pano.shape[:2]

    # Panorama must be clearly horizontal
    if w < h * 1.5:
        raise RuntimeError(
            f"Invalid panorama aspect ratio (w={w}, h={h})"
        )

    # Reject tiny outputs (usually bad homography)
    if w < 800 or h < 300:
        raise RuntimeError(
            f"Panorama too small (w={w}, h={h})"
        )

    return pano
