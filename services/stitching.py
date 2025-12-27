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


def stitch_images(image_paths: List[str], output_path: str | None = None) -> str | np.ndarray:
    """
    OpenCV-based panorama stitching.

    Rules:
    - Input: list of image paths
    - Output:
        - if output_path is given -> writes file, returns output_path
        - else -> returns pano as np.ndarray
    """

    # --- Sanitize paths ---
    paths = [p for p in image_paths if isinstance(p, str) and os.path.exists(p)]
    if len(paths) < 2:
        raise ValueError("Need at least 2 images to stitch")

    imgs = [_read(p) for p in paths]

    # --- Resize for stability ---
    resized: list[np.ndarray] = []
    MAX_DIM = 1600

    for img in imgs:
        h, w = img.shape[:2]
        max_dim = max(h, w)

        if max_dim > MAX_DIM:
            scale = MAX_DIM / max_dim
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        resized.append(img)

    # --- Create stitcher ---
    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    else:
        stitcher = cv2.createStitcher(False)

    status, pano = stitcher.stitch(resized)

    # --- HARD SAFETY CHECKS (NumPy-safe) ---
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"OpenCV stitch failed with status={status}")

    if pano is None or not isinstance(pano, np.ndarray) or pano.size == 0:
        raise RuntimeError("OpenCV stitch returned empty panorama")

    # --- Aspect ratio sanity check ---
    h, w = pano.shape[:2]
    if w < h * 1.2:
        raise RuntimeError("Invalid panorama aspect ratio")

    # --- Output handling ---
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, pano)
        return output_path

    return pano
