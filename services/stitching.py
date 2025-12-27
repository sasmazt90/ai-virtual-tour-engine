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
    OpenCV-based panorama stitching.
    This is a FALLBACK method – not the primary AI logic.
    """

    paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(paths) < 2:
        raise ValueError("Need at least 2 images to stitch")

    imgs = [_read(p) for p in paths]

    # Resize for stability
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

    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    else:
        stitcher = cv2.createStitcher(False)

    status, pano = stitcher.stitch(resized)

    if status != cv2.Stitcher_OK or pano is None or pano.size == 0:
        raise RuntimeError(f"OpenCV stitch failed with status={status}")

    h, w = pano.shape[:2]
    if w < h * 1.2:
        raise RuntimeError("Invalid panorama aspect ratio")

    return pano
