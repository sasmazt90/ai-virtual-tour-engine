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
    OpenCV Stitcher ile panorama dener.
    - 4+ foto için hedeflenir.
    - Başarısız olursa exception fırlatır (pipeline fallback'a gider).
    """
    paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(paths) < 2:
        raise ValueError("Need at least 2 images to stitch")

    imgs = [_read(p) for p in paths]

    # OpenCV version compatibility
    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    else:
        stitcher = cv2.createStitcher(False)

    status, pano = stitcher.stitch(imgs)
    if status != 0 or pano is None or pano.size == 0:
        raise RuntimeError(f"OpenCV stitch failed with status={status}")

    return pano
