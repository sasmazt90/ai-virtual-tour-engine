from _future_ import annotations
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
    paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(paths) < 4:
        raise ValueError("Need at least 4 images")

    imgs = [_read(p) for p in paths]

    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    else:
        stitcher = cv2.createStitcher(False)

    status, pano = stitcher.stitch(imgs)
    if status != cv2.Stitcher_OK or pano is None:
        raise RuntimeError("OpenCV stitch failed")

    return pano
