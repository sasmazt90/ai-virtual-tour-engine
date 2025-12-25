from typing import List
import cv2
import numpy as np


def stitch_images(images_bgr: List[np.ndarray]) -> np.ndarray:
    """
    Low-level stitching helper
    """
    if len(images_bgr) < 2:
        raise ValueError("Need at least 2 images to stitch")

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images_bgr)

    if status != cv2.Stitcher_OK or pano is None:
        raise RuntimeError(f"OpenCV stitching failed (status={status})")

    return pano


def stitch_images_to_panorama(images_bgr: List[np.ndarray]) -> np.ndarray:
    """
    Backward-compatible alias used by app.py
    """
    return stitch_images(images_bgr)
