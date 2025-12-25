import cv2
import numpy as np
from typing import List


def _prep_for_stitch(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def stitch_panorama(images_bgr: List[np.ndarray], stitcher_mode: int, max_side: int) -> np.ndarray:
    imgs = [_prep_for_stitch(img, max_side) for img in images_bgr]

    stitcher = cv2.Stitcher_create(stitcher_mode)
    status, pano = stitcher.stitch(imgs)

    if status != cv2.Stitcher_OK or pano is None:
        raise RuntimeError(f"OpenCV stitching failed with status {status}")

    if pano.dtype != np.uint8:
        pano = pano.clip(0, 255).astype(np.uint8)

    return pano
