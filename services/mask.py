import cv2
import numpy as np


def _morph_close(mask: np.ndarray, k: int = 7) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def compute_black_gap_mask(
    pano_bgr: np.ndarray,
    thresh: int,
    dilate_px: int
) -> np.ndarray:
    gray = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray <= thresh).astype(np.uint8) * 255

    hsv = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask2 = ((val <= thresh + 10) & (sat <= 20)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(mask, mask2)
    mask = _morph_close(mask, 7)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
        )
        mask = cv2.dilate(mask, kernel)

    return mask
