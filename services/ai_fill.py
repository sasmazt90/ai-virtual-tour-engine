from __future__ import annotations
import numpy as np
import cv2

def ai_fill_panorama(pano: np.ndarray) -> np.ndarray:
    h, w = pano.shape[:2]
    pad = int(w * 0.15)

    left = pano[:, :pad]
    right = pano[:, -pad:]

    blended = np.concatenate([right, pano, left], axis=1)
    blended = cv2.GaussianBlur(blended, (0, 0), sigmaX=1.2)
    return blended
