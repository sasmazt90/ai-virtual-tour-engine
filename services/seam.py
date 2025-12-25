import numpy as np


def wrap_seam_blend(pano_bgr: np.ndarray, band: int = 96) -> np.ndarray:
    h, w = pano_bgr.shape[:2]
    band = max(8, min(band, w // 6))

    left = pano_bgr[:, :band].astype(np.float32)
    right = pano_bgr[:, w - band:].astype(np.float32)

    alpha = np.linspace(0, 1, band).reshape(1, band, 1)
    alpha = np.repeat(alpha, h, axis=0)

    pano_bgr[:, :band] = (right * (1 - alpha) + left * alpha).astype(np.uint8)
    pano_bgr[:, w - band:] = (right * alpha + left * (1 - alpha)).astype(np.uint8)

    return pano_bgr
