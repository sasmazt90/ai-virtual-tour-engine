import cv2
import numpy as np


def make_seamless_horizontal(image_bgr: np.ndarray, blend_width: int = 80) -> np.ndarray:
    """
    Makes a panorama horizontally seamless (left/right edges match).
    Assumes BGR image (OpenCV format).
    """

    h, w, c = image_bgr.shape

    # Duplicate image horizontally
    double = np.concatenate([image_bgr, image_bgr], axis=1)

    # Create blending mask
    mask = np.zeros((h, w * 2, 3), dtype=np.float32)
    mask[:, :w] = 1.0

    # Linear fade on seam
    for i in range(blend_width):
        alpha = 1.0 - (i / blend_width)
        mask[:, w - i - 1] = alpha
        mask[:, w + i] = 1.0 - alpha

    blended = (double * mask + double[:, ::-1] * (1 - mask)).astype(np.uint8)

    # Crop back to original width
    seamless = blended[:, :w]

    return seamless
